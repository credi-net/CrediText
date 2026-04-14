# halo.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HALOModel(nn.Module):
    """A wrapper class for extracting sample embeddings and managing class-centroids."""

    def __init__(self, model, n_classes, embedding_dim):
        super(HALOModel, self).__init__()
        self.model = model
        self.n_classes = n_classes
        self.emb_dims = embedding_dim
        self.centroids = nn.Parameter(
            torch.randn(n_classes, embedding_dim, dtype=torch.float32)
        )

    def forward(self, x):
        embeddings = self.model(x)
        centroids = self.centroids
        centroids = centroids - centroids.mean(dim=0, keepdim=True)
        return embeddings, centroids
    def predict(self, x):
        return self.forward(torch.tensor(x).float()).argmax(dim=-1) 


class HALOLoss(torch.nn.Module):
    def __init__(
        self,
        emb_dims,
        num_classes,
        learn_gamma=True,
        distill=True,
        label_smoothing=0.1,
        reduction="mean",
    ):
        super().__init__()
        assert emb_dims > 1, "Embedding dimensions must be > 1"
        self.D = float(emb_dims)
        self.K = float(num_classes)
        self.learn_gamma = learn_gamma
        self.distill = distill
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        r_sq_target = 1.0 - (2.0 / self.D)

        r_sq_init = 2.0
        init_gamma = 20.0 / (r_sq_init - r_sq_target)

        # unnormalized abstain bias
        if label_smoothing > 0:
            # Exact expected target probabilities strictly over K classes
            max_prob = 1.0 - label_smoothing + (label_smoothing / self.K)
            min_prob = label_smoothing / self.K
        else:
            max_prob = 0.99
            min_prob = 0.01 / self.K

        margin_ce = math.log(max_prob / min_prob)
        t_ideal = init_gamma * (1.0 - r_sq_target)
        self.abstain_bias = t_ideal - margin_ce

        # inverse softplus
        if init_gamma > 20.0:
            gamma_start = init_gamma
        else:
            gamma_start = math.log(math.expm1(init_gamma))

        self.gamma = nn.Parameter(
            torch.tensor([gamma_start], dtype=torch.float32),
            requires_grad=learn_gamma,
        )

    def forward(self, pos, target, centroids, centroid_targets):
        pos = pos.to(torch.float32)
        cen = centroids.to(torch.float32)

        x_sq = pos.pow(2).mean(dim=-1, keepdim=True)
        y_sq = cen.pow(2).mean(dim=-1, keepdim=True)
        dot_product = (pos @ cen.T) / self.D

        gamma = F.softplus(self.gamma)

        # Softmax is shift-invariant, so we factor out -(x_sq * gamma).
        # This leaves standard dot-product similarity with an L2 penalty on keys!
        logits_k_shifted = gamma * (2.0 * dot_product - y_sq.T)

        # The Abstain class acts as an origin sink using the theoretically ideal equilibrium.
        logit_abstain_shifted = torch.full(
            (pos.size(0), 1), self.abstain_bias, dtype=pos.dtype, device=pos.device
        )

        # Shape: N x K+1
        logits_k_plus_1 = torch.cat([logits_k_shifted, logit_abstain_shifted], dim=-1)

        # Reconstruct true absolute distances strictly for Distillation & Return Values
        # (Clamping against float errors)
        logits_k_true = torch.clamp(logits_k_shifted - (gamma * x_sq), max=0.0)

        # cross_entropy on k+1 calsses
        if self.distill:
            mask = target.unsqueeze(1) == centroid_targets.unsqueeze(0)
            with torch.no_grad():
                # Margin calculation uses the absolute clamped distances
                margin = logits_k_true / self.label_smoothing
                target_logits = torch.where(mask, 0.0, margin)
                target_probs_k = F.softmax(target_logits, dim=-1)

                zeros = torch.zeros(
                    (pos.size(0), 1), device=pos.device, dtype=pos.dtype
                )
                target_probs = torch.cat([target_probs_k, zeros], dim=-1)

            loss_ce = F.cross_entropy(
                logits_k_plus_1,  # Uses the shifted, un-clamped logits for smooth gradients!
                target_probs,
                reduction=self.reduction,
            )
        else:
            # Labels smoothing with explicit abstain class handling
            with torch.no_grad():
                K = logits_k_shifted.size(1)
                target_probs = torch.full_like(
                    logits_k_plus_1,
                    self.label_smoothing / K,
                    dtype=pos.dtype,
                    device=pos.device,
                )
                target_probs.scatter_(
                    1,
                    target.unsqueeze(1),
                    1.0 - self.label_smoothing + (self.label_smoothing / K),
                )
                # set the abstain class probability to strict 0.0
                target_probs[:, -1] = 0.0

            loss_ce = F.cross_entropy(
                logits_k_plus_1,
                target_probs,
                reduction=self.reduction,
            )

        # Geometric Regularizer
        diff_true = pos - cen[target]
        r_sq_true = diff_true.pow(2).mean(dim=-1).to(pos.dtype)

        volume_coeff = 0.5 - 1.0 / self.D
        volume_term = volume_coeff * torch.log(r_sq_true)
        gaussian_term = -0.5 * r_sq_true
        radial_nll = -(volume_term + gaussian_term)

        if self.reduction == "mean":
            radial_nll = radial_nll.mean()
        elif self.reduction == "none":
            pass
        else:
            raise NotImplementedError

        total_loss = loss_ce + radial_nll

        return total_loss, logits_k_true