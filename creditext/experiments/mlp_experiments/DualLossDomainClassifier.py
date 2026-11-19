import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_metric_learning import losses

class DualLossDomainClassifier(nn.Module):
    def __init__(self, input_dim=2304, hidden_dim=512, proj_dim=128):
        super().__init__()
        
        # Shared Encoder: Learns the core representation
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Head 1: Projection Head for Metric Learning (SupCon)
        # It is critical to use a non-linear projection head rather than 
        # applying SupCon directly to the classification features.
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        # Head 2: Classification Head for Cross Entropy
        self.classification_head = nn.Linear(hidden_dim, 1) # Binary output
        
    def forward(self, x):
        shared_features = self.shared_encoder(x)
        
        # Project and L2 normalize for contrastive loss
        projections = self.projection_head(shared_features)
        normalized_projections = F.normalize(projections, p=2, dim=1)
        
        # Generate classification logits
        logits = self.classification_head(shared_features)
        
        return normalized_projections, logits


# Initialize model, optimizer, and loss functions
model = DualLossDomainClassifier()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Binary Cross Entropy with Logits for the classification head
criterion_ce = nn.BCEWithLogitsLoss()

# Supervised Contrastive Loss for the projection head
# Temperature (\tau) controls the penalty for hard negatives (0.07 is a standard default)
criterion_supcon = losses.SupConLoss(temperature=0.07)

# Weighting hyperparameter to balance the two losses
alpha = 0.5 

# --- Training Loop Step ---
def train_step(batch_embeddings, labels):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    projections, logits = model(batch_embeddings)
    
    # 1. Calculate Supervised Contrastive Loss
    # Pulls domains of the same class together, pushes different classes apart
    loss_supcon = criterion_supcon(projections, labels)
    
    # 2. Calculate Binary Cross Entropy Loss
    loss_ce = criterion_ce(logits.squeeze(), labels.float())
    
    # 3. Combine Losses
    total_loss = (alpha * loss_supcon) + ((1 - alpha) * loss_ce)
    
    # Backpropagation
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), loss_supcon.item(), loss_ce.item() 