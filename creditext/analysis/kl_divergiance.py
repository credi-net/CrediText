import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def error_entropy(error_dict):

    # Example: error bins and counts
    counts_lst =[np.array([0,0,0,0,0,21]),
                np.array([1,2,3,4,5,6]),
                np.array([6,5,4,3,2,1]),
                np.array([6,5,4,1,2,3]),
                np.array([6,4,2,1,3,5])]
    for counts in counts_lst:
        print(f"error area={sum([elem*idx for idx,elem in enumerate(counts)])/((len(counts)-1)*sum(counts))}")
        error_bins = np.array([elem*0.1 for elem in range(0,len(counts))])
        p = counts / counts.sum()
        # Mean error
        mean_error = np.sum(p * error_bins)
        # Standard deviation of error
        std_error = np.sqrt(np.sum(p * (error_bins - mean_error)**2))
        print(f"Mean error: {mean_error:.4f}")
        print(f"Error spread (std): {std_error:.4f}")

def error_entropy_norm():
    counts_lst =[np.array([1,2,3,4,5,6]),
                np.array([6,5,4,3,2,1]),
                np.array([6,5,4,1,2,3]),
                np.array([6,4,2,1,3,5])]
    for counts in counts_lst:
      
        K = len(counts)
        # Convert counts to probabilities
        p = counts / counts.sum()

        # Remove zero-probability bins
        p_nonzero = p[p > 0]

        # Raw entropy (nats)
        H = -np.sum(p_nonzero * np.log(p_nonzero))

        # Maximum possible entropy
        H_max = np.log(K)  # natural log; use np.log2(K) for bits

        # Normalized entropy [0,1]
        H_norm = H / H_max
        print(f"Normalized entropy: {H_norm:.4f}")

# error_entropy(None)
# error_entropy_norm()


num_classes = 10
eps = 1e-9
data_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediGraph/plots"
files_lst=["dqr_nov_pc1_sklearn_text_embeddingTE3L_GAT-text-avg_run0_agg_dqr_pred.csv",
          "dqr_dec_pc1_sklearn_text_embeddingTE3L__run0_dqr_pred.csv"]



for file_path in files_lst:
    print(f"file_path={file_path}")
    true_pred_df=pd.read_csv(f"{data_path}/{file_path}")
    true_pred_df["diff"]=true_pred_df.apply(lambda row: abs(row["true"]-row["pred"]),axis=1)
    true_pred_df["diff_norm"]=true_pred_df["diff"].apply(lambda x: int(round(x*100)))
    vc=true_pred_df["diff_norm"].value_counts()
    # 3. Plot the value counts (e.g., as a bar chart)
    vc.plot(kind='bar', color='skyblue')
    # 4. Add labels and title for clarity
    plt.xlabel('Error Pin')
    plt.ylabel('Count')
    plt.title('Error Values Dist')
    # 5. Display the plot
    plt.savefig(f"{data_path}/{"Error_freq_"+file_path.split(".csv")[0]+".pdf"}",bbox_inches='tight', pad_inches=0.1)
    counts=vc.value_counts().sort_index().values
    print(f"error area={sum([elem*idx for idx,elem in enumerate(counts)])/((len(counts)-1)*sum(counts))}")
    # print(f"error_entropy={error_entropy(counts.values)}")
    # y_true=true_pred_df["true"].tolist()
    # y_pred=true_pred_df["pred"].tolist()
    # p = np.bincount(y_true, minlength=num_classes).astype(float)
    # q = np.bincount(y_pred, minlength=num_classes).astype(float)
    # p = (p + eps) / (p.sum() + eps * num_classes)
    # q = (q + eps) / (q.sum() + eps * num_classes)
    # kl = entropy(p, q)                 # KL(P || Q)
    # js = jensenshannon(p, q) ** 2       # Jensen–Shannon divergence
    # l1 = np.sum(np.abs(p - q))
    # l2 = np.linalg.norm(p - q)
    # print(f"kl={kl}\tjs={js}\tl1={l1}\tl2={l2}")