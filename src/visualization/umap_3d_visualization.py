import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# === Load data ===
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

df_train = pd.read_csv(DATA_DIR / "TrainSet.txt", sep="\t")
df_desc = pd.read_csv(DATA_DIR / "molecular_descriptors_data.txt", sep="\t")

# === Filter only "high" intensity ===
df_high = df_train[df_train["Intensity"].str.strip() == "high"]

# === Compute average odor ratings ===
odor_labels = ["SWEET", "FRUIT", "FISH", "GARLIC", "SPICES", "COLD", "SOUR",
               "BURNT", "ACID", "WARM", "MUSKY", "SWEATY", "AMMONIA/URINOUS",
               "DECAYED", "WOOD", "GRASS", "FLOWER", "CHEMICAL", "BAKERY"]
df_targets = df_high.groupby("Compound Identifier")[odor_labels].mean().reset_index()
df_targets.columns = ["CID"] + odor_labels

# === Binarize odor descriptors ===
for label in odor_labels:
    threshold = df_targets[label].quantile(0.65)
    df_targets[label + "_label"] = (df_targets[label] > threshold).astype(int)

# === Merge with molecular features ===
df_targets["CID"] = df_targets["CID"].astype(int)
df_desc["CID"] = df_desc["CID"].astype(int)
df_merged = pd.merge(df_targets, df_desc, on="CID")

# === Prepare X and Y ===
label_columns = [col for col in df_merged.columns if col.endswith("_label")]
X = df_merged.drop(columns=["CID"] + label_columns)
y = df_merged[label_columns]

# Drop NaN rows in sync
mask = X.notna().all(axis=1)
X = X[mask]
y = y[mask]

# === Assign dominant label per molecule ===
dominant_label = y.apply(lambda row: row[row == 1].index.tolist(), axis=1)
dominant_label = dominant_label.apply(lambda lst: lst[0] if lst else "none")

# === UMAP 3D projection ===
X_scaled = StandardScaler().fit_transform(X)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# === Plot 3D ===
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

unique_labels = dominant_label.unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    idx = dominant_label == label
    ax.scatter(X_umap[idx, 0], X_umap[idx, 1], X_umap[idx, 2],
               label=label.replace("_label", ""), alpha=0.7, s=50, c=[color])

ax.set_title("3D UMAP of odor space by dominant label")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.set_zlabel("UMAP-3")
ax.legend(loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
