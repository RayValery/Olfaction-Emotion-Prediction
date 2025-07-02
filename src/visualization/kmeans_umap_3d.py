import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
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

# === Prepare X ===
label_columns = [col for col in df_merged.columns if col.endswith("_label")]
X = df_merged.drop(columns=["CID"] + label_columns)

# Синхронне прибирання NaN
mask = X.notna().all(axis=1)
X = X[mask]

# === UMAP 3D ===
X_scaled = StandardScaler().fit_transform(X)
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

# === KMeans clustering ===
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_umap)

# === Plot 3D with cluster coloring ===
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

for cl in range(n_clusters):
    idx = cluster_labels == cl
    ax.scatter(X_umap[idx, 0], X_umap[idx, 1], X_umap[idx, 2],
               label=f"Cluster {cl}", alpha=0.7, s=50, c=[colors[cl]])

ax.set_title(f"3D UMAP + KMeans clustering (n={n_clusters})")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.set_zlabel("UMAP-3")
ax.legend(loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
