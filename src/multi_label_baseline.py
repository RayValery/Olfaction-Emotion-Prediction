import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, hamming_loss, \
    accuracy_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

df_train = pd.read_csv(DATA_DIR / "TrainSet.txt", sep="\t")
df_desc = pd.read_csv(DATA_DIR / "molecular_descriptors_data.txt", sep="\t")

df_high = df_train[df_train["Intensity"].str.strip() == "high"]

odor_labels = ["BAKERY", "SWEET", "FRUIT", "FISH", "GARLIC", "SPICES", "COLD", "SOUR",
               "BURNT", "ACID", "WARM", "MUSKY", "SWEATY", "AMMONIA/URINOUS", "DECAYED",
               "WOOD", "GRASS", "FLOWER", "CHEMICAL"]
df_targets = df_high.groupby("Compound Identifier")[odor_labels].mean().reset_index()
df_targets.columns = ["CID"] + odor_labels

for label in odor_labels:
    threshold = df_targets[label].quantile(0.65)
    df_targets[label + "_label"] = (df_targets[label] > threshold).astype(int)

df_targets["CID"] = df_targets["CID"].astype(int)
df_desc["CID"] = df_desc["CID"].astype(int)
df_merged = pd.merge(df_targets, df_desc, on="CID")

label_columns = [col for col in df_merged.columns if col.endswith("_label")]
X = df_merged.drop(columns=label_columns + ["CID"])
y = df_merged[label_columns]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

for i, label in enumerate(y.columns):
    print(f"\n===== {label} =====")
    print(classification_report(y_test[label], y_pred[:, i]))

metrics = {
    "F1-score (macro)": f1_score(y_test, y_pred, average="macro"),
    "Precision (macro)": precision_score(y_test, y_pred, average="macro"),
    "Recall (macro)": recall_score(y_test, y_pred, average="macro"),
    "Hamming Loss": hamming_loss(y_test, y_pred),
    "Subset Accuracy": accuracy_score(y_test, y_pred)
}

try:
    roc_auc = roc_auc_score(y_test, y_pred_proba, average="macro")
    metrics["ROC AUC (macro)"] = roc_auc
except:
    metrics["ROC AUC (macro)"] = "Cannot calculate — need probabilities"

# Вивести в табличці
metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
print(metrics_df)

