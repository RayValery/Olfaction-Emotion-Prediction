from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, f1_score, precision_score, \
    recall_score, hamming_loss, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from Oversampling import oversample_weak_labels

# 2. Завантаження даних
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

df_train = pd.read_csv(DATA_DIR / "TrainSet.txt", sep="\t")
df_desc = pd.read_csv(DATA_DIR / "molecular_descriptors_data.txt", sep="\t")

# 3. Перевірка даних
# print(df_train.head)
# print(df_train.info)
# print(df_train.describe())
# print(df_train.isnull().sum())

odor_labels = ["BAKERY", "SWEET", "FRUIT", "FISH", "GARLIC", "SPICES", "COLD", "SOUR",
               "BURNT", "ACID", "WARM", "MUSKY", "SWEATY", "AMMONIA/URINOUS", "DECAYED",
               "WOOD", "GRASS", "FLOWER", "CHEMICAL"]

#============================ Single label baseline: ===================================
#
# df_grouped = df_train.groupby("Compound Identifier")[odor_labels].sum().reset_index()
# df_grouped["Target odor"] = df_grouped[odor_labels].idxmax(axis=1)  # create new column Target odor and store the greatest odor value for this CID
#
# df_grouped.rename(columns={"Compound Identifier": "CID"}, inplace=True)
# df_grouped = df_grouped[["CID", "Target odor"]] # drop all columns except CID and Target odor
#
# df_merged = pd.merge(df_grouped, df_desc, on="CID")
#========================================================================================
# df_train = df_train[df_train["Intensity"].str.strip() == "high"]

df_grouped = df_train.groupby("Compound Identifier")[odor_labels].sum().reset_index()
df_grouped.columns = ["CID"] + odor_labels

df_grouped["Total votes"] = df_grouped[odor_labels].sum(axis=1)

for label in odor_labels:
    df_grouped[label] = (df_grouped[label] / df_grouped["Total votes"] >= 0.05).astype(int)

# print(df_grouped.head)
# for label in odor_labels:
#     counts = df_grouped[label].value_counts()
#     print(f"{label}:\n{counts}\n")


df_merged = pd.merge(df_grouped, df_desc, on="CID")

# 4. EDA
# print(df_merged.shape)
# print(df_merged.isnull().sum())

# plt.figure(figsize=(12,10))
# sns.heatmap(df_grouped[odor_labels].corr(), annot=True, cmap="viridis", cbar=False)
# plt.title("Correlation between odors")
# plt.show()

# 5. Препроцесинг
# перевіримо пропуски: емає пропусків → нічого робити не треба
# print(df_merged.isnull().sum())

# 6. Вибір features & target
X = df_merged.drop(columns=odor_labels + ["CID"])
y = df_merged[odor_labels]

# 7. Розбиття на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.columns = X_train.columns.astype(str).str.replace(r"[<>\[\]]", " ", regex=True)
X_test.columns = X_test.columns.astype(str).str.replace(r"[<>\[\]]", " ", regex=True)

# print("Before oversampling:")
# for label in y_train.columns:
#     print(label)
#     print(y_train[label].value_counts())
# print("Before:", X_train.shape, y_train.shape)

# X_resampled, y_resampled = oversample_weak_labels(X_train, y_train, min_pos_threshold=0.3, target_pos_count=100)

# print("\nAfter oversampling:")
# for label in y_resampled.columns:
#     print(label)
#     print(y_resampled[label].value_counts())
# print("After:", X_resampled.shape, y_resampled.shape)

# 9. Вибір моделі
model = OneVsRestClassifier(
    XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        eval_metric="logloss",      # інакше буде warning
        scale_pos_weight=5,         # допомагає з дисбалансом класів
        random_state=42
    )
)

# 10. Навчання
model.fit(X_train, y_train)

# 11. Передбачення
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 12. Оцінка
for i, label in enumerate(y.columns):
    print(f"\n===== {label} =====")
    print(classification_report(y_test[label], y_pred[:, i]))

metrics = {
    "F1-score (macro)": f1_score(y_test, y_pred, average="macro"),
    "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
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