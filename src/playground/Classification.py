from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor

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

df_grouped = df_train.groupby("Compound Identifier")[odor_labels].sum().reset_index()
df_grouped.columns = ["CID"] + odor_labels

df_grouped["Total votes"] = df_grouped[odor_labels].sum(axis=1)

for label in odor_labels:
    df_grouped[label] = (df_grouped[label] / df_grouped["Total votes"] >= 0.05).astype(int)

print(df_grouped.head)
for label in odor_labels:
    counts = df_grouped[label].value_counts()
    print(f"{label}:\n{counts}\n")


df_merged = pd.merge(df_grouped, df_desc, on="CID")

# 4. EDA
# print(df_merged.shape)
# print(df_merged["Target odor"].value_counts())
# print(df_merged.isnull().sum())