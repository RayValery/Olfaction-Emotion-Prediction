import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === 1. Load training data ===
# Contains perceptual responses from multiple human raters
df_train = pd.read_csv("data/TrainSet.txt", sep="\t")

# Contains molecular descriptors for each compound (over 4000 features)
df_desc = pd.read_csv("data/molecular_descriptors_data.txt", sep="\t")

# === 2. Filter for "high" concentration samples only ===
# Low concentrations may not evoke full perceptual profiles
df_high = df_train[df_train["Intensity"].str.strip() == "high"]

# === 3. Aggregate perceptual ratings across subjects per compound ===
# We'll average across human responses to get a stable signal per odor
descriptors = ["SWEET", "FRUIT", "MUSKY", "FISH", "FLOWER", "BURNT", "SOUR", "GARLIC"]
df_targets = df_high.groupby("Compound Identifier")[descriptors].mean().reset_index()
df_targets.columns = ["CID"] + descriptors


# === 4. Define binary label for classification ===
# Here we use a dynamic threshold (e.g., top 25% SWEET scores)
sweet_threshold = df_targets["SWEET"].quantile(0.55)
df_targets["sweet_label"] = (df_targets["SWEET"] > sweet_threshold).astype(int)


df_targets["CID"] = df_targets["CID"].astype(int)
df_desc["CID"] = df_desc["CID"].astype(int)

# === 5. Join with molecular descriptors ===
df_merged = pd.merge(df_targets, df_desc, on="CID")


# === 6. Select features (for now, use last 100 molecular descriptors)
# Later we can apply feature selection / PCA
X = df_merged.iloc[:, -100:]
y = df_merged["sweet_label"]

# === 7. Train-test split and model training ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

# === 8. Evaluate performance ===
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
