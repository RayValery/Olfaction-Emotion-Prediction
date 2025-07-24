from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

from ThresholdTuner import tune_thresholds

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

df_train = pd.read_csv(DATA_DIR / "TrainSet.txt", sep="\t")
df_desc = pd.read_csv(DATA_DIR / "molecular_descriptors_data.txt", sep="\t")

odor_labels = ["BAKERY", "SWEET", "FRUIT", "FISH", "GARLIC", "SPICES", "COLD", "SOUR",
               "BURNT", "ACID", "WARM", "MUSKY", "SWEATY", "AMMONIA/URINOUS", "DECAYED",
               "WOOD", "GRASS", "FLOWER", "CHEMICAL"]

df_grouped = df_train.groupby("Compound Identifier")[odor_labels].sum().reset_index()
df_grouped.columns = ["CID"] + odor_labels

df_grouped["Total votes"] = df_grouped[odor_labels].sum(axis=1)

for label in odor_labels:
    df_grouped[label] = (df_grouped[label] / df_grouped["Total votes"] >= 0.05).astype(int)

df_merged = pd.merge(df_grouped, df_desc, on="CID")

X = df_merged.drop(columns=odor_labels + ["CID"])
y = df_merged[odor_labels]

# Розбиття на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.columns = X_train.columns.astype(str).str.replace(r"[<>\[\]]", " ", regex=True)
X_test.columns = X_test.columns.astype(str).str.replace(r"[<>\[\]]", " ", regex=True)

X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())  # ⚠️ використовуй mean з train!

#======================================================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 1e-3
batch_size = 64
num_epochs = 30

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
assert not np.isnan(X_train_scaled).any(), "NaN in X_train"
assert not np.isinf(X_train_scaled).any(), "Inf in X_train"

# Перетвори NumPy / Pandas у тензори
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)  # для multi-label класифікації — float32

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Створи датасети
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Створи DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Create model
class SmellClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SmellClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # бо мульти-лейбл
        )

    def forward(self, x):
        return self.model(x)

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
model = SmellClassifier(input_dim, output_dim)
model = model.to(device)

# Initialize the loss function and optimizer
los_fun = nn.BCELoss()  # Binary Cross-Entropy Loss
                        # Це найкраща функція втрат для задачі мульти-лейбл класифікації, де кожен з 19 запахів —
                        # незалежний бінарний клас (є / нема).
                        # У тебе на виході стоїть Sigmoid, а не Softmax, отже, кожен запах класифікується окремо.
                        # BCELoss якраз розраховує втрату по кожному класу незалежно.
                        # Якщо би це була multi-class задача (тільки один запах на приклад), тоді краще було б CrossEntropyLoss,
                        # але в multi-label — саме BCELoss або BCEWithLogitsLoss.

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)      # Adam — адаптивний оптимізатор, який автоматично змінює learning rate
                                                                        # для кожного параметра. Він:
                                                                        # 📉 швидше сходиться, ніж класичний SGD
                                                                        # 📈 стабільніший у задачах з високою кількістю фіч (4800+)
                                                                        # ❌ не потребує сильного тюнінгу на початку

def train_loop(train_dataLoader, model, los_fun, optimizer, device):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_dataLoader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Compute prediction and loss
        pred = model(X_batch)
        print(pred.min().item(), pred.max().item())
        loss = los_fun(pred, y_batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(train_dataLoader.dataset)

def evaluate_with_thresholds_loop(test_dataLoader, model, los_fun, label_names, device):
    model.eval()
    running_loss = 0.0
    all_pred = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_dataLoader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Compute prediction and loss
            pred = model(X_batch)
            loss = los_fun(pred, y_batch)

            running_loss += loss.item() * X_batch.size(0)

            all_pred.append(pred.cpu())
            all_targets.append(y_batch.cpu())

    y_true = torch.cat(all_targets).numpy()
    y_pred_proba = torch.cat(all_pred).numpy()

    # Перетворення ймовірностей у бінарні лейбли
    thresholds = tune_thresholds(y_true, y_pred_proba, label_names)
    y_pred = np.zeros_like(y_pred_proba, dtype=int)
    for i, label in enumerate(label_names):
        y_pred[:, i] = (y_pred_proba[:, i] >= thresholds[label]).astype(int)

    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "y_pred_proba": y_pred_proba,
        "thresholds": thresholds,
        "loss": running_loss / len(test_dataLoader.dataset)
    }

for epoch in range(num_epochs):
    train_loss = train_loop(train_dataloader, model, los_fun, optimizer, device)
    results = evaluate_with_thresholds_loop(test_dataloader, model, los_fun, odor_labels, device)

    # Метрики
    print("\n=== Classification Report (with tuned thresholds) ===")
    print(classification_report(results["y_true"], results["y_pred"], target_names=odor_labels))

