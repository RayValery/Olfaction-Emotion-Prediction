import numpy as np
from sklearn.metrics import f1_score

def tune_thresholds(y_true, y_pred_proba, label_names, metric="f1", step=0.05):
    thresholds = {}

    for i, label in enumerate(label_names):
        best_threshold = 0.5
        best_score = 0.0

        for t in np.arange(0.05, 0.95, step):
            y_pred = (y_pred_proba[:,i] >= t).astype(int)

            score = 0.0
            if metric == "f1":
                score = f1_score(y_true[:,i], y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = t

        thresholds[label] = best_threshold
        print(f"{label}: best threshold = {best_threshold:.2f}, best {metric} = {best_score:.3f}")

    return thresholds