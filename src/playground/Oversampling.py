import pandas as pd
import numpy as np

def oversample_weak_labels(X, y, min_pos_threshold=0.2, target_pos_count=50, random_state=42):
    """
        Oversample samples with rare (positive) labels in multilabel classification.

        Parameters:
        - X: pd.DataFrame — features
        - y: pd.DataFrame — multilabel targets (binary columns per class)
        - min_pos_threshold: int — classes with fewer positive samples than this will be oversampled
        - target_pos_count: int — number of positive samples to reach via oversampling
        - random_state: int — for reproducibility

        Returns:
        - X_oversampled, y_oversampled: pd.DataFrames
        """
    np.random.seed(random_state)    # Коли ти викликаєш np.random.choice() (чи будь-яку іншу випадкову функцію),
                                    # результат залежить від випадку. Щоб мати однакові результати при кожному запуску
                                    # коду (тобто зробити результат відтворюваним), задається фіксоване "зерно" (seed).
    X_oversampled = X.copy()
    y_oversampled = y.copy()

    for col in y.columns:
        pos_value_count = y[col].sum()
        neg_value_count = len(y) - pos_value_count
        if (pos_value_count < neg_value_count) & (pos_value_count/len(y) < min_pos_threshold):
            oversample_needed = target_pos_count - pos_value_count
            if oversample_needed <= 0:
                continue

            # знайти індекси рядків, де є позитивний клас
            pos_indexes = y[y[col] == 1].index.tolist()
            if not pos_indexes:
                continue

            # випадково вибрати індекси для дублювання
            choosen = np.random.choice(pos_indexes, size=oversample_needed, replace=True)

            # дублювати X і y
            X_oversampled = pd.concat([X_oversampled, X.loc[choosen]], ignore_index=True)
            y_oversampled = pd.concat([y_oversampled, y.loc[choosen]], ignore_index=True)

    return X_oversampled, y_oversampled

