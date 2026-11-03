"""
Модели-кандидаты для GridSearchCV.
Каждая модель стоит в Pipeline: scaler - model.
Для деревьев scaler не нужен ("passthrough" = шаг пропускается).
"""
from lightgbm import LGBMClassifier, early_stopping
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def split_for_early_stopping(X, y):
    # Откладываем 10% данных, на них бустинг проверяет, пора ли остановиться
    return train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)


class LGBMEarlyStop(LGBMClassifier):
    """LightGBM с early stopping: останавливается, если 30 деревьев подряд нет улучшения."""

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = split_for_early_stopping(X, y)
        return super().fit(X_train, y_train, eval_set=[(X_val, y_val)],
                           callbacks=[early_stopping(30, verbose=False)])


class XGBEarlyStop(XGBClassifier):
    """XGBoost с early stopping (число раундов задается в early_stopping_rounds)."""

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = split_for_early_stopping(X, y)
        return super().fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


def make_pipeline():
    # Базовый конвейер. Какие шаги в нем будут, решает сетка ниже.
    return Pipeline([("scaler", "passthrough"), ("model", LogisticRegression())])


def make_grid():
    # Каждый словарь - один кандидат и его гиперпараметры.
    # SVM не включен: на 60 000 картинках с вероятностями он обучается часами.
    return [
        {"scaler": [StandardScaler()],
         "model": [LogisticRegression(max_iter=300)]},

        {"scaler": [StandardScaler()],
         "model": [MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, random_state=42)]},

        {"model": [RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)]},

        {"model": [ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=42)]},

        {"model": [LGBMEarlyStop(n_estimators=1000, max_bin=63, colsample_bytree=0.5,
                                 random_state=42, verbose=-1)],
         "model__num_leaves": [31, 63]},

        {"model": [XGBEarlyStop(n_estimators=1000, early_stopping_rounds=30, max_bin=63,
                                colsample_bytree=0.5, tree_method="hist", random_state=42)],
         "model__max_depth": [6, 8]},
    ]
