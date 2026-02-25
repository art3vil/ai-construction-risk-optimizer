from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from .train_regressor import FEATURE_COLUMNS, _encode_categories


def train_classifier(
    data_path: str = "data/raw/synthetic_construction_projects.csv",
    model_path: str = "models/risk_model.pkl",
) -> dict:
    """
    Обучение модели риска перерасхода бюджета.
    Возвращает ROC-AUC и confusion matrix.
    """
    df = pd.read_csv(data_path)
    df_enc = _encode_categories(df)

    X = df_enc[FEATURE_COLUMNS]
    y = df_enc["budget_overrun"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)

    metrics = {"roc_auc": roc_auc, "confusion_matrix": cm, "classification_report": report}
    print(f"Risk model saved to {model_path}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Confusion matrix:\n{cm}")
    return metrics

