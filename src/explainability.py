import joblib
import pandas as pd
import shap
import plotly.express as px

from .train_regressor import FEATURE_COLUMNS, _encode_categories


def explain_margin(
    data_path: str = "data/raw/synthetic_construction_projects.csv",
    model_path: str = "models/margin_model.pkl",
    sample_size: int = 500,
) -> None:
    """
    SHAP summary + feature importance + локальные объяснения для модели маржи.
    """
    df = pd.read_csv(data_path)
    df_enc = _encode_categories(df)

    model = joblib.load(model_path)

    X = df_enc[FEATURE_COLUMNS]
    if sample_size and len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # SHAP summary (global)
    shap.summary_plot(shap_values, X, plot_type="bar")

    # Feature importance (по среднему абсолютному SHAP)
    shap.summary_plot(shap_values, X)

    # Local explanations для первых нескольких проектов
    for i in range(min(3, len(X))):
        shap.plots.waterfall(shap_values[i], max_display=10)


def explain_risk(
    data_path: str = "data/raw/synthetic_construction_projects.csv",
    model_path: str = "models/risk_model.pkl",
    sample_size: int = 500,
) -> None:
    """
    SHAP summary + feature importance + локальные объяснения для модели риска.
    """
    df = pd.read_csv(data_path)
    df_enc = _encode_categories(df)

    model = joblib.load(model_path)

    X = df_enc[FEATURE_COLUMNS]
    if sample_size and len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # SHAP summary + глобальное влияние
    shap.summary_plot(shap_values, X, plot_type="bar")
    shap.summary_plot(shap_values, X)

    # Локальные объяснения
    for i in range(min(3, len(X))):
        shap.plots.waterfall(shap_values[i], max_display=10)

    # Feature importance из встроенных важностей модели (Plotly bar)
    importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
    ).sort_values(by="importance", ascending=False)

    fig = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance — Risk of Budget Overrun",
    )
    fig.show()

