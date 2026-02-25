from pathlib import Path

import math
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


FEATURE_COLUMNS = [
    "district_class",
    "land_price_per_m2",
    "soil_complexity",
    "site_accessibility",
    "land_area_m2",
    "house_area_m2",
    "design_complexity",
    "materials_class",
    "planned_duration_days",
    "planned_budget",
    "crew_experience_years",
    "crew_efficiency_score",
    "crew_current_load",
    "supplier_reliability_score",
    "delivery_distance_km",
    "weather_season",
    "material_price_index",
    "mortgage_rate",
    "market_demand_index",
    "client_type",
    "labor_cost_index",
]


def _encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = ["district_class", "materials_class", "weather_season", "client_type"]
    encoded = df.copy()
    for c in cat_cols:
        le = LabelEncoder()
        encoded[c] = le.fit_transform(encoded[c])
    return encoded


def train_regressor(
    data_path: str = "data/raw/synthetic_construction_projects.csv",
    model_path: str = "models/margin_model.pkl",
) -> dict:
    """
    Обучение модели маржи и сохранение на диск.
    Возвращает MAE и RMSE на тесте.
    """
    df = pd.read_csv(data_path)
    df_enc = _encode_categories(df)

    X = df_enc[FEATURE_COLUMNS]
    y = df_enc["actual_margin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, model_path)

    metrics = {"mae": mae, "rmse": rmse}
    print(f"Margin model saved to {model_path}")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return metrics


