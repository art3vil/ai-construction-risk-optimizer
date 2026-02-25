from typing import Any, Dict, Optional

import joblib
import pandas as pd

from .train_regressor import FEATURE_COLUMNS, _encode_categories


def simulate_scenario(
    base_index: int,
    overrides: Optional[Dict[str, Any]] = None,
    data_path: str = "data/raw/synthetic_construction_projects.csv",
    margin_model_path: str = "models/margin_model.pkl",
    risk_model_path: str = "models/risk_model.pkl",
) -> Dict[str, Any]:
    """
    What-if симуляция:
    - берём проект с индексом base_index,
    - применяем overrides к его признакам,
    - пересчитываем прогноз маржи и риска перерасхода.
    """
    overrides = overrides or {}

    df = pd.read_csv(data_path)
    if base_index < 0 or base_index >= len(df):
        raise IndexError("base_index вне диапазона данных")

    original_row = df.iloc[base_index].copy()
    scenario_row = original_row.copy()
    for k, v in overrides.items():
        if k not in df.columns:
            raise KeyError(f"Неизвестный признак в overrides: {k}")
        scenario_row[k] = v

    # Подготовка данных
    df_enc = _encode_categories(df)
    original_enc = df_enc.iloc[base_index][FEATURE_COLUMNS]

    df_scenario = df.copy()
    df_scenario.loc[base_index] = scenario_row
    scenario_enc = _encode_categories(df_scenario).iloc[base_index][FEATURE_COLUMNS]

    margin_model = joblib.load(margin_model_path)
    risk_model = joblib.load(risk_model_path)

    original_margin_pred = float(margin_model.predict([original_enc])[0])
    scenario_margin_pred = float(margin_model.predict([scenario_enc])[0])

    original_risk_prob = float(risk_model.predict_proba([original_enc])[0, 1])
    scenario_risk_prob = float(risk_model.predict_proba([scenario_enc])[0, 1])

    result = {
        "original_margin_pred": original_margin_pred,
        "scenario_margin_pred": scenario_margin_pred,
        "original_risk_prob": original_risk_prob,
        "scenario_risk_prob": scenario_risk_prob,
        "delta_margin": scenario_margin_pred - original_margin_pred,
        "delta_risk": scenario_risk_prob - original_risk_prob,
    }

    return result


