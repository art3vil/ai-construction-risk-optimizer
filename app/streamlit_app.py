import os
import sys

import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator import simulate_scenario


def _format_margin(margin: float) -> str:
    """Маржа в процентах, читаемо."""
    pct = margin * 100
    return f"{pct:+.1f}%"


def _format_risk(prob: float) -> str:
    """Вероятность перерасхода в процентах."""
    return f"{prob * 100:.1f}%"


def _get_recommendations(result: dict) -> tuple[str, list[str]]:
    """
    По результату симуляции возвращает: (краткий вердикт, список рекомендаций).
    """
    margin = result["scenario_margin_pred"]
    risk = result["scenario_risk_prob"]
    delta_m = result["delta_margin"]
    delta_r = result["delta_risk"]

    recommendations = []

    # Вердикт по марже
    if margin < 0:
        verdict_m = "Маржа отрицательная — проект убыточен по прогнозу."
        recommendations.append("Пересмотрите смету: снизьте затраты на материалы или логистику, либо увеличьте плановый бюджет.")
        recommendations.append("Рассмотрите более опытную бригаду (выше эффективность при том же сроке).")
    elif margin < 0.05:
        verdict_m = "Низкая маржа — небольшой запас на риски."
        recommendations.append("Закладывайте буфер в бюджет или снижайте издержки (поставщики, доставка, класс материалов).")
    elif margin < 0.15:
        verdict_m = "Маржа в приемлемом диапазоне."
    else:
        verdict_m = "Хорошая прогнозная маржа."

    # Вердикт по риску перерасхода
    if risk > 0.6:
        verdict_r = "Высокий риск перерасхода бюджета."
        recommendations.append("Усилььте контроль: выберите более надёжных поставщиков (supplier_reliability_score ближе к 1.0).")
        recommendations.append("Сократите логистические риски: уменьшите дистанцию доставки или заложите запас по срокам.")
        recommendations.append("По возможности перенесите старт на сезон с лучшей погодой (лето).")
    elif risk > 0.4:
        verdict_r = "Умеренный риск перерасхода."
        recommendations.append("Мониторьте ключевые драйверы: цены на материалы, сроки поставок, погоду.")
    else:
        verdict_r = "Риск перерасхода низкий."

    # Изменение относительно базового сценария
    if delta_m > 0.01:
        recommendations.append("Текущие параметры улучшили маржу относительно базового проекта — такой сценарий выгоднее.")
    elif delta_m < -0.01:
        recommendations.append("Текущие параметры снизили маржу относительно базового проекта — рассмотрите откат части изменений.")

    if delta_r > 0.1:
        recommendations.append("Риск перерасхода вырос по сравнению с базой — стоит усилить контроль или смягчить условия (поставщики, сроки, сезон).")
    elif delta_r < -0.1:
        recommendations.append("Риск перерасхода снизился — выбранный сценарий безопаснее базового.")

    verdict = f"{verdict_m} {verdict_r}"
    return verdict, recommendations


def main() -> None:
    st.title("AI Construction Risk Optimizer")

    st.header("Выбор базового проекта")
    base_index = st.number_input(
        "Индекс проекта (0-based из датасета)", min_value=0, value=0, step=1
    )

    st.header("Параметры проекта (можно менять все)")

    col1, col2 = st.columns(2)

    with col1:
        district_class = st.selectbox(
            "Класс района (district_class)", ["econom", "standard", "premium"]
        )
        land_price_per_m2 = st.number_input(
            "Стоимость земли, ₽/м² (land_price_per_m2)", min_value=1000, max_value=50000, value=10000, step=500
        )
        soil_complexity = st.slider(
            "Сложность грунта (soil_complexity)", min_value=1, max_value=5, value=3
        )
        site_accessibility = st.slider(
            "Доступность площадки (site_accessibility)", min_value=1, max_value=5, value=3
        )
        land_area_m2 = st.number_input(
            "Площадь участка, м² (land_area_m2)", min_value=100, max_value=5000, value=800, step=50
        )
        house_area_m2 = st.number_input(
            "Площадь дома, м² (house_area_m2)", min_value=50, max_value=1000, value=150, step=10
        )
        design_complexity = st.slider(
            "Сложность проекта (design_complexity)", min_value=1, max_value=5, value=3
        )
        materials_class = st.selectbox(
            "Класс материалов (materials_class)", ["econom", "standard", "premium"]
        )
        planned_duration_days = st.number_input(
            "Плановая длительность, дни (planned_duration_days)", min_value=30, max_value=730, value=180, step=10
        )
        planned_budget = st.number_input(
            "Плановый бюджет, ₽ (planned_budget)", min_value=1_000_000, max_value=500_000_000, value=20_000_000, step=500_000
        )

    with col2:
        crew_experience_years = st.slider(
            "Опыт бригады, лет (crew_experience_years)", min_value=1, max_value=10, value=5
        )
        crew_efficiency_score = st.slider(
            "Эффективность бригады (crew_efficiency_score)", min_value=0.7, max_value=1.0, value=0.85, step=0.01
        )
        crew_current_load = st.slider(
            "Текущая загрузка бригады (crew_current_load)", min_value=0, max_value=3, value=1
        )
        supplier_reliability_score = st.slider(
            "Надёжность поставщиков (supplier_reliability_score)", min_value=0.7, max_value=1.0, value=0.9, step=0.01
        )
        delivery_distance_km = st.number_input(
            "Дистанция доставки, км (delivery_distance_km)", min_value=1, max_value=200, value=30, step=5
        )
        weather_season = st.selectbox(
            "Сезон строительства (weather_season)", ["winter", "spring", "summer", "autumn"]
        )
        material_price_index = st.number_input(
            "Индекс стоимости материалов (material_price_index)", min_value=0.5, max_value=2.0, value=1.0, step=0.05
        )
        mortgage_rate = st.number_input(
            "Ставка ипотеки, % (mortgage_rate)", min_value=1.0, max_value=25.0, value=10.0, step=0.5
        )
        market_demand_index = st.number_input(
            "Индекс спроса рынка (market_demand_index)", min_value=0.5, max_value=2.0, value=1.0, step=0.05
        )
        labor_cost_index = st.number_input(
            "Индекс стоимости труда (labor_cost_index)", min_value=0.5, max_value=2.0, value=1.0, step=0.05
        )
        client_type = st.selectbox(
            "Тип клиента (client_type)", ["private", "commercial"]
        )

    overrides = {
        "district_class": district_class,
        "land_price_per_m2": land_price_per_m2,
        "soil_complexity": soil_complexity,
        "site_accessibility": site_accessibility,
        "land_area_m2": land_area_m2,
        "house_area_m2": house_area_m2,
        "design_complexity": design_complexity,
        "materials_class": materials_class,
        "planned_duration_days": planned_duration_days,
        "planned_budget": planned_budget,
        "crew_experience_years": crew_experience_years,
        "crew_efficiency_score": crew_efficiency_score,
        "crew_current_load": crew_current_load,
        "supplier_reliability_score": supplier_reliability_score,
        "delivery_distance_km": delivery_distance_km,
        "weather_season": weather_season,
        "material_price_index": material_price_index,
        "mortgage_rate": mortgage_rate,
        "market_demand_index": market_demand_index,
        "labor_cost_index": labor_cost_index,
        "client_type": client_type,
    }

    st.markdown("---")
    if st.button("Рассчитать риск и маржу"):
        result = simulate_scenario(base_index=base_index, overrides=overrides)

        st.subheader("Результат симуляции")

        # Карточки с ключевыми показателями
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Прогноз маржи (сценарий)",
                _format_margin(result["scenario_margin_pred"]),
                _format_margin(result["delta_margin"]),
            )
        with col2:
            st.metric(
                "Вероятность перерасхода бюджета",
                _format_risk(result["scenario_risk_prob"]),
                f"{result['delta_risk'] * 100:+.1f} п.п.",
            )
        with col3:
            base_m = _format_margin(result["original_margin_pred"])
            base_r = _format_risk(result["original_risk_prob"])
            st.caption("Базовый проект")
            st.markdown(f"Маржа: **{base_m}** · Риск: **{base_r}**")

        # Вердикт и рекомендации
        verdict, recommendations = _get_recommendations(result)
        st.info(verdict)

        st.subheader("Рекомендации по улучшению")
        for rec in recommendations:
            st.markdown(f"- {rec}")

        with st.expander("Числовые значения (для отчётов)"):
            st.json(result)


if __name__ == "__main__":
    main()

