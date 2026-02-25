import numpy as np
import pandas as pd
from pathlib import Path


def generate_data(
    output_path: str = "data/raw/synthetic_construction_projects.csv",
    n_projects: int = 6000,
    seed: int = 42,
    material_price_range=(0.9, 1.1),
    mortgage_rate_range=(7.0, 12.0),
    market_demand_range=(0.8, 1.2),
    labor_cost_range=(0.9, 1.2),
) -> None:
    """
    Генерация синтетических проектов строительства с возможностью
    менять параметры рынка.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    N = n_projects

    # 1. District / Land
    district_class = np.random.choice(
        ["econom", "standard", "premium"], size=N, p=[0.5, 0.4, 0.1]
    )
    land_price_per_m2 = np.array(
        [
            np.random.randint(3000, 5001)
            if d == "econom"
            else np.random.randint(6000, 12001)
            if d == "standard"
            else np.random.randint(15000, 25001)
            for d in district_class
        ]
    )
    soil_complexity = np.random.randint(1, 6, N)
    site_accessibility = np.random.randint(1, 6, N)
    land_area_m2 = np.random.randint(400, 1001, N)

    # 2. House / Project
    house_area_m2 = np.random.normal(150, 50, N).clip(80, 350)
    design_complexity = np.random.randint(1, 6, N)
    materials_class = np.random.choice(
        ["econom", "standard", "premium"], size=N, p=[0.4, 0.4, 0.2]
    )
    planned_duration_days = (house_area_m2 / 150 * 90 + design_complexity * 10).astype(
        int
    )
    base_cost_per_m2 = 50000

    materials_multiplier = np.array(
        [
            0.8 if m == "econom" else 1.0 if m == "standard" else 1.3
            for m in materials_class
        ]
    )
    planned_budget = (
        house_area_m2 * base_cost_per_m2 * materials_multiplier
        + design_complexity * 50000
        + land_price_per_m2 * land_area_m2
    )

    # 3. Construction (включая скрытое качество управления)
    crew_experience_years = np.random.randint(1, 11, N)
    crew_efficiency_score = np.clip(
        np.random.normal(0.8 + crew_experience_years / 50, 0.05, N), 0.7, 1.0
    )
    crew_current_load = np.random.randint(0, 4, N)
    supplier_reliability_score = np.clip(
        np.random.normal(0.9, 0.05, N), 0.7, 1.0
    )
    # Скрытый фактор управления проектом (не попадает в фичи)
    management_quality = np.random.randint(1, 6, N)  # 1 = плохо, 5 = отлично
    delivery_distance_km = np.random.randint(5, 51, N)
    weather_season = np.random.choice(["winter", "spring", "summer", "autumn"], N)

    weather_factor = np.array(
        [
            5 if w == "winter" else 2 if w == "spring" else 1 if w == "summer" else 3
            for w in weather_season
        ]
    )

    # 4. External / Market (управляемые параметры рынка + неожиданные события)
    material_price_index = np.random.uniform(
        material_price_range[0], material_price_range[1], N
    )
    mortgage_rate = np.random.uniform(
        mortgage_rate_range[0], mortgage_rate_range[1], N
    )
    market_demand_index = np.random.uniform(
        market_demand_range[0], market_demand_range[1], N
    )
    labor_cost_index = np.random.uniform(
        labor_cost_range[0], labor_cost_range[1], N
    )
    # Скрытый индекс неожиданных событий (force majeurs)
    unexpected_events_index = np.random.exponential(scale=0.4, size=N)
    unexpected_events_index = np.clip(unexpected_events_index, 0, 2.0)

    client_type = np.random.choice(["private", "commercial"], size=N, p=[0.7, 0.3])

    # 5. Calculate delays (с учётом управления и неожиданных событий)
    delay_days = (
        planned_duration_days * 0.05 * (5 - crew_efficiency_score * 5)
        + (5 - supplier_reliability_score * 5) * 10
        + soil_complexity * 2
        + delivery_distance_km * 0.1
        + weather_factor
        + (6 - management_quality) * 3.0
        + unexpected_events_index * 6.0
        + np.random.normal(0, 5, N)
    ).clip(0, None)

    # 6. Actual cost (доп. влияние неожиданных событий и стоимости труда)
    actual_cost = planned_budget * (1 + 0.02 * delay_days / planned_duration_days)
    actual_cost *= material_price_index
    actual_cost *= 1 + 0.05 * (5 - crew_efficiency_score * 5)
    actual_cost *= 1 + 0.03 * unexpected_events_index
    # рост стоимости труда сильнее бьёт по смете
    labor_centered = (labor_cost_index - 1.0) / 0.1
    actual_cost *= 1 + 0.04 * labor_centered

    penalty_cost = np.zeros(N)
    penalty_mask = (client_type == "commercial") & (
        delay_days > planned_duration_days * 1.1
    )
    penalty_cost[penalty_mask] = (
        0.01 * planned_budget[penalty_mask] * delay_days[penalty_mask]
    )
    actual_cost += penalty_cost

    # 7. Targets (вероятностная модель перерасхода)
    actual_margin = (planned_budget - actual_cost) / planned_budget
    # базовый логит перерасхода:
    # чем выше перерасход, хуже управление, сложнее грунт, хуже погода и длиннее логистика — тем выше риск
    overrun_ratio = actual_cost / (planned_budget + 1e-9) - 1.0
    soil_centered = soil_complexity - 3  # -2..+2
    delivery_centered = delivery_distance_km - 30  # около 0
    weather_centered = weather_factor - 2  # -1..+3
    material_centered = (material_price_index - 1.0) / 0.1  # примерно -1..+1

    logit = (
        -1.0
        + 6.0 * overrun_ratio
        + 0.4 * (5 - management_quality)
        + 0.25 * soil_centered
        + 0.01 * delivery_centered
        + 0.3 * weather_centered
        + 0.4 * material_centered
        + np.random.normal(0, 0.8, N)
    )
    prob_overrun = 1 / (1 + np.exp(-logit))
    budget_overrun = (np.random.rand(N) < prob_overrun).astype(int)
    final_profit = planned_budget - actual_cost

    # 8. Create DataFrame
    df = pd.DataFrame(
        {
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
            "client_type": client_type,
            "labor_cost_index": labor_cost_index,
            "delay_days": delay_days,
            "actual_cost": actual_cost,
            "actual_margin": actual_margin,
            "budget_overrun": budget_overrun,
            "final_profit": final_profit,
        }
    )

    df.to_csv(output_path, index=False)
    print(f"{N} проектов сгенерированы и сохранены в '{output_path}'")


