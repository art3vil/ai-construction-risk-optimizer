from pprint import pprint

from src.data_generator import generate_data
from src.train_regressor import train_regressor
from src.train_classifier import train_classifier
from src.simulator import simulate_scenario


def main() -> None:
    print("=== 1. Генерация данных ===")
    generate_data()

    print("\n=== 2. Обучение модели маржи ===")
    reg_metrics = train_regressor()
    pprint(reg_metrics)

    print("\n=== 3. Обучение модели риска ===")
    clf_metrics = train_classifier()
    pprint(clf_metrics)

    print("\n=== 4. What-if симуляция для проекта 0 ===")
    result = simulate_scenario(
        base_index=0,
        overrides={"materials_class": "premium", "delivery_distance_km": 10},
    )
    pprint(result)


if __name__ == "__main__":
    main()

