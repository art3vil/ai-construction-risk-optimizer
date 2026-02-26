import subprocess
import sys
import os
from pprint import pprint

from src.data_generator import generate_data
from src.train_regressor import train_regressor
from src.train_classifier import train_classifier
from src.simulator import simulate_scenario


def run_web_app():
    """Функция для запуска Streamlit приложения"""
    print("\n=== Запуск веб-интерфейса... ===")
    # Путь к вашему файлу
    app_path = "app/streamlit_app.py"

    if not os.path.exists(app_path):
        print(f"Ошибка: Файл {app_path} не найден в корневой директории.")
        return

    try:
        # Запускаем streamlit как подпроцесс
        subprocess.run(["streamlit", "run", app_path])
    except KeyboardInterrupt:
        print("\nВеб-интерфейс остановлен.")
    except Exception as e:
        print(f"Не удалось запустить веб-интерфейс: {e}")


def main() -> None:
    # 1. Всегда генерируем актуальные данные перед запуском
    print("=== 1. Подготовка данных ===")
    generate_data()

    # Спрашиваем пользователя, что он хочет сделать
    print("\nВыберите режим работы:")
    print("[1] Консольный анализ (обучение и тест-симуляция)")
    print("[2] Запуск Веб-панели (Streamlit)")
    print("[3] Все вместе")

    choice = input("\nВведите номер (1/2/3): ")

    if choice in ['1', '3']:
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

    if choice in ['2', '3']:
        run_web_app()


if __name__ == "__main__":
    main()