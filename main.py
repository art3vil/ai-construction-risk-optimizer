import subprocess
import os
import sys
from pprint import pprint

from src.data_generator import generate_data
from src.train_regressor import train_regressor
from src.train_classifier import train_classifier
from src.simulator import simulate_scenario


def run_web_app():
    """Функция для автоматического запуска Streamlit"""
    # Путь к вашему файлу (согласно вашей структуре)
    app_path = os.path.join("app", "streamlit_app.py")

    print(f"\n🚀 Запуск веб-интерфейса из: {app_path}...")

    if not os.path.exists(app_path):
        print(f"❌ Ошибка: Файл {app_path} не найден! Проверьте папку app.")
        return

    try:
        # Запускаем streamlit.
        # Используем sys.executable для запуска через текущее виртуальное окружение
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    except KeyboardInterrupt:
        print("\n\n✅ Веб-интерфейс остановлен пользователем.")
    except Exception as e:
        print(f"❌ Не удалось запустить веб-интерфейс: {e}")


def main() -> None:
    print("==============================================")
    print("🧱 ПОДГОТОВКА СИСТЕМЫ ОПТИМИЗАЦИИ РИСКОВ")
    print("==============================================")

    # 1. Генерация данных
    print("\n[1/3] Генерация синтетических данных...")
    generate_data()

    # 2. Обучение моделей (чтобы проверить работоспособность и создать файлы моделей)
    print("\n[2/3] Обучение моделей и проверка метрик...")
    reg_metrics = train_regressor()
    clf_metrics = train_classifier()

    print("--- Метрики регрессии (Прибыль) ---")
    pprint(reg_metrics)
    print("--- Метрики классификации (Риски) ---")
    pprint(clf_metrics)

    # 3. Запуск веб-приложения
    print("\n[3/3] Все готово! Переходим к визуализации...")
    run_web_app()


if __name__ == "__main__":
    main()