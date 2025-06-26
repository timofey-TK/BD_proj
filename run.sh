#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Очистка предыдущих артефактов ---
echo "--- Очистка старых артефактов ---"
rm -rf embedding_logreg_model test_data_with_embeddings.parquet predictions.parquet
rm -rf artifacts/*
echo "Старые артефакты удалены."
echo ""

# --- Запуск пайплайна ---
echo "--- Шаг 1: Предобработка данных ---"
python -m src.preprocess_data

echo ""
echo "--- Шаг 2: Обучение модели ---"
python -m src.train_model

echo ""
echo "--- Шаг 3: Предсказание на тестовых данных ---"
python -m src.predict

echo ""
echo "--- Шаг 4: Визуализация результатов ---"
python -m src.visualize_sentiment

echo ""
echo "--- Пайплайн успешно выполнен! ---"
echo "Все артефакты находятся в директории 'artifacts'."
echo "Откройте 'artifacts/embeddings_visualization.html' в браузере для просмотра интерактивного графика." 