from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from src.logging_config import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Инициализация Spark-сессии для предсказания.")
    with SparkSession.builder.appName("SentimentPrediction").getOrCreate() as spark:
        spark.sparkContext.setLogLevel("ERROR")

        logger.info("Загрузка обученной модели и тестовых данных.")
        lr_model = LogisticRegressionModel.load("embedding_logreg_model")
        test_data = spark.read.parquet("test_data_with_embeddings.parquet")

        logger.info("Выполнение предсказаний на тестовых данных.")
        predictions = lr_model.transform(test_data)

        evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)

        logger.info("-" * 50)
        logger.info(f"Итоговая точность модели на тестовых данных: {accuracy:.4f}")
        logger.info("-" * 50)

        # Сохраняем все необходимые данные для визуализации
        predictions.select("processed_comment", "label", "prediction", "features").write.mode("overwrite").parquet("predictions.parquet")
        logger.info("Предсказания, эмбеддинги и текст сохранены для визуализации.")

if __name__ == "__main__":
    main() 