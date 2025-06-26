import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import LogisticRegression
from sentence_transformers import SentenceTransformer
from src.logging_config import get_logger


logger = get_logger(__name__)

def main():
    logger.info("Загрузка данных для генерации эмбеддингов.")
    with SparkSession.builder.appName("PandasLoader").getOrCreate() as spark:
        spark.sparkContext.setLogLevel("ERROR")
        df_spark = spark.read.parquet("processed_comments.parquet")
        pandas_df = df_spark.select("processed_comment", "label").toPandas()

    logger.info("Загрузка предобученной языковой модели 'all-MiniLM-L6-v2'.")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    logger.info("Генерация эмбеддингов...")
    embeddings = model.encode(pandas_df['processed_comment'].tolist(), show_progress_bar=True)
    pandas_df['features'] = list(embeddings)

    logger.info("Инициализация Spark-сессии для обучения.")
    with SparkSession.builder.appName("SentimentTraining").getOrCreate() as spark:
        spark.sparkContext.setLogLevel("ERROR")
        schema = StructType([
            StructField("processed_comment", StringType(), True),
            StructField("label", IntegerType(), True),
            StructField("features", VectorUDT(), True)
        ])
        
        def to_vector(v):
            return Vectors.dense(v)

        rows = [(row['processed_comment'], int(row['label']), to_vector(row['features'])) for _, row in pandas_df.iterrows()]
        spark_df = spark.createDataFrame(rows, schema=schema)

        (trainingData, testData) = spark_df.randomSplit([0.8, 0.2], seed=42)

        logger.info("Обучение модели LogisticRegression на эмбеддингах.")
        lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=20, regParam=0.01)
        lr_model = lr.fit(trainingData)

        logger.info("Обучение модели RandomForestClassifier на эмбеддингах.")
        from pyspark.ml.classification import RandomForestClassifier
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=5)
        rf_model = rf.fit(trainingData)

        logger.info("Сохранение обученных моделей и тестовых данных.")
        lr_model.write().overwrite().save("embedding_logreg_model")
        rf_model.write().overwrite().save("embedding_rf_model")
        testData.write.mode("overwrite").parquet("test_data_with_embeddings.parquet")
        logger.info("Артефакты для предсказания успешно сохранены.")

if __name__ == "__main__":
    main()


