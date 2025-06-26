from pyspark.sql import SparkSession
from dotenv import load_dotenv
import os
load_dotenv()

# Инициализация SparkSession
spark = SparkSession.builder \
    .appName("YouTubeCommentsSentimentAnalysis") \
    .getOrCreate()

# Загрузка данных
df = spark.read.csv(os.getenv("DATA_PATH"), header=True, inferSchema=True)

# Показываем схему и первые 5 строк данных
df.printSchema()
df.show(5)

# Останавливаем SparkSession
spark.stop()


