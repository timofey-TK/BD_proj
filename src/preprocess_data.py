from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from src.logging_config import get_logger
from dotenv import load_dotenv
import os
load_dotenv()

logger = get_logger(__name__)

def main():
    logger.info("Инициализация Spark-сессии для предобработки данных.")
    with SparkSession.builder.appName("DataPreprocessing").getOrCreate() as spark:
        spark.sparkContext.setLogLevel("ERROR")

        logger.info("Загрузка исходного набора данных YoutubeCommentsDataSet.csv.")
        df = spark.read.csv(os.getenv("DATA_PATH"), header=True, inferSchema=True)

        def clean_text(text):
            if text is None:
                return ""
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            return text.lower()

        clean_text_udf = udf(clean_text, StringType())
        df = df.withColumn("cleaned_comment", clean_text_udf(df["Comment"]))

        stop_words = set(stopwords.words("english"))
        def tokenize_and_remove_stopwords(text):
            if text is None:
                return ""
            tokens = word_tokenize(text)
            return " ".join([word for word in tokens if word not in stop_words])

        tokenize_udf = udf(tokenize_and_remove_stopwords, StringType())
        df = df.withColumn("processed_comment", tokenize_udf(df["cleaned_comment"]))

        def sentiment_to_int(sentiment):
            if sentiment == "positive": return 2
            elif sentiment == "neutral": return 1
            else: return 0

        sentiment_to_int_udf = udf(sentiment_to_int, IntegerType())
        df = df.withColumn("label", sentiment_to_int_udf(df["Sentiment"]))

        logger.info("Сохранение предобработанных данных в 'processed_comments.parquet'.")
        df.write.mode("overwrite").parquet("processed_comments.parquet")
        logger.info("Предобработка успешно завершена.")

if __name__ == "__main__":
    main()


