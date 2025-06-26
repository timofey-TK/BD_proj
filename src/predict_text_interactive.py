import os
import re
import sys
from sentence_transformers import SentenceTransformer
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.linalg import DenseVector
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging

# Отключаем лишние логи и предупреждения
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--conf spark.ui.showConsoleProgress=false "
    "--conf spark.driver.extraJavaOptions='-Dlog4j.logLevel=OFF' "
    "--conf spark.executor.extraJavaOptions='-Dlog4j.logLevel=OFF' "
    "pyspark-shell"
)
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_CONF_DIR"] = os.path.abspath(".")
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

console = Console()

def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def tokenize_and_remove_stopwords(text):
    if text is None:
        return ""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in tokens if word not in stop_words])

def sentiment_to_str(pred):
    mapping = {
        0.0: ("Negative", "😡", "bold red"),
        1.0: ("Neutral", "😐", "bold yellow"),
        2.0: ("Positive", "😃", "bold green")
    }
    return mapping.get(pred, ("Unknown", "❓", "bold white"))

def main():
    console.print(Panel("[bold cyan]Sentiment Analysis[/bold cyan]\nВведите текст для анализа (на английском, [bold]exit[/bold] — выход):", title="[bold blue]Text Classifier"))
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as progress:
        progress.add_task(description="[cyan]Загрузка эмбеддера...", total=None)
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as progress:
        progress.add_task(description="[cyan]Запуск Spark и загрузка модели...", total=None)
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("TextPredictInteractive").getOrCreate()
        spark.sparkContext.setLogLevel("OFF")
        lr_model = LogisticRegressionModel.load("embedding_logreg_model")
    console.print("[green]Готово! Вводите текст. Для выхода напишите [bold]exit[/bold] или оставьте строку пустой.[/green]\n")
    while True:
        user_text = Prompt.ask("[bold magenta]>>>[/bold magenta]")
        if not user_text.strip() or user_text.strip().lower() == "exit":
            console.print("[bold cyan]До свидания![/bold cyan]")
            break
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as progress:
            progress.add_task(description="[cyan]Препроцессинг...", total=None)
            cleaned = clean_text(user_text)
            processed = tokenize_and_remove_stopwords(cleaned)
        console.print(f"[bold]Обработанный текст:[/bold] [italic]{processed}[/italic]", style="dim")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as progress:
            progress.add_task(description="[cyan]Генерация эмбеддинга и предсказание...", total=None)
            embedding = model.encode([processed])[0]
            features = DenseVector(embedding)
            df = spark.createDataFrame([(processed, features)], ["processed_comment", "features"])
            prediction = lr_model.transform(df).collect()[0].prediction
        sentiment, emoji, color = sentiment_to_str(prediction)
        console.print(Panel(f"[bold]{emoji}  {sentiment}[/bold]", title="[bold white]Результат анализа[/bold white]", style=color))
    spark.stop()

if __name__ == "__main__":
    main() 