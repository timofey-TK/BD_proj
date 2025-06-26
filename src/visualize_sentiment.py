import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import webbrowser
import threading
import http.server
import socketserver
from src.logging_config import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Процесс визуализации запущен.")
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
        logger.info("Создана директория 'artifacts'.")

    with SparkSession.builder.appName("Visualization").getOrCreate() as spark:
        spark.sparkContext.setLogLevel("ERROR")
        predictions_df = spark.read.parquet("predictions.parquet")
        pandas_df = predictions_df.toPandas()

    # --- График 1: Распределение тональностей ---
    create_sentiment_distribution_plot(pandas_df)

    # --- График 2: Матрица ошибок ---
    create_confusion_matrix_plot(pandas_df)
    
    # --- График 3: Интерактивная 3D-визуализация эмбеддингов ---
    create_interactive_embedding_plot(pandas_df)

    logger.info("Визуализация успешно завершена. Все артефакты в директории 'artifacts'.")
    
    # --- Запуск веб-сервера для просмотра результатов ---
    start_web_server()

def create_sentiment_distribution_plot(df):
    logger.info("Создание графика распределения тональностей...")
    sentiment_map = {0.0: "Negative", 1.0: "Neutral", 2.0: "Positive"}
    df['sentiment'] = df['prediction'].map(sentiment_map)
    
    plt.figure(figsize=(8, 6))
    sns.countplot(x='sentiment', data=df, order=['Positive', 'Neutral', 'Negative'], palette="viridis", hue='sentiment', legend=False)
    plt.title("Распределение предсказанной тональности")
    plt.xlabel("Тональность")
    plt.ylabel("Количество комментариев")
    
    path = "artifacts/sentiment_distribution.png"
    plt.savefig(path)
    logger.info(f"График распределения сохранен в '{path}'")
    plt.close()

def create_confusion_matrix_plot(df):
    logger.info("Создание матрицы ошибок...")
    cm = confusion_matrix(df['label'], df['prediction'])
    class_names = ['Negative', 'Neutral', 'Positive']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Матрица ошибок')
    plt.ylabel('Истинная метка')
    plt.xlabel('Предсказанная метка')
    
    path = "artifacts/confusion_matrix.png"
    plt.savefig(path)
    logger.info(f"Матрица ошибок сохранена в '{path}'")
    plt.close()

def create_interactive_embedding_plot(df):
    logger.info("Создание интерактивной 3D-визуализации эмбеддингов.")
    
    # Ограничим выборку для ускорения вычислений
    sample_n = min(len(df), 3000)
    logger.info(f"Для 3D-визуализации будет использована выборка из {sample_n} точек.")
    sample_df = df.sample(n=sample_n, random_state=42)
    
    features = np.array(sample_df['features'].tolist())
    
    # Подготовка данных для визуализации
    sentiment_map = {0.0: "Negative", 1.0: "Neutral", 2.0: "Positive"}
    sample_df = sample_df.copy()
    sample_df['sentiment_pred'] = sample_df['prediction'].map(sentiment_map)
    sample_df['sentiment_true'] = sample_df['label'].map(sentiment_map)
    
    # Создание 3D t-SNE
    logger.info("Применение 3D t-SNE... Это может занять время.")
    tsne_3d = TSNE(n_components=3, perplexity=30, max_iter=500, random_state=42, 
                   learning_rate='auto', init='random')
    embeddings_3d = tsne_3d.fit_transform(features)
    
    # Добавляем координаты в DataFrame
    sample_df['x'] = embeddings_3d[:, 0]
    sample_df['y'] = embeddings_3d[:, 1]
    sample_df['z'] = embeddings_3d[:, 2]
    
    # Создание интерактивного 3D графика
    logger.info("Создание интерактивного 3D графика с помощью Plotly.")
    
    # Цветовая схема для каждого класса
    color_map = {
        "Positive": "#2E8B57",    # Зеленый
        "Negative": "#DC143C",    # Красный
        "Neutral": "#4169E1"      # Синий
    }
    
    # Создание основного 3D scatter plot
    fig = go.Figure()
    
    for sentiment in ["Positive", "Negative", "Neutral"]:
        mask = sample_df['sentiment_pred'] == sentiment
        subset = sample_df[mask]
        
        # Определяем размер точек на основе уверенности модели (если есть)
        marker_size = 6
        
        # Подготавливаем данные для hover
        hover_texts = []
        for idx, row in subset.iterrows():
            # Обрабатываем комментарий: убираем лишние пробелы и переносы, ограничиваем длину
            comment = str(row['processed_comment']).strip().replace('\n', ' ').replace('\r', ' ')
            # Разбиваем длинный текст на строки для лучшего отображения
            if len(comment) > 120:
                words = comment.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) <= 40:
                        current_line += (" " + word) if current_line else word
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                        if len(lines) >= 3:  # Ограничиваем 3 строками
                            lines.append("...")
                            break
                if current_line and len(lines) < 3:
                    lines.append(current_line)
                comment = "<br>".join(lines)
            
            hover_text = (
                f"<b>{sentiment} ({len(subset)} точек)</b><br>"
                f"<b>Комментарий:</b><br>{comment}<br>"
                f"─────────────────────────────<br>"
                f"<b>Истинная метка:</b> {row['sentiment_true']}<br>"
                f"<b>Предсказание:</b> {row['sentiment_pred']}<br>"
                f"<b>Координаты:</b> ({row['x']:.2f}, {row['y']:.2f}, {row['z']:.2f})"
            )
            hover_texts.append(hover_text)
        
        fig.add_trace(go.Scatter3d(
            x=subset['x'],
            y=subset['y'],
            z=subset['z'],
            mode='markers',
            name=f'{sentiment} ({len(subset)} точек)',
            marker=dict(
                size=marker_size,
                color=color_map[sentiment],
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            hoverinfo='text'
        ))
    
    # Настройка макета
    fig.update_layout(
        title={
            'text': "3D Визуализация эмбеддингов комментариев (t-SNE)<br><sub>Интерактивная карта тональности текстов</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        scene=dict(
            xaxis_title="t-SNE Компонента 1",
            yaxis_title="t-SNE Компонента 2", 
            zaxis_title="t-SNE Компонента 3",
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="lightgray",
                showbackground=True,
                zerolinecolor="lightgray",
            ),
            yaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="lightgray",
                showbackground=True,
                zerolinecolor="lightgray",
            ),
            zaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="lightgray",
                showbackground=True,
                zerolinecolor="lightgray",
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1200,
        height=800,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=0, t=80),
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    # Сохранение основного 3D графика
    path_3d = "artifacts/embeddings_3d_visualization.html"
    fig.write_html(path_3d)
    logger.info(f"3D интерактивный график сохранен в '{path_3d}'")
    
    # Создание дополнительного 2D графика с проекциями
    logger.info("Создание дополнительных 2D проекций...")
    
    # Создание subplot с тремя проекциями
    fig_2d = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Проекция XY', 'Проекция XZ', 'Проекция YZ', 'Статистика'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Проекция XY
    for sentiment in ["Positive", "Negative", "Neutral"]:
        mask = sample_df['sentiment_pred'] == sentiment
        subset = sample_df[mask]
        
        # Подготавливаем hover тексты для 2D проекций
        hover_texts_2d = []
        for idx, row in subset.iterrows():
            # Обрабатываем комментарий для 2D (короче, чем для 3D)
            comment = str(row['processed_comment']).strip().replace('\n', ' ').replace('\r', ' ')
            if len(comment) > 80:
                words = comment.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) <= 35:
                        current_line += (" " + word) if current_line else word
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                        if len(lines) >= 2:  # Ограничиваем 2 строками для 2D
                            lines.append("...")
                            break
                if current_line and len(lines) < 2:
                    lines.append(current_line)
                comment = "<br>".join(lines)
            
            hover_text = (
                f"<b>{sentiment}</b><br>"
                f"<b>Комментарий:</b><br>{comment}<br>"
                f"<b>Истинная метка:</b> {row['sentiment_true']}<br>"
                f"<b>Предсказание:</b> {row['sentiment_pred']}"
            )
            hover_texts_2d.append(hover_text)
        
        fig_2d.add_trace(
            go.Scatter(
                x=subset['x'], y=subset['y'],
                mode='markers',
                name=sentiment,
                marker=dict(color=color_map[sentiment], size=4, opacity=0.7),
                showlegend=True if sentiment == "Positive" else False,
                text=hover_texts_2d,
                hovertemplate='%{text}<extra></extra>',
                hoverinfo='text'
            ),
            row=1, col=1
        )
        
        # Проекция XZ
        fig_2d.add_trace(
            go.Scatter(
                x=subset['x'], y=subset['z'],
                mode='markers',
                name=sentiment,
                marker=dict(color=color_map[sentiment], size=4, opacity=0.7),
                showlegend=False,
                text=hover_texts_2d,
                hovertemplate='%{text}<extra></extra>',
                hoverinfo='text'
            ),
            row=1, col=2
        )
        
        # Проекция YZ
        fig_2d.add_trace(
            go.Scatter(
                x=subset['y'], y=subset['z'],
                mode='markers',
                name=sentiment,
                marker=dict(color=color_map[sentiment], size=4, opacity=0.7),
                showlegend=False,
                text=hover_texts_2d,
                hovertemplate='%{text}<extra></extra>',
                hoverinfo='text'
            ),
            row=2, col=1
        )
    
    # Статистика распределения
    sentiment_counts = sample_df['sentiment_pred'].value_counts()
    fig_2d.add_trace(
        go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker_color=[color_map[sent] for sent in sentiment_counts.index],
            showlegend=False,
            text=sentiment_counts.values,
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig_2d.update_layout(
        title_text="2D Проекции 3D эмбеддингов и статистика",
        height=800,
        showlegend=True
    )
    
    # Сохранение 2D проекций
    path_2d = "artifacts/embeddings_2d_projections.html"
    fig_2d.write_html(path_2d)
    logger.info(f"2D проекции сохранены в '{path_2d}'")
    
    # Создание сводного отчета
    create_embedding_report(sample_df, embeddings_3d)

def create_embedding_report(df, embeddings_3d):
    """Создает сводный отчет о визуализации эмбеддингов"""
    logger.info("Создание сводного отчета о визуализации...")
    
    # Вычисление статистик
    stats = {
        'total_points': len(df),
        'positive_count': len(df[df['sentiment_pred'] == 'Positive']),
        'negative_count': len(df[df['sentiment_pred'] == 'Negative']),
        'neutral_count': len(df[df['sentiment_pred'] == 'Neutral']),
        'embedding_variance': np.var(embeddings_3d, axis=0),
        'embedding_range': np.ptp(embeddings_3d, axis=0)
    }
    
    # Создание HTML отчета
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
        <title>Отчет о визуализации эмбеддингов</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
            .container {{ background: white; border-radius: 15px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }}
            .header {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; }}
            .stats {{ display: flex; justify-content: space-around; margin: 20px 0; flex-wrap: wrap; }}
            .stat-box {{ 
                background: linear-gradient(135deg, #e8f4f8 0%, #d1ecf1 100%); 
                padding: 15px; 
                border-radius: 8px; 
                text-align: center; 
                min-width: 120px;
                margin: 5px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .links {{ margin-top: 30px; }}
            .link-button {{ 
                display: inline-block; 
                padding: 12px 24px; 
                margin: 10px; 
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); 
                color: white; 
                text-decoration: none; 
                border-radius: 8px; 
                transition: transform 0.2s;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }}
            .link-button:hover {{ transform: translateY(-2px); }}
            .description {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .instructions {{ background: #e8f5e8; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .server-info {{ 
                background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); 
                padding: 15px; 
                border-radius: 8px; 
                margin: 20px 0; 
                text-align: center;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Отчет о визуализации эмбеддингов</h1>
                <p>Интерактивная 3D визуализация тональности текстовых комментариев</p>
            </div>
            
            <div class="server-info">
                🌐 Веб-сервер запущен! Все файлы доступны по адресу: <strong>http://localhost:8000</strong>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <h3>Всего точек</h3>
                    <h2>{stats['total_points']}</h2>
                </div>
                <div class="stat-box">
                    <h3>Позитивные</h3>
                    <h2 style="color: #2E8B57;">{stats['positive_count']}</h2>
                </div>
                <div class="stat-box">
                    <h3>Негативные</h3>
                    <h2 style="color: #DC143C;">{stats['negative_count']}</h2>
                </div>
                <div class="stat-box">
                    <h3>Нейтральные</h3>
                    <h2 style="color: #4169E1;">{stats['neutral_count']}</h2>
                </div>
            </div>
            
            <div class="description">
                <h2>🎯 Описание визуализации</h2>
                <ul>
                    <li><strong>3D t-SNE:</strong> Трехмерное представление высокоразмерных эмбеддингов</li>
                    <li><strong>Цветовое кодирование:</strong> Зеленый - позитивные, красный - негативные, синий - нейтральные</li>
                    <li><strong>Интерактивность:</strong> Наведите курсор на точку для просмотра комментария</li>
                    <li><strong>Кластеризация:</strong> Близкие точки имеют схожую семантику</li>
                </ul>
            </div>
            
            <div class="links">
                <h2>🔗 Ссылки на визуализации</h2>
                <a href="embeddings_3d_visualization.html" class="link-button" target="_blank">🎮 3D Интерактивная визуализация</a>
                <a href="embeddings_2d_projections.html" class="link-button" target="_blank">📈 2D Проекции</a>
                <a href="sentiment_distribution.png" class="link-button" target="_blank">📊 Распределение тональности</a>
                <a href="confusion_matrix.png" class="link-button" target="_blank">🎯 Матрица ошибок</a>
            </div>
            
            <div class="instructions">
                <h2>📋 Инструкция по использованию</h2>
                <ol>
                    <li>Откройте <strong>3D Интерактивную визуализацию</strong> для основного анализа</li>
                    <li>Используйте мышь для поворота, масштабирования и перемещения по 3D пространству</li>
                    <li>Наводите курсор на точки для просмотра содержимого комментариев</li>
                    <li>Используйте легенду для скрытия/показа различных классов</li>
                    <li>Анализируйте кластеры: близкие точки имеют схожую семантику</li>
                    <li><strong>Веб-сервер будет работать до завершения программы</strong></li>
                </ol>
            </div>
            
            <div style="text-align: center; margin-top: 30px; color: #666;">
                <p>💡 <em>Для остановки сервера нажмите Ctrl+C в терминале</em></p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open("artifacts/visualization_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info("Сводный отчет сохранен в 'artifacts/visualization_report.html'")

def start_web_server():
    """Запускает локальный веб-сервер для просмотра HTML файлов"""
    port = 8000
    
    def run_server():
        try:
            # Переходим в директорию artifacts
            os.chdir("artifacts")
            
            # Создаем HTTP сервер
            handler = http.server.SimpleHTTPRequestHandler
            
            with socketserver.TCPServer(("", port), handler) as httpd:
                logger.info(f"🌐 Веб-сервер запущен на http://localhost:{port}")
                logger.info("📁 Обслуживаются файлы из директории 'artifacts'")
                logger.info("🔗 Основные ссылки:")
                logger.info(f"   • Главная страница: http://localhost:{port}/visualization_report.html")
                logger.info(f"   • 3D визуализация: http://localhost:{port}/embeddings_3d_visualization.html")
                logger.info(f"   • 2D проекции: http://localhost:{port}/embeddings_2d_projections.html")
                logger.info("⚠️  Для остановки сервера нажмите Ctrl+C")
                
                # Открываем браузер с главной страницей
                webbrowser.open(f"http://localhost:{port}/visualization_report.html")
                
                # Запускаем сервер
                httpd.serve_forever()
                
        except OSError as e:
            if e.errno == 98:  # Address already in use
                logger.warning(f"Порт {port} уже занят. Попробуйте другой порт или завершите процесс, использующий этот порт.")
                # Пробуем альтернативные порты
                for alt_port in [8001, 8002, 8003, 8080]:
                    try:
                        with socketserver.TCPServer(("", alt_port), handler) as httpd:
                            logger.info(f"🌐 Веб-сервер запущен на http://localhost:{alt_port}")
                            webbrowser.open(f"http://localhost:{alt_port}/visualization_report.html")
                            httpd.serve_forever()
                            break
                    except OSError:
                        continue
            else:
                logger.error(f"Ошибка при запуске веб-сервера: {e}")
        except KeyboardInterrupt:
            logger.info("🛑 Веб-сервер остановлен пользователем")
        except Exception as e:
            logger.error(f"Неожиданная ошибка веб-сервера: {e}")
    
    # Запускаем сервер в отдельном потоке
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    logger.info("🚀 Веб-сервер запускается в фоновом режиме...")
    logger.info("📖 Браузер откроется автоматически с главной страницей отчета")
    
    # Ждем немного, чтобы сервер успел запуститься
    import time
    time.sleep(2)
    
    # Показываем пользователю, что делать дальше
    print("\n" + "="*60)
    print("🎉 ВИЗУАЛИЗАЦИЯ ГОТОВА!")
    print("="*60)
    print(f"🌐 Веб-сервер: http://localhost:8000")
    print(f"📊 Главная страница: http://localhost:8000/visualization_report.html")
    print(f"🎮 3D визуализация: http://localhost:8000/embeddings_3d_visualization.html")
    print("="*60)
    print("💡 Сервер работает в фоновом режиме")
    print("⚠️  Для остановки нажмите Ctrl+C")
    print("="*60 + "\n")
    
    # Держим основной поток активным
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Программа завершена пользователем")

if __name__ == "__main__":
    main()


