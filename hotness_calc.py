import math
import datetime
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import sys
import os

base_path, _ = os.path.split(os.getcwd())
if base_path not in sys.path:
    sys.path.append(base_path)

from dedupe import extract_entities_batch, EmbeddingBackend
# Константы
WEIGHTS = {
    'surprise': 0.25,
    'materiality': 0.3,
    'velocity': 0.2,
    'coverage': 0.15,
    'credibility': 0.1
}

# Каждому источнику присвается рейтинг от 1 до 10
SOURCE_RATINGS = {
    'Stock Market News': 9,
    'Yahoo Finance': 9,
    'UK homepage': 8,
    'All Articles on Seeking Alpha': 8
}

def get_embedding(text: str) -> np.ndarray:
    """
    Получение эмбеддинга текста.
    """
    encoder = EmbeddingBackend(use_sentence_transformers=True)
    embedding = encoder.encode([text])[1][0]
    return embedding
    

def get_reference_news(entities: List[str]) -> str:
    """
    Получение референсных новостей по списку сущностей.
    В реальной реализации здесь будет запрос к News API или другой базе новостей.
    """
    return " ".join(entities) + " reference news content about market trends and financial analysis."

def calculate_surprise(news_text: str, entities: List[str]) -> float:
    """Расчет неожиданности новости"""
    emb_news = get_embedding(news_text)
    ref_text = get_reference_news(entities)
    emb_refs = get_embedding(ref_text)
    
    similarity = cosine_similarity([emb_news], [emb_refs])[0][0]
    return 1 - similarity

def calculate_materiality(text: str) -> float:
    """
    Расчет материальности на основе тональности и ключевых слов.
    Предлагаемый подход: анализ тональности + наличие ключевых финансовых терминов.
    """
    # Простой анализ на основе ключевых слов
    material_keywords = {
        'earnings': 0.8, 'revenue': 0.7, 'profit': 0.7, 'loss': 0.9,
        'merger': 0.8, 'acquisition': 0.8, 'bankruptcy': 0.95,
        'fed': 0.9, 'interest': 0.8, 'inflation': 0.85,
        'crisis': 0.95, 'growth': 0.6, 'decline': 0.8
    }

    text_lower = text.lower()
    score = 0.0
    max_score = 0.0
    
    for keyword, weight in material_keywords.items():
        if keyword in text_lower:
            score += weight
            max_score += weight
    
    if max_score > 0:
        materiality = min(score / max_score, 1.0)
    else:
        materiality = 0.1  # Базовая материальность
    
    # Усиление материальности для экстремальных тональностей
    sentiment_score = analyze_sentiment(text)
    materiality *= (1 + abs(sentiment_score) * 0.5)
    
    return min(materiality, 1.0)

def analyze_sentiment(text: str) -> float:
    """Упрощенный анализ тональности текста"""
    positive_words = ['good', 'great', 'positive', 'profit', 'growth', 'rise', 'up']
    negative_words = ['bad', 'poor', 'negative', 'loss', 'decline', 'fall', 'down']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    
    return (pos_count - neg_count) / total

def calculate_velocity(cluster_news: List[Dict]) -> float:
    """Расчет скорости распространения новости в кластере"""
    if len(cluster_news) <= 1:
        return 0.0
    
    times = [news['published'] for news in cluster_news]
    min_time = min(times)
    max_time = max(times)
    
    time_diff = (max_time - min_time).total_seconds() / 3600.0  # Разница в часах
    if time_diff == 0:
        return len(cluster_news)  # Если все новости в одно время
    
    return len(cluster_news) / time_diff

def calculate_coverage(entities: List[str]) -> float:
    """Расчет широты охвата на основе количества именованных сущностей"""
    entity_count = len(entities)
    return 0.5 * math.tanh(entity_count)

def calculate_credibility(cluster_news: List[Dict]) -> float:
    """Расчет достоверности на основе рейтингов источников"""
    total_rating = 0
    max_possible = len(cluster_news) * 10  # Максимальный возможный рейтинг
    
    for news in cluster_news:
        source = news['source']
        rating = SOURCE_RATINGS.get(source, 5)  # Средний рейтинг по умолчанию
        total_rating += rating
    
    return total_rating / max_possible

def sigmoid(x: float) -> float:
    """Сигмоидальная функция для нормализации в диапазон [0, 1]"""
    return 1 / (1 + math.exp(-x))

def calculate_hotness_for_cluster(cluster_news: List[Dict]) -> float:
    """Основная функция расчета горячести для кластера новостей"""
    if not cluster_news:
        return 0.0
    
    # Берем первую новость как представителя кластера для некоторых метрик
    representative_news = cluster_news[0]
    full_text = f"{representative_news['title']} {representative_news['text']}"
    
    # Извлекаем сущности
    entities = extract_entities_batch([full_text])[0]
    entities = [ent["name"] for ent in entities]
    
    # Расчет компонентов
    surprise = calculate_surprise(full_text, entities)
    materiality = calculate_materiality(full_text)
    velocity = calculate_velocity(cluster_news)
    coverage = calculate_coverage(entities)
    credibility = calculate_credibility(cluster_news)
    
    # Взвешенная сумма
    weighted_sum = (
        WEIGHTS['surprise'] * surprise +
        WEIGHTS['materiality'] * materiality +
        WEIGHTS['velocity'] * velocity +
        WEIGHTS['coverage'] * coverage +
        WEIGHTS['credibility'] * credibility
    )
    
    # Применяем сигмоиду для нормализации
    hotness = sigmoid(weighted_sum * 10 - 5)  # Масштабируем для лучшего поведения сигмоиды
    
    return hotness

def calculate_hotness_for_all_clusters(news_list: List[Dict]) -> Dict[int, float]:
    """Расчет горячести для всех кластеров"""
    clusters = defaultdict(list)
    
    # Группируем новости по кластерам
    for news in news_list:
        clusters[news['group_id']].append(news)
    
    results = {}
    for cluster_id, cluster_news in clusters.items():
        results[cluster_id] = calculate_hotness_for_cluster(cluster_news)
    
    return results

# Пример использования
if __name__ == "__main__":
    # Пример данных
    sample_news = [
        {
            'group_id': 1,
            'title': 'Company XYZ Reports Strong Quarterly Earnings',
            'text': 'XYZ corporation announced record profits this quarter, exceeding analyst expectations.',
            'url': 'http://example.com/news1',
            'source': 'Bloomberg',
            'published': datetime.datetime(2024, 1, 15, 10, 30, 0)
        },
        {
            'group_id': 1,
            'title': 'XYZ Earnings Beat Estimates',
            'text': 'The company showed impressive growth in revenue and profit margins.',
            'url': 'http://example.com/news2',
            'source': 'Reuters',
            'published': datetime.datetime(2024, 1, 15, 11, 15, 0)
        }
    ]
    
    results = calculate_hotness_for_all_clusters(sample_news)
    print(results)