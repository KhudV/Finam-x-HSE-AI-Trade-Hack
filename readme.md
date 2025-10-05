Этот репозиторий содержит решение по треку RADAR Finam x HSE AI Trade Hack.
# Установка библиотек:
pip install --upgrade pip

pip install -r requirements.txt
# Установка моделей spaCy:
### легкие модели (быстро)

python -m spacy download en_core_news_sm

python -m spacy download ru_core_news_sm

### опционально: transformer модель для английского языка, большая для русского (лучшее качество, больше ресурсов)

python -m spacy download en_core_web_trf

python -m spacy download ru_core_news_lg

# Установка переменных окружения

Установить API-ключ в переменную окружения OPENROUTER_API_KEY

# Устранение неполадок

файл main.py
строка 297 Заменить "hotness": hot_res.get("hotness", 0.0) на "hotness": hot_res
строка 298 Закомментировать

# Пример запуска

python main.py 2025-10-01T00:00:00 2025-10-02T00:00:00 20