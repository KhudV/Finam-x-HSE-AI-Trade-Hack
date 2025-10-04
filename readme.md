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