from datetime import datetime, timedelta
from news_parser import get_financial_news, DEFAULT_FINANCE_FEEDS
from dedupe import dedupe_articles
# ------------- fetch news (last 24h) -------------
end = datetime.utcnow()
start = end - timedelta(hours=24)   # последние 24 часа

print("Fetching news...")
news = get_financial_news(start.isoformat(), end.isoformat(),
                          feed_urls=DEFAULT_FINANCE_FEEDS,
                          max_workers=6,
                          fetch_text=True)

print(f"Fetched {len(news)} items")

# print a few items
for n in news[:10]:
    print(n.get('published'), n.get('source'), n.get('title'), n.get('url')[:120])

if not news:
    print("No news fetched — exiting.")
    exit(0)

# ------------- prepare articles list for dedupe -------------
articles = []
for n in news:
    articles.append({
        'title': n.get('title') or "",
        'text': n.get('text') or "",
        'url': n.get('url') or "",
        'published': n.get('published'),
        'source': n.get('source') or ""
    })

# ------------- run dedupe -------------
print("Running semantic dedupe (dedupe.dedupe_articles)... this may take a while for many items.")
annotated_articles, clusters_meta = dedupe_articles(articles,
                                                    similarity_threshold=0.75,
                                                    model_name="all-MiniLM-L6-v2",
                                                    use_sentence_transformers=True)
print("Clusters meta:", clusters_meta)
for a in annotated_articles:
    print(a['dedup_group_id'], a['title'][:80], a['source'], a['entities'])
