from datetime import datetime, timedelta
from news_parser import get_financial_news, DEFAULT_FINANCE_FEEDS

end = datetime.utcnow()
start = end - timedelta(hours=24)   # последние 24 часа

news = get_financial_news(start.isoformat(), end.isoformat(),
                          feed_urls=DEFAULT_FINANCE_FEEDS,
                          max_workers=6,
                          fetch_text=True)

print(f"Fetched {len(news)} items")
for n in news[:10]:
    print(n['published'], n['source'], n['title'], n['url'][:80])
