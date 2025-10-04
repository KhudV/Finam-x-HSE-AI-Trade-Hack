# news_parser.py
import feedparser
import requests
from newspaper import Article
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from dateutil import tz
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Iterable, Union, Optional
import hashlib
import time
from tqdm import tqdm

# ---------------- helpers ----------------
def parse_time(t: Union[str, datetime]) -> datetime:
    """Parse ISO string or pass-through datetime. Return timezone-aware UTC datetime."""
    if isinstance(t, datetime):
        dt = t
    else:
        dt = dateparser.parse(str(t))
    if dt.tzinfo is None:
        # assume local -> convert to UTC
        dt = dt.replace(tzinfo=tz.tzlocal())
    return dt.astimezone(tz.tzutc())

def normalize_url(u: str) -> str:
    # simple normalization for deduplication
    return u.split('?')[0].rstrip('/')

def fingerprint(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

# --------------- feed fetching ----------------
def fetch_feed_entries(feed_url: str, start_utc: datetime, end_utc: datetime, max_items: int = 200) -> List[Dict]:
    """
    Pull entries from RSS/Atom (feedparser). Filter by published/updated time.
    Returns list of dicts: {title, link, summary, published (datetime), source}
    """
    out = []
    try:
        parsed = feedparser.parse(feed_url)
        source_title = parsed.feed.get('title') or feed_url
        entries = parsed.entries[:max_items]
        for e in entries:
            # try multiple date fields
            date_str = None
            if 'published' in e:
                date_str = e.get('published')
            elif 'updated' in e:
                date_str = e.get('updated')
            elif 'pubDate' in e:
                date_str = e.get('pubDate')
            else:
                # sometimes feedparser gives structured time
                if e.get('published_parsed'):
                    # convert published_parsed (struct_time) to ISO
                    try:
                        published_ts = time.mktime(e.published_parsed)
                        date_str = datetime.utcfromtimestamp(published_ts).isoformat()
                    except Exception:
                        date_str = None

            if date_str:
                try:
                    published = parse_time(date_str)
                except Exception:
                    published = None
            else:
                published = None

            # If no date, keep but mark as None (filter out later if needed)
            if published is None:
                # optional: skip entries with no date
                continue

            # time window filter
            if published < start_utc or published > end_utc:
                continue

            title = e.get('title', '').strip()
            link = e.get('link') or e.get('id') or ''
            summary = e.get('summary') or e.get('description') or ''

            out.append({
                'title': title,
                'link': link,
                'summary': summary,
                'published': published,
                'source': source_title
            })
    except Exception as ex:
        # log and return empty
        print(f"[feed error] {feed_url} -> {ex}")
    return out

# --------------- article text extraction ----------------
def extract_article_text(url: str, timeout: int = 10) -> str:
    """
    Try newspaper3k first; if it fails, fallback to requests + BeautifulSoup main text extraction.
    Returns extracted text (could be empty string on failure).
    """
    # Try newspaper
    try:
        art = Article(url, language='en')
        art.download()
        art.parse()
        text = art.text or ''
        if text.strip():
            return text.strip()
    except Exception:
        pass

    # Fallback: basic requests + heuristics
    try:
        resp = requests.get(url, timeout=timeout, headers={'User-Agent': 'news-parser/1.0 (+https://example.com)'})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        # Heuristic: prefer article tags, else big text blocks
        article_tag = soup.find('article')
        if article_tag:
            text = ' '.join(p.get_text(separator=' ', strip=True) for p in article_tag.find_all('p'))
            if text.strip():
                return text.strip()

        # fallback: gather <p> from main container
        ps = soup.find_all('p')
        if ps:
            # join largest contiguous block of <p>
            paragraphs = [p.get_text(separator=' ', strip=True) for p in ps]
            text = '\n\n'.join(paragraphs)
            return text.strip()
    except Exception:
        pass

    return ""

# --------------- main orchestrator ----------------
def get_financial_news(start: Union[str, datetime],
                       end: Union[str, datetime],
                       feed_urls: Optional[Iterable[str]] = None,
                       max_workers: int = 8,
                       max_items_per_feed: int = 500,
                       fetch_text: bool = True) -> List[Dict]:
    """
    Main entrypoint.
    start, end: ISO strings or datetimes
    feed_urls: iterable of RSS/Atom URLs. If None, you must provide a list.
    fetch_text: if True, tries to download full article text (slower).
    Returns deduplicated list of news dicts:
      {title, text, url, published (datetime UTC), source}
    """
    if feed_urls is None:
        raise ValueError("feed_urls must be provided (list of RSS/Atom URLs)")

    start_utc = parse_time(start)
    end_utc = parse_time(end)

    # Step 1: collect entries from feeds (concurrently)
    entries = []
    with ThreadPoolExecutor(max_workers=min(len(list(feed_urls)), max_workers)) as ex:
        futures = {ex.submit(fetch_feed_entries, url, start_utc, end_utc, max_items_per_feed): url for url in feed_urls}
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                res = fut.result()
                entries.extend(res)
            except Exception as e:
                print(f"[feed-fail] {url} -> {e}")

    # Deduplicate by normalized URL (and by title fingerprint fallback)
    seen_urls = set()
    seen_title_fp = set()
    candidates = []
    for e in entries:
        link = normalize_url(e.get('link','') or '')
        title = e.get('title','') or ''
        if link:
            if link in seen_urls:
                continue
            seen_urls.add(link)
        else:
            fp = fingerprint(title)  # fallback
            if fp in seen_title_fp:
                continue
            seen_title_fp.add(fp)
        candidates.append(e)

    # Optionally fetch full texts concurrently
    results = []
    if fetch_text:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for e in candidates:
                link = e.get('link','')
                futures[ex.submit(extract_article_text, link)] = e
            for fut in tqdm(as_completed(futures), total=len(futures), desc="fetching articles"):
                e = futures[fut]
                try:
                    text = fut.result()
                except Exception as exn:
                    text = ""
                results.append({
                    'title': e.get('title',''),
                    'text': text or e.get('summary',''),
                    'url': e.get('link',''),
                    'published': e.get('published'),
                    'source': e.get('source','')
                })
    else:
        for e in candidates:
            results.append({
                'title': e.get('title',''),
                'text': e.get('summary',''),
                'url': e.get('link',''),
                'published': e.get('published'),
                'source': e.get('source','')
            })

    # Final sort by published desc
    results.sort(key=lambda x: x.get('published') or datetime.min.replace(tzinfo=tz.tzutc()), reverse=True)
    return results

# ---------------- RSS list ----------------
DEFAULT_FINANCE_FEEDS = [
    "http://feeds.reuters.com/reuters/businessNews",
    "https://www.ft.com/?format=rss",               # Financial Times
    "https://www.bloomberg.com/feed/podcast/etf-report.xml",
    "https://seekingalpha.com/feed.xml",            # Seeking Alpha
    "https://www.investing.com/rss/news_25.rss",    # Investing.com
    "https://www.wsj.com/xml/rss/3_7031.xml",       # WSJ markets
    "https://finance.yahoo.com/news/rssindex",       # Yahoo Finance
    "https://rssexport.rbc.ru/rbcnews/economics/20/full.rss", # РБК
    "https://www.forbes.ru/newrss.xml", # Forbes
    "https://www.finam.ru/analysis/conews/rsspoint/", # Finam
    "https://www.investfunds.ru/news/rss/", # Investfunds
    "https://www.banki.ru/xml/news.rss" # Banki.ru

]
