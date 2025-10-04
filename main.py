from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import heapq
import re

from news_parser import get_financial_news, DEFAULT_FINANCE_FEEDS
from dedupe import dedupe_articles  # возвращает annotated_articles, clusters_meta
import os
from draft_generator import generate_draft_for_event, make_client_from_env
from openai import OpenAI


try:
    _OPENAI_CLIENT = make_client_from_env(api_key_env="OPENROUTER_API_KEY", base_url="https://openrouter.ai/api/v1")
except Exception as e:
    _OPENAI_CLIENT = None
    import logging
    logging.getLogger(__name__).warning("OpenAI client not initialized: %s", e)


# ---------- Настройки / словари ----------
# Ключевые слова для детекции событий в тексте (англ + рус)
CONFIRM_KEYWORDS = [
    "confirmed", "confirms", "confirm", "according to", "подтверд", "сообщает", "сообщили", "передает",
    "has confirmed", "was confirmed", "confirmed by"
]
UPDATE_KEYWORDS = [
    "update", "updated", "апдейт", "уточн", "исправлен", "correction", "clarification", "amend", "revise",
    "revised", "updated at", "обновлен", "изменен", "изменение"
]
RETRACTION_KEYWORDS = ["retract", "retracted", "отозв", "отмен", "отказ"]
# Words suggesting press release / official statement
PRESS_KEYWORDS = ["press release", "press-release", "press release:", "пресс-служба", "пресс-релиз", "press service"]

# Domains considered high-credibility (can be extended)
HIGH_CRED_DOMAINS = [
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com", "financialtimes.com",
    "tass.ru", "ria.ru", "rbc.ru", "kommersant.ru", "vedomosti.ru", "interfax.ru"
]

# Helper: normalize domain
def domain_of(url: str) -> str:
    try:
        p = urlparse(url)
        d = p.netloc.lower()
        # strip www.
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""

# Helper: simple credibility by domain presence
def source_cred_score(url: str) -> float:
    d = domain_of(url)
    for hd in HIGH_CRED_DOMAINS:
        if hd in d:
            return 1.0
    # default low-medium
    return 0.4

# Lower-case joined text for fast search
def combined_text_for_search(title: str, text: str) -> str:
    s = (title or "") + " " + (text or "")
    return s.lower()

# Check presence of any keyword from list in text (fast)
def contains_keyword(text: str, keywords: List[str]) -> bool:
    lt = text.lower()
    for kw in keywords:
        if kw in lt:
            return True
    return False

# Select representative headline for cluster:
# preference: article with highest source_cred_score, tie-breaker earliest publish
def choose_headline_for_cluster(articles: List[Dict[str, Any]]) -> str:
    best = None
    best_score = -1.0
    for a in articles:
        url = a.get("url") or ""
        score = source_cred_score(url)
        pub = a.get("published") or datetime.min
        # composite key (score, negative timestamp so earliest preferred when same score)
        key = (score, pub)
        if score > best_score:
            best_score = score
            best = a
    if best:
        return best.get("title") or best.get("text", "")[:120]
    # fallback
    return articles[0].get("title") or articles[0].get("text","")[:120]

# Build sources list for cluster:
# - earliest (first report)
# - add other high-cred sources that mention event (confirmation)
# - add updates/clarifications if present (detected by keywords)
# limit to max_sources
def build_sources_and_timeline_for_cluster(cluster_id: int,
                                          cluster_article_indices: List[int],
                                          annotated_articles: List[Dict[str, Any]],
                                          max_sources: int = 5) -> Tuple[List[str], List[Dict[str, Any]]]:
    # gather articles in this cluster (annotated_articles is list of article dicts with dedup_group_id)
    arts = [a for a in annotated_articles if a.get("dedup_group_id") == cluster_id]
    if not arts:
        return [], []
    # sort by published asc
    arts_sorted = sorted(arts, key=lambda x: x.get("published") or datetime.min)
    # timeline events collector
    timeline: List[Dict[str, Any]] = []
    sources_set = []
    # first report = earliest
    first = arts_sorted[0]
    first_url = first.get("url") or ""
    first_source = first.get("source") or domain_of(first_url)
    first_ts = first.get("published")
    timeline.append({"ts": first_ts, "type": "first_report", "source": first_source, "url": first_url, "note": first.get("title")})
    sources_set.append(first_url)

    # scan others for confirmations / updates / clarifications / retractions
    confirmations = []
    updates = []
    retractions = []
    others = []
    for a in arts_sorted[1:]:
        url = a.get("url") or ""
        title = a.get("title") or ""
        text = a.get("text") or ""
        combined = combined_text_for_search(title, text)
        # high credibility check
        cred = source_cred_score(url)
        # if contains confirm keywords OR source is high cred and not the first source -> confirmation
        if contains_keyword(combined, CONFIRM_KEYWORDS) or (cred >= 0.9 and url != first_url):
            confirmations.append((a, cred))
            timeline.append({"ts": a.get("published"), "type": "confirmation", "source": a.get("source") or domain_of(url), "url": url, "note": title})
        elif contains_keyword(combined, UPDATE_KEYWORDS):
            updates.append((a, cred))
            timeline.append({"ts": a.get("published"), "type": "update", "source": a.get("source") or domain_of(url), "url": url, "note": title})
        elif contains_keyword(combined, RETRACTION_KEYWORDS):
            retractions.append((a, cred))
            timeline.append({"ts": a.get("published"), "type": "retraction", "source": a.get("source") or domain_of(url), "url": url, "note": title})
        else:
            others.append((a, cred))
        # add url if not present to sources_set
        if url and url not in sources_set:
            sources_set.append(url)

    # Build prioritized source list: earliest + confirmations (high cred) + updates (if any) + other high cred
    # Use heap to pick top by credibility (but preserve first)
    prioritized = [first_url]
    # confirmations sorted by cred desc
    conf_sorted = sorted(confirmations, key=lambda x: (-x[1], x[0].get("published") or datetime.min))
    for a, cred in conf_sorted:
        if a.get("url") not in prioritized:
            prioritized.append(a.get("url"))
    # updates next
    upd_sorted = sorted(updates, key=lambda x: (-x[1], x[0].get("published") or datetime.min))
    for a, cred in upd_sorted:
        if a.get("url") not in prioritized:
            prioritized.append(a.get("url"))
    # fill remaining with other high-cred sources
    other_sorted = sorted(others, key=lambda x: (-x[1], x[0].get("published") or datetime.min))
    for a, cred in other_sorted:
        if a.get("url") not in prioritized:
            prioritized.append(a.get("url"))
    # cap to max_sources
    prioritized = prioritized[:max_sources]

    # ensure timeline sorted by ts asc
    timeline = sorted(timeline, key=lambda x: (x["ts"] or datetime.min))

    # dedupe timeline by url (keep earliest type if duplicates)
    seen_urls = set()
    compact_timeline = []
    for ev in timeline:
        u = ev.get("url") or ""
        if u and u in seen_urls:
            continue
        seen_urls.add(u)
        compact_timeline.append(ev)

    return prioritized, compact_timeline

# Aggregate entities for cluster: prefer clusters_meta if available in dedupe output; otherwise collect from article-level entities
def aggregate_entities_for_cluster(cluster_id: int,
                                   clusters_meta: Dict[int, Dict[str, Any]],
                                   annotated_articles: List[Dict[str, Any]],
                                   top_n: int = 20) -> List[Dict[str, Any]]:
    # try clusters_meta
    meta = clusters_meta.get(cluster_id)
    if meta and "entities" in meta and meta["entities"]:
        # return entities as-is but filter to only allowed types and non-empty names
        ents = []
        for e in meta["entities"]:
            name = e.get("name")
            etype = e.get("type")
            ticker = e.get("ticker")
            if not name:
                continue
            if etype not in ("company", "ticker", "country", "sector"):
                continue
            ents.append({"name": name, "type": etype, "ticker": ticker})
        return ents[:top_n]

    # fallback: aggregate from annotated_articles
    counter = {}
    for a in annotated_articles:
        if a.get("dedup_group_id") != cluster_id:
            continue
        for e in a.get("entities", []) or []:
            name = e.get("name")
            etype = e.get("type")
            ticker = e.get("ticker")
            if not name or etype not in ("company", "ticker", "country", "sector"):
                continue
            key = (name, etype, ticker)
            counter[key] = counter.get(key, 0) + 1
    # sort by count desc
    items = sorted(counter.items(), key=lambda kv: -kv[1])
    ents = [{"name": k[0], "type": k[1], "ticker": k[2], "count": v} for (k, v) in items]
    # drop "count" for returned format (user asked only entities list) — but keep ticker if present
    out = [{"name": e["name"], "type": e["type"], "ticker": e.get("ticker")} for e in ents]
    return out[:top_n]

# Main function: given start/end, return list of events with required fields
def extract_events_for_interval(start: str,
                                end: str,
                                feed_urls: Optional[List[str]] = None,
                                max_workers: int = 6,
                                fetch_text: bool = True,
                                similarity_threshold: float = 0.78,
                                model_name: str = "all-MiniLM-L6-v2",
                                use_sentence_transformers: Optional[bool] = None
                                ) -> List[Dict[str, Any]]:
    """
    start, end: ISO datetime strings or datetimes (strings recommended)
    returns: list of event dicts with keys: dedup_group, headline, entities, sources, timeline
    """
    if feed_urls is None:
        feed_urls = DEFAULT_FINANCE_FEEDS

    # 1) fetch news
    news = get_financial_news(start, end, feed_urls=feed_urls, max_workers=max_workers, fetch_text=fetch_text)
    if not news:
        return []

    # 2) prepare articles for dedupe
    articles = []
    for n in news:
        articles.append({
            "title": n.get("title",""),
            "text": n.get("text",""),
            "url": n.get("url",""),
            "published": n.get("published"),
            "source": n.get("source")
        })

    # 3) dedupe / cluster
    annotated_articles, clusters_meta = dedupe_articles(articles,
                                                        similarity_threshold=similarity_threshold,
                                                        model_name=model_name,
                                                        use_sentence_transformers=use_sentence_transformers)

    # compute set of cluster ids
    cluster_ids = sorted({a.get("dedup_group_id") for a in annotated_articles if a.get("dedup_group_id") is not None})

    events = []
    for cid in cluster_ids:
        # compute cluster article indices (optional)
        cluster_article_indices = [i for i,a in enumerate(annotated_articles) if a.get("dedup_group_id")==cid]
        # choose headline
        cluster_articles = [a for a in annotated_articles if a.get("dedup_group_id")==cid]
        if not cluster_articles:
            continue
        headline = choose_headline_for_cluster(cluster_articles)
        # entities aggregated
        entities = aggregate_entities_for_cluster(cid, clusters_meta, annotated_articles)
        # sources and timeline with detection of confirmations/updates
        sources, timeline = build_sources_and_timeline_for_cluster(cid, cluster_article_indices, annotated_articles, max_sources=5)

        event = {
            "dedup_group": cid,
            "headline": headline,
            "entities": entities,
            "sources": sources,
            "timeline": timeline
        }

        # Generate draft: title separately, rest as text
        draft_obj = {"title": None, "text": None, "raw": None}
        try:
            if _OPENAI_CLIENT is not None:
                dg = generate_draft_for_event(event, client=_OPENAI_CLIENT, model="openai/gpt-5", temperature=0.0)
                # dg has keys: title, text, raw
                draft_obj = dg
            else:
                # try to create client from env as fallback
                try:
                    client_tmp = make_client_from_env(api_key_env="OPENROUTER_API_KEY", base_url="https://openrouter.ai/api/v1")
                    draft_obj = generate_draft_for_event(event, client=client_tmp, model="openai/gpt-5", temperature=0.0)
                except Exception:
                    draft_obj = {"title": None, "text": None, "raw": None}
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Draft generation failed for cluster %s: %s", cid, e)
            draft_obj = {"title": None, "text": None, "raw": None}

        event["draft"] = draft_obj
        events.append(event)
    return events

# -------------------- demo / example usage --------------------
if __name__ == "__main__":
    from datetime import datetime, timedelta
    end = datetime.utcnow()
    start = end - timedelta(hours=24)
    print("Running extractor for last 24 hours...")
    evs = extract_events_for_interval(start.isoformat(), end.isoformat(), max_workers=6, fetch_text=True)
    print(f"Found {len(evs)} events (clusters).")
    for e in evs[:10]:
        print("=== Cluster", e["dedup_group"], "===")
        print("Headline:", e["headline"])
        print("Entities:", e["entities"])
        print("Sources:", e["sources"])
        print("Timeline:")
        for t in e["timeline"]:
            print("  ", t["ts"], t["type"], t["source"], t["url"])
        print()
