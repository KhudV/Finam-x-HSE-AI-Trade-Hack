# dedupe.py
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np
import html

# ---------- Filtered NER: companies / tickers / countries / sectors only ----------


# ticker and token regexes
TICKER_REGEX = re.compile(r"\$([A-Za-z]{1,6}\b)")            # $AAPL
UPPER_ALNUM_RE = re.compile(r"\b[A-Z0-9]{1,5}\b")

# sector keywords (extendable)
SECTOR_KEYWORDS_EN = [
    "technology", "tech", "financial", "finance", "healthcare", "health care",
    "energy", "utilities", "consumer", "industrials", "materials", "real estate",
    "telecom", "communication", "semiconductor", "banking", "retail", "automotive"
]
SECTOR_KEYWORDS_RU = [
    "технолог", "финанс", "энергетик", "банк", "рознич", "потребительск", "промышлен",
    "недвижим", "полупровод", "телеком", "фармацевт", "нефтегаз", "сельскохозяйств"
]

def is_sector_phrase(text: str) -> bool:
    t = (text or "").lower()
    for kw in SECTOR_KEYWORDS_EN:
        if kw in t:
            return True
    for kw in SECTOR_KEYWORDS_RU:
        if kw in t:
            return True
    return False

def map_spacy_label(label: str) -> str:
    # spaCy labels: ORG, GPE, LOC, NORP, PERSON, DATE, MONEY, PERCENT, ...
    if label in ("ORG",):
        return "company"
    if label in ("GPE", "LOC", "NORP"):
        return "country"
    return "other"

def lookup_ticker_by_name(name: str, ticker_map: dict = None):
    if not ticker_map:
        return None
    return ticker_map.get(name.strip().lower())

# ----------------- spaCy multilingual loading -----------------
SPACY_AVAILABLE = False
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    spacy = None
    SPACY_AVAILABLE = False

# We'll attempt to load best available models per language.
# Priority: trf -> small. If not installed, leave None for fallback.
NER_MODELS = {"en": None, "ru": None}
NER_BACKEND_NAME = {"en": None, "ru": None}

if SPACY_AVAILABLE:
    # English
    try:
        nlp_en = spacy.load("en_core_web_trf")
        NER_MODELS["en"] = nlp_en
        NER_BACKEND_NAME["en"] = "spacy:en_core_web_trf"
    except Exception:
        try:
            nlp_en = spacy.load("en_core_news_sm")
            NER_MODELS["en"] = nlp_en
            NER_BACKEND_NAME["en"] = "spacy:en_core_news_sm"
        except Exception:
            NER_MODELS["en"] = None
            NER_BACKEND_NAME["en"] = None

    # Russian
    try:
        nlp_ru = spacy.load("ru_core_news_lg")
        NER_MODELS["ru"] = nlp_ru
        NER_BACKEND_NAME["ru"] = "spacy:ru_core_news_lg"
    except Exception:
        try:
            nlp_ru = spacy.load("ru_core_news_sm")
            NER_MODELS["ru"] = nlp_ru
            NER_BACKEND_NAME["ru"] = "spacy:ru_core_news_sm"
        except Exception:
            NER_MODELS["ru"] = None
            NER_BACKEND_NAME["ru"] = None

# ----------------- language detection heuristic (fast) -----------------
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

def guess_lang_by_text(text: str) -> str:
    """Very fast heuristic: if text contains Cyrillic -> 'ru', else 'en'."""
    if not text:
        return "en"
    if CYRILLIC_RE.search(text):
        return "ru"
    return "en"

# -------------- cleaning helpers (simple) ----------------
from bs4 import BeautifulSoup
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

def clean_text_for_ner(raw_text: str) -> str:
    """Strip HTML, URLs and unescape entities."""
    if not raw_text:
        return ""
    try:
        soup = BeautifulSoup(raw_text, "lxml")
        text = soup.get_text(separator=" ", strip=True)
    except Exception:
        text = HTML_TAG_RE.sub(" ", raw_text)
    text = URL_RE.sub(" ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_entity_name(name: str) -> str:
    if not name:
        return ""
    s = html.unescape(name).strip()
    s = HTML_TAG_RE.sub("", s)
    s = re.sub(r"^[\W_]+|[\W_]+$", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s)
    return s

def is_valid_entity_name(s: str) -> bool:
    if not s:
        return False
    if len(s) < 2 or len(s) > 120:
        return False
    low = s.lower()
    if "<" in s or ">" in s or "=" in s or "&" in s:
        return False
    if "http" in low or "www." in low:
        return False
    if not re.search(r"[A-Za-zА-Яа-яёЁ]", s):
        return False
    non_word_ratio = sum(1 for ch in s if not re.match(r"[\wа-яА-ЯёЁ]", ch)) / max(1, len(s))
    if non_word_ratio > 0.4:
        return False
    if re.search(r"[;:,/\\]$", s):
        return False
    return True

# ----------------- main multilingual extract_entities_batch -----------------
def extract_entities_batch(texts: List[str], ticker_map: dict = None) -> List[List[Dict[str, Any]]]:
    """
    Multilingual batch NER:
      - splits texts by guessed language (fast heuristic),
      - uses best available spaCy model per language (trf preferred),
      - if model missing for language -> fallback regex heuristics for that language.
    Returns results in same order as input: list of lists of entities,
    where each entity is {"name":..., "type": "company|ticker|country|sector", "ticker": optional}
    """
    n = len(texts)
    results: List[Optional[List[Dict[str, Any]]]] = [None] * n

    # prepare per-language buckets
    buckets = {"en": [], "ru": []}            # list of (idx, cleaned_text)
    for i, t in enumerate(texts):
        # clean text to remove html shards (important before language guess)
        cleaned = clean_text_for_ner(t or "")
        lang = guess_lang_by_text(cleaned)
        buckets[lang].append((i, cleaned))

    # helper: process a bucket with spaCy model (or regex fallback)
    def _process_bucket(lang: str, items: List[tuple]):
        if not items:
            return
        model = NER_MODELS.get(lang)
        if model is not None:
            # run spaCy in batch
            docs = model.pipe([txt for (_, txt) in items], batch_size=32)
            for (idx, _), doc in zip(items, docs):
                ents = []
                seen = set()
                text_full = doc.text or ""
                # spaCy entities
                for e in doc.ents:
                    mapped = map_spacy_label(e.label_)
                    name = normalize_entity_name(e.text)
                    if not is_valid_entity_name(name):
                        continue
                    if mapped == "company":
                        ticker = None
                        if ticker_map:
                            ticker = lookup_ticker_by_name(name, ticker_map) or lookup_ticker_by_name(name.lower(), ticker_map)
                        key = (name.lower(), "company", ticker)
                        if key in seen:
                            continue
                        seen.add(key)
                        ents.append({"name": name, "type": "company", "ticker": ticker})
                    elif mapped == "country":
                        key = (name.lower(), "country", None)
                        if key in seen:
                            continue
                        seen.add(key)
                        ents.append({"name": name, "type": "country", "ticker": None})
                    # ignore other spaCy types

                # explicit $TICKER
                for m in TICKER_REGEX.finditer(text_full):
                    tck = m.group(1).upper()
                    if 2 <= len(tck) <= 6:
                        key = (tck, "ticker", tck)
                        if key not in seen:
                            seen.add(key)
                            ents.append({"name": tck, "type": "ticker", "ticker": tck})

                # uppercase alnum candidates only if in ticker_map values (conservative)
                if ticker_map:
                    ticker_values = set(ticker_map.values())
                    for m in UPPER_ALNUM_RE.finditer(text_full):
                        tok = m.group(0).upper()
                        if 2 <= len(tok) <= 5 and tok in ticker_values:
                            key = (tok, "ticker", tok)
                            if key not in seen:
                                seen.add(key)
                                ents.append({"name": tok, "type": "ticker", "ticker": tok})

                # detect sector mention if keyword present
                if is_sector_phrase(text_full):
                    for kw in SECTOR_KEYWORDS_EN + SECTOR_KEYWORDS_RU:
                        if kw in text_full.lower():
                            snippet = kw if len(kw) <= 40 else kw[:40]
                            if is_valid_entity_name(snippet):
                                key = (snippet.lower(), "sector", None)
                                if key not in seen:
                                    seen.add(key)
                                    ents.append({"name": snippet, "type": "sector", "ticker": None})
                            break

                # final filter: keep only allowed canonical types
                filtered = []
                for ent in ents:
                    et = ent.get("type")
                    if et == "ticker":
                        if ent.get("ticker"):
                            filtered.append({"name": ent["ticker"], "type": "ticker", "ticker": ent["ticker"]})
                    elif et in ("company", "country", "sector"):
                        if is_valid_entity_name(ent.get("name")):
                            filtered.append({"name": ent["name"], "type": et, "ticker": ent.get("ticker")})
                results[idx] = filtered
        else:
            # fallback heuristic for language (regex-based)
            for idx, raw in items:
                text = raw or ""
                ents = []
                seen = set()
                # $TICKER
                for m in TICKER_REGEX.finditer(text):
                    tck = m.group(1).upper()
                    if 2 <= len(tck) <= 6:
                        if ("ticker", tck) not in seen:
                            seen.add(("ticker", tck))
                            ents.append({"name": tck, "type": "ticker", "ticker": tck})
                # countries by keywords (very approximate)
                for country_kw in ["russia", "china", "usa", "united states", "россия", "китай", "сша", "украина", "германия", "франция"]:
                    if country_kw in text.lower():
                        if ("country", country_kw) not in seen:
                            seen.add(("country", country_kw))
                            ents.append({"name": country_kw.title(), "type": "country", "ticker": None})
                # sector keywords
                for kw in SECTOR_KEYWORDS_EN + SECTOR_KEYWORDS_RU:
                    if kw in text.lower():
                        if ("sector", kw) not in seen:
                            seen.add(("sector", kw))
                            ents.append({"name": kw, "type": "sector", "ticker": None})
                        break
                # TitleCase company candidates
                for m in re.finditer(r"([A-ZА-ЯЁ][a-zа-яё0-9]{2,}(?:\s+[A-ZА-ЯЁ][a-zа-яё0-9]{2,}){0,3})", text):
                    cand = normalize_entity_name(m.group(1))
                    if not is_valid_entity_name(cand):
                        continue
                    key = ("company", cand.lower())
                    if key not in seen:
                        seen.add(key)
                        ents.append({"name": cand, "type": "company", "ticker": lookup_ticker_by_name(cand, ticker_map)})
                # filter keep only allowed types and validate
                filtered = []
                for ent in ents:
                    if ent['type'] == "ticker" and ent.get('ticker'):
                        filtered.append({"name": ent['ticker'], "type": "ticker", "ticker": ent['ticker']})
                    elif ent['type'] in ("company", "country", "sector") and is_valid_entity_name(ent['name']):
                        filtered.append({"name": ent['name'], "type": ent['type'], "ticker": ent.get('ticker')})
                results[idx] = filtered

    # process English bucket, then Russian bucket
    _process_bucket("en", buckets["en"])
    _process_bucket("ru", buckets["ru"])

    # any None remaining -> set to empty list
    for i in range(n):
        if results[i] is None:
            results[i] = []
    return results

# Optional libs
try:
    from sentence_transformers import SentenceTransformer
    S_T_AVAILABLE = True
except Exception:
    S_T_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_TFIDF_AVAILABLE = True
except Exception:
    SKLEARN_TFIDF_AVAILABLE = False

# --------------------- helpers ---------------------
def normalize_url(url: str) -> str:
    if not url:
        return ""
    # strip query params and anchors
    u = url.split('#')[0].split('?')[0].rstrip('/')
    return u

def fingerprint_text(text: str, n_chars: int = 200) -> str:
    """Simple fingerprint: sha1 of normalized start of text/title."""
    s = (text or "").strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = s[:n_chars]
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def normalize_text_for_sim(text: str) -> str:
    t = text or ""
    t = t.lower()
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

# --------------------- embeddings utility ---------------------
class EmbeddingBackend:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_sentence_transformers: Optional[bool] = None):
        self.model_name = model_name
        self.model = None
        self.backend = None
        # Try to use sentence-transformers if allowed and installed
        try:
            if S_T_AVAILABLE and (use_sentence_transformers is not False):
                self.model = SentenceTransformer(model_name)
                self.backend = "sentence-transformers"
        except Exception:
            self.model = None
            self.backend = None

        if self.model is None:
            # we'll fallback to sklearn TF-IDF later, mark accordingly
            self.backend = "tfidf-fallback" if SKLEARN_TFIDF_AVAILABLE else "tokencount-fallback"

    def encode(self, texts: List[str]) -> Tuple[str, Any]:
        """
        Returns (backend_name, embeddings)
        - if sentence-transformers: numpy array shape (N, D)
        - if tfidf-fallback: dense numpy matrix shape (N, D)
        - if tokencount fallback: list of lists (vectors)
        """
        if self.backend == "sentence-transformers" and self.model is not None:
            embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return self.backend, embs
        if SKLEARN_TFIDF_AVAILABLE:
            vect = TfidfVectorizer(ngram_range=(1,2), min_df=1)
            X = vect.fit_transform(texts).toarray()
            return "tfidf-fallback", X
        # absolute fallback: simple bag-of-top-tokens counts
        def simple_count_vec(s):
            toks = re.findall(r"[A-Za-zА-Яа-я0-9]+", s.lower())
            c = {}
            for t in toks:
                if len(t) <= 2:
                    continue
                c[t] = c.get(t, 0) + 1
            items = sorted(c.items(), key=lambda x: (-x[1], x[0]))[:50]
            vec = [v for _, v in items]
            if len(vec) < 50:
                vec = vec + [0]*(50-len(vec))
            return vec
        M = [simple_count_vec(t) for t in texts]
        return "tokencount-fallback", M

# --------------------- similarity utils ---------------------
def cosine_sim_vectors(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# --------------------- main dedupe function ---------------------
def dedupe_articles(articles: List[Dict[str, Any]],
                    title_weight: float = 0.4,
                    text_weight: float = 0.6,
                    similarity_threshold: float = 0.78,
                    model_name: str = "all-MiniLM-L6-v2",
                    use_sentence_transformers: Optional[bool] = None
                    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    articles: list of dicts with keys: 'title', 'text', 'url', 'published', 'source' (published optional)
    Returns:
      - annotated_articles: same list with 'dedup_group_id' added (int)
      - clusters: mapping dedup_group_id -> cluster metadata
    Algorithm:
      1. Hard dedup by normalized URL (exact).
      2. Hard dedup by title fingerprint (exact SHA1 of normalized beginning).
      3. Semantic greedy clustering by embeddings + cosine similarity:
         - compute embeddings for concatenated (title * title_weight + text * text_weight)
         - iterate articles, assign to first cluster whose rep has similarity >= threshold,
           otherwise create new cluster.
    Notes:
      - similarity_threshold default 0.78 is a reasonable starting point for headlines+short text.
      - For large batches, consider ANN search (FAISS/Annoy) instead of greedy O(N^2).
    """
    # 0. prepare normalized text fields
    for a in articles:
        a.setdefault('title', '')
        a.setdefault('text', '')
        a.setdefault('url', '')
        a.setdefault('source', '')
        a['_norm_title'] = normalize_text_for_sim(a['title'])
        a['_norm_text'] = normalize_text_for_sim(a['text'])
        a['_norm_url'] = normalize_url(a.get('url',''))
        a['_title_fp'] = fingerprint_text(a['_norm_title'] or (a['_norm_text'][:200]))

    # 1. hard dedup by URL
    url_to_index = {}
    dedup_group = {}
    next_group_id = 0
    annotated = [None]*len(articles)
    for i, a in enumerate(articles):
        u = a['_norm_url']
        if u:
            if u in url_to_index:
                gid = url_to_index[u]
            else:
                gid = next_group_id
                url_to_index[u] = gid
                next_group_id += 1
            dedup_group.setdefault(gid, []).append(i)
            annotated[i] = gid

    # 2. hard dedup by title fingerprint (for those without URL or not matched)
    fp_to_index = {}
    for i, a in enumerate(articles):
        if annotated[i] is not None:
            continue
        fp = a['_title_fp']
        if fp in fp_to_index:
            gid = fp_to_index[fp]
        else:
            gid = next_group_id
            fp_to_index[fp] = gid
            next_group_id += 1
        dedup_group.setdefault(gid, []).append(i)
        annotated[i] = gid

    # Now we'll have groups, but many groups are singletons.
    # For semantic merging we should take one representative per group (first index).
    group_reps = []
    group_rep_idx = []
    for gid, idxs in dedup_group.items():
        rep_idx = idxs[0]
        group_reps.append(representative_text(articles[rep_idx], title_weight, text_weight))
        group_rep_idx.append(rep_idx)

    # If there were no groups (unlikely), create initial groups from annotated array
    if not group_reps:
        # fallback: every article new group
        dedup_group = {}
        annotated = [None]*len(articles)
        group_reps = []
        group_rep_idx = []
        next_group_id = 0
        for i in range(len(articles)):
            dedup_group[next_group_id] = [i]
            annotated[i] = next_group_id
            group_reps.append(representative_text(articles[i], title_weight, text_weight))
            group_rep_idx.append(i)
            next_group_id += 1

    # 3. semantic embeddings for group representatives (to merge similar groups)
    texts_for_embed = group_reps  # strings
    emb_backend = EmbeddingBackend(model_name=model_name, use_sentence_transformers=use_sentence_transformers)
    backend_name, embs = emb_backend.encode(texts_for_embed)

    # convert embs to numpy array if needed
    embs_arr = np.asarray(embs, dtype=float)

    # 4. greedy clustering of group reps by cosine similarity
    rep_count = len(embs_arr)
    rep_assigned = [-1]*rep_count  # mapping rep_idx -> new_cluster_id
    new_clusters = {}  # new_cluster_id -> list of rep indices
    new_cluster_id = 0
    for i in range(rep_count):
        if rep_assigned[i] != -1:
            continue
        # start new cluster with rep i
        new_clusters[new_cluster_id] = [i]
        rep_assigned[i] = new_cluster_id
        # compare to others
        for j in range(i+1, rep_count):
            if rep_assigned[j] != -1:
                continue
            sim = cosine_sim_vectors(embs_arr[i], embs_arr[j])
            # map sim from [-1,1] -> [0,1] for safety (but sentence-transformers usually in [0,1])
            sim_mapped = max(0.0, min(1.0, (sim + 1.0)/2.0))
            if sim_mapped >= similarity_threshold:
                rep_assigned[j] = new_cluster_id
                new_clusters[new_cluster_id].append(j)
        new_cluster_id += 1

    # 5. build final dedup groups: merge original dedup_group indices belonging to same new cluster
    final_clusters = {}  # final_gid -> list of article indices
    gid_map = {}  # old gid -> final_gid
    for old_gid_idx, old_gid in enumerate(list(dedup_group.keys())):
        # map order: dedup_group keys are in insertion order; group_reps correspond to them
        rep_index_position = old_gid_idx  # position in group_reps/embs
        assigned_new = rep_assigned[rep_index_position]
        final_gid = assigned_new  # reuse new_cluster id as final id
        gid_map[old_gid] = final_gid
        # append all article indices from old group to final cluster
        final_clusters.setdefault(final_gid, [])
        final_clusters[final_gid].extend(dedup_group[old_gid])

    # 6. produce annotated articles list with dedup_group_id and cluster meta
    annotated_articles = []
    # --- BEFORE building clusters_meta, we will extract entities for ALL articles once (batch) ---
    all_texts_for_ner = []
    for idx in range(len(articles)):
        # compose a compact text for NER: title + short text
        title = articles[idx].get('title','') or ''
        text = articles[idx].get('text','') or ''
        snippet = (text[:1000] + '...') if len(text) > 1000 else text
        all_texts_for_ner.append(title + "\n" + snippet)

    # run NER in batch
    ner_results = extract_entities_batch(all_texts_for_ner)  # list of list-of-entities
    # attach entities to each article (simple list)
    for i, ents in enumerate(ner_results):
        articles[i].setdefault('entities', ents)
    clusters_meta = {}
    for final_gid, idxs in final_clusters.items():
        # compute some meta: rep (first), size, sources, earliest, latest, avg_similarity_to_rep
        rep_idx_pos = None
        # find representative index position within group_reps: we need to map final_gid (which equals new_cluster_id)
        rep_positions = new_clusters.get(final_gid, [])
        # choose first rep position as canonical
        rep_pos = rep_positions[0] if rep_positions else 0
        rep_article_index = group_rep_idx[rep_pos] if rep_pos < len(group_rep_idx) else idxs[0]
        rep_text_vec = embs_arr[rep_pos] if rep_pos < len(embs_arr) else None

        # compute avg similarity of each article to rep (using text-based sim)
        sims = []
        for i in idxs:
            # compute sim between article i and rep article (use text/title combination)
            v_i = representative_text(articles[i], title_weight, text_weight)
            # for speed we compute text-based similarity via embeddings if available: encode single pair
            # fallback to simple jaccard-like measure
            try:
                # encode via same backend: we can reuse emb_backend.encode on pair
                _, pair_embs = emb_backend.encode([v_i, group_reps[rep_pos]])
                vec_i = np.asarray(pair_embs[0], dtype=float)
                vec_rep = np.asarray(pair_embs[1], dtype=float)
                s = cosine_sim_vectors(vec_i, vec_rep)
                s = max(0.0, min(1.0, (s+1.0)/2.0))
            except Exception:
                # fallback: token overlap
                s = token_jaccard_sim(v_i, group_reps[rep_pos])
            sims.append(s)

        avg_sim = float(sum(sims)/len(sims)) if sims else 1.0
        sources = list({articles[i].get('source','') for i in idxs})
        dates = [articles[i].get('published') for i in idxs if articles[i].get('published') is not None]
        earliest = min(dates) if dates else None
        latest = max(dates) if dates else None
        # aggregate entities:
        ent_counter = {}  # key (name,type,ticker) -> count
        for i in idxs:
            for ent in articles[i].get('entities', []):
                key = (ent.get('name'), ent.get('type'), ent.get('ticker'))
                ent_counter[key] = ent_counter.get(key, 0) + 1

        # build list sorted by count desc
        ent_list = []
        for (name, etype, ticker), cnt in sorted(ent_counter.items(), key=lambda kv: kv[1], reverse=True):
            ent_list.append({"name": name, "type": etype, "ticker": ticker, "count": cnt})

        clusters_meta[final_gid] = {
            "size": len(idxs),
            "rep_article_index": rep_article_index,
            "sources": sources,
            "earliest": earliest,
            "latest": latest,
            "avg_similarity": round(avg_sim, 4),
            "backend": backend_name,
            "entities": ent_list
        }

        for i in idxs:
            a = dict(articles[i])  # copy
            a['dedup_group_id'] = final_gid
            # ensure we only include lightweight entity representation per article
            a['entities'] = [{"name": e.get('name'), "type": e.get('type'), "ticker": e.get('ticker')} for e in a.get('entities', [])]
            annotated_articles.append(a)

    # sort annotated_articles by published desc
    annotated_articles.sort(key=lambda x: x.get('published') or datetime.min, reverse=True)

    return annotated_articles, clusters_meta

# --------------------- small helpers used above ---------------------
def representative_text(article: Dict[str, Any], title_weight: float = 0.4, text_weight: float = 0.6) -> str:
    """
    Compose a single string combining title and a short snippet of text for embedding / comparison.
    Weights determine how much title vs body matters; implemented by repetition heuristic.
    """
    title = article.get('title','') or ''
    text = article.get('text','') or ''
    # Use repetition trick: repeat title proportional to title_weight to bias embedding
    # Calculate repeats (small integers)
    t_rep = max(1, int(round(title_weight * 5)))
    txt_rep = max(1, int(round(text_weight * 5)))
    # Use short text snippet to avoid huge inputs
    text_snip = (text[:800] + '...') if len(text) > 800 else text
    composed = (" " .join([title]*t_rep)) + " " + (" ".join([text_snip]*txt_rep))
    return composed.strip()

def token_jaccard_sim(a: str, b: str) -> float:
    ta = set(re.findall(r"[A-Za-zА-Яа-я0-9]+", (a or "").lower()))
    tb = set(re.findall(r"[A-Za-zА-Яа-я0-9]+", (b or "").lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = ta.intersection(tb)
    uni = ta.union(tb)
    return float(len(inter))/len(uni)

# --------------------- usage example ---------------------
if __name__ == "__main__":
    # Quick demo
    demo_articles = [
        {"title":"Apple misses guidance, shares plunge", "text":"Apple reported lower than expected revenues for Q3...", "url":"https://news.example.com/a1", "published": datetime.utcnow(), "source":"reuters"},
        {"title":"Apple misses guidance; CFO resigns", "text":"Shock in Cupertino as Apple announces revenue miss...", "url":"https://news.example.com/a2", "published": datetime.utcnow(), "source":"bloomberg"},
        {"title":"Small biotech signs research partnership", "text":"A university and small biotech announced a multi-year grant...", "url":"https://local.example/bio", "published": datetime.utcnow(), "source":"localnews"},
    ]

    annotated, clusters = dedupe_articles(demo_articles, similarity_threshold=0.7)
    print("Clusters meta:", clusters)
    for a in annotated:
        print(a['dedup_group_id'], a['title'][:80], a['source'])
