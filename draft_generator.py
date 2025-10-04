# draft_generator.py
import os
import json
import re
import time
import logging
from typing import Dict, Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_JSON_RE = re.compile(r"(\{[\s\S]*\})", re.MULTILINE)

def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        m = _JSON_RE.search(text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
    return None

def make_client_from_env(api_key_env: str = "OPENROUTER_API_KEY", base_url: str = "https://openrouter.ai/api/v1"):
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env} not set.")
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client

def generate_draft_for_event(event: Dict[str, Any],
                             client: Optional[OpenAI] = None,
                             model: str = "openai/gpt-5",
                             temperature: float = 0.0,
                             max_tokens: int = 512,
                             retries: int = 2) -> Dict[str, Any]:
    """
    Returns: {"title": str or None, "text": str or None, "raw": str}
    Format requirement: model MUST return strict JSON with keys "title" and "text".
    - title: короткий заголовок (<=120 chars)
    - text: единый текстовый блок — лид (1-2 предложения), затем 3 буллета (каждый на новой строке, с "- "),
            затем цитата/сноска (если есть). Все дополнительные факты — только из event['sources']/['timeline']/['entities'].
    """
    if client is None:
        client = make_client_from_env()

    payload = {
        "headline": event.get("headline"),
        "entities": event.get("entities", []),
        "sources": event.get("sources", []),
        "timeline": event.get("timeline", []),
        "cluster_id": event.get("dedup_group")
    }

    system_msg = (
        "Вы — редактор, который пишет короткие заметки по финансовым событиям. "
        "ВАЖНО: используйте только информацию, предоставлённую в поле 'headline', 'entities', 'sources', 'timeline'. "
        "Ни при каких условиях не придумывайте факты. "
        "ОТВЕТ: вернуть строго ВАЛИДНЫЙ JSON и ничего кроме JSON. Формат:\n"
        "{\n"
        '  "title": "<краткий заголовок (строка)>",\n'
        '  "text": "<лид (1-2 предложения)\\n\\n- буллет 1\\n- буллет 2\\n- буллет 3\\n\\n\"Цитата / сноска (источник)\" (или пустая строка)>"\n'
        "}\n"
        "Требования по тексту:\n"
        "- Заголовок должен быть основан на headline и не содержать вымышленных данных.\n"
        "- В поле text сначала 1–2 предложения lead, объясняющие почему новость важна сейчас, затем пустая строка, затем ровно 3 буллета (каждый с префиксом '- '), затем пустая строка и одну строку цитаты/сноски (в кавычках) или пустую строку.\n"
        "- Если нет данных для цитаты — верните пустую строку для цитаты.\n"
        "- В буллетах и цитате можно в квадратных скобках указывать ссылку из sources, например [https://reuters.com/...].\n"
    )

    user_msg = "Входные данные (JSON):\n" + json.dumps(payload, ensure_ascii=False, indent=2)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    attempt = 0
    last_err = None
    while attempt <= retries:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            # extract model text (robust paths)
            model_text = None
            try:
                model_text = resp.choices[0].message["content"]
            except Exception:
                try:
                    model_text = resp.choices[0].message.content
                except Exception:
                    try:
                        model_text = resp.choices[0].get("message", {}).get("content")
                    except Exception:
                        model_text = str(resp)

            if not model_text:
                model_text = str(resp)

            parsed = _extract_json_from_text(model_text)
            if parsed is None:
                logger.warning("Не удалось распарсить JSON от модели. Возвращаю raw в поле 'raw'.")
                return {"title": None, "text": None, "raw": model_text}

            title = parsed.get("title")
            text = parsed.get("text")

            # basic post-checks: ensure strings
            if title is not None:
                title = str(title).strip()
            if text is not None:
                text = str(text).strip()

            return {"title": title, "text": text, "raw": model_text}

        except Exception as e:
            last_err = e
            attempt += 1
            logger.exception("Draft generation failed attempt %s: %s", attempt, e)
            time.sleep(1 + attempt * 2)

    raise RuntimeError(f"Draft generation failed after {retries+1} attempts. Last error: {last_err}")

