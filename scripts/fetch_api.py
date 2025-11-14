import asyncio
import json
import sys
import logging
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

import httpx

# --- Configuration ---
SEARCH_ENDPOINT = "https://data.europa.eu/api/hub/search/search"
OUTPUT_FILE = "all_datasets_api_async.jsonl"
START_PAGE = 0
LANGUAGE = "en"
QUERY: str = ""
TIMEOUT_SECONDS = 60
LIMIT = 1000  # docs: up to 1000
USE_SCROLL = True
MAX_CONCURRENCY = 20  # used for page-based parallel fetching
INCLUDES = (
    "id,title.en,description.en,languages,modified,keywords,issued,"
    "catalog.id,catalog.title,catalog.country.id,distributions.id,"
    "distributions.format.label,distributions.format.id,distributions.license,"
    "categories.label,publisher"
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("download_api_async.log"),
    ],
)


def _pick_text(value: Any, lang: str = LANGUAGE) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        if lang in value and isinstance(value[lang], str) and value[lang].strip():
            return value[lang].strip()
        parts = [str(v).strip() for v in value.values() if isinstance(v, (str, int, float))]
        return "; ".join([p for p in parts if p])
    if isinstance(value, (list, tuple)):
        parts: List[str] = []
        for item in value:
            if isinstance(item, (str, int, float)):
                s = str(item).strip()
                if s:
                    parts.append(s)
            elif isinstance(item, dict):
                s = _pick_text(item.get("label") or item.get("value") or item.get("title") or item.get("name") or item, lang)
                if s:
                    parts.append(s)
        return "; ".join([p for p in parts if p])
    if isinstance(value, (str, int, float)):
        s = str(value).strip()
        return s
    return ""


def _first_non_empty(*values: Any) -> str:
    for v in values:
        s = _pick_text(v)
        if s:
            return s
    return ""


def _normalize_item(item: Dict[str, Any]) -> Dict[str, Optional[str]]:
    dataset_uri = (
        item.get("id") or item.get("dataset") or item.get("uri") or item.get("@id") or item.get("identifier")
    )
    dataset = _pick_text(dataset_uri)

    title = _first_non_empty(
        item.get("title.en"),
        (item.get("title") or {}).get("en") if isinstance(item.get("title"), dict) else None,
        item.get("title"),
        item.get("titles"),
        item.get("label"),
        item.get("name"),
        item.get("dct:title"),
    )

    description = _first_non_empty(
        item.get("description.en"),
        (item.get("description") or {}).get("en") if isinstance(item.get("description"), dict) else None,
        item.get("description"),
        item.get("abstract"),
        item.get("dct:description"),
        item.get("notes"),
    )

    keywords_raw = item.get("keywords") or item.get("tags") or item.get("theme") or item.get("themes")
    keywords = _pick_text(keywords_raw)

    if not dataset:
        landing = item.get("landingPage") or item.get("landingPages") or item.get("url")
        dataset = _pick_text(landing)

    return {
        "dataset": dataset or None,
        "title": title or None,
        "description": description or None,
        "keywords": keywords or None,
    }


def _extract_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if payload is None:
        return []
    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("results"), list):
        return result["results"]
    if isinstance(payload.get("items"), list):
        return payload["items"]
    if isinstance(payload.get("results"), list):
        return payload["results"]
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    if isinstance(payload.get("hits"), list):
        return payload["hits"]
    if isinstance(payload, list):
        return payload
    return []


def _extract_total(payload: Dict[str, Any]) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    result = payload.get("result")
    if isinstance(result, dict):
        for k in ("total", "count", "resultCount"):
            v = result.get(k)
            if isinstance(v, int):
                return v
    for k in ("total", "count", "resultCount"):
        v = payload.get(k)
        if isinstance(v, int):
            return v
    return None


def _extract_scroll_id(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    val = payload.get("scrollId")
    if isinstance(val, str) and val:
        return val
    result = payload.get("result")
    if isinstance(result, dict):
        val = result.get("scrollId")
        if isinstance(val, str) and val:
            return val
    scroll_obj = payload.get("scroll")
    if isinstance(scroll_obj, dict):
        val = scroll_obj.get("id")
        if isinstance(val, str) and val:
            return val
    return None


async def fetch_page(client: httpx.AsyncClient, page: int, *, limit: int = LIMIT, scroll: bool = False) -> Optional[Dict[str, Any]]:
    # When starting a scroll session, omit the 'page' parameter per API docs.
    params: Dict[str, Any] = {
        "q": QUERY if QUERY is not None else "",
        "filter": "dataset",
        "includes": INCLUDES,
        "limit": limit,
        "scroll": str(scroll).lower(),
    }
    if not scroll:
        params["page"] = page
    try:
        r = await client.get(SEARCH_ENDPOINT, params=params)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error on /search page={page}: {e.response.status_code} {e.response.text[:200]}")
    except httpx.RequestError as e:
        logging.error(f"Request error on /search page={page}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error on /search page={page}: {e}")
    return None


async def fetch_scroll(client: httpx.AsyncClient, scroll_id: str) -> Optional[Dict[str, Any]]:
    url = SEARCH_ENDPOINT.rsplit("/", 1)[0] + "/scroll"
    params = {"scrollId": scroll_id}
    try:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error on /scroll: {e.response.status_code} {e.response.text[:200]}")
    except httpx.RequestError as e:
        logging.error(f"Request error on /scroll: {e}")
    except Exception as e:
        logging.error(f"Unexpected error on /scroll: {e}")
    return None


async def main_async() -> None:
    page = START_PAGE
    total_written = 0
    limits = httpx.Limits(max_keepalive_connections=MAX_CONCURRENCY, max_connections=MAX_CONCURRENCY)
    timeout = httpx.Timeout(TIMEOUT_SECONDS)

    async with httpx.AsyncClient(http2=True, timeout=timeout, limits=limits, headers={"Accept": "application/json"}) as client:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
            # First page
            first_payload = await fetch_page(client, page, limit=LIMIT, scroll=USE_SCROLL)
            if first_payload is None:
                logging.error("Failed to fetch the first page; aborting.")
                return
            first_items = _extract_items(first_payload)
            if not first_items:
                logging.info("First page returned no items; nothing to do.")
                return

            normalized_first: List[Dict[str, Optional[str]]] = []
            for it in first_items:
                if not isinstance(it, dict):
                    continue
                norm = _normalize_item(it)
                if not any(norm.values()):
                    continue
                normalized_first.append(norm)
            for row in normalized_first:
                outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_written += 1
            logging.info(f"   Wrote {len(normalized_first)} rows from page {page}. Total so far: {total_written:,}")

            total = _extract_total(first_payload)
            effective_page_size = len(first_items)

            if USE_SCROLL:
                scroll_failed = False
                scroll_id = _extract_scroll_id(first_payload)
                if not scroll_id:
                    logging.warning("No scrollId returned from first response; cannot continue scrolling. Will attempt page-based fallback.")
                    scroll_failed = True
                while scroll_id:
                    payload = await fetch_scroll(client, scroll_id)
                    if payload is None:
                        logging.warning("Scroll fetch failed; stopping scroll loop.")
                        scroll_failed = True
                        break
                    items = _extract_items(payload)
                    if not items:
                        logging.info("Scroll returned no items; finished.")
                        break
                    rows: List[Dict[str, Optional[str]]] = []
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        norm = _normalize_item(it)
                        if not any(norm.values()):
                            continue
                        rows.append(norm)
                    for row in rows:
                        outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
                        total_written += 1
                    logging.info(f"   Wrote {len(rows)} rows from scroll. Total so far: {total_written:,}")
                    next_scroll_id = _extract_scroll_id(payload)
                    if not next_scroll_id:
                        break
                    scroll_id = next_scroll_id

                # Fallback if scroll failed mid-way: resume with page-based fetching for remaining pages
                if scroll_failed:
                    if total and effective_page_size > 0:
                        total_pages = ceil(total / effective_page_size)
                        remaining_pages = list(range(page + 1, total_pages + 1))
                        logging.info(
                            f"Scroll failed; falling back to page-based fetching. total={total} (~{total_pages} pages). Remaining {len(remaining_pages)} pages (max concurrency={MAX_CONCURRENCY})."
                        )

                        sem = asyncio.Semaphore(MAX_CONCURRENCY)

                        async def fetch_and_normalize(p: int) -> Tuple[int, List[Dict[str, Optional[str]]]]:
                            async with sem:
                                pl = await fetch_page(client, p, limit=LIMIT, scroll=False)
                            if pl is None:
                                return p, []
                            its = _extract_items(pl)
                            rows: List[Dict[str, Optional[str]]] = []
                            for it in its:
                                if not isinstance(it, dict):
                                    continue
                                norm = _normalize_item(it)
                                if not any(norm.values()):
                                    continue
                                rows.append(norm)
                            return p, rows

                        tasks = [asyncio.create_task(fetch_and_normalize(p)) for p in remaining_pages]
                        for coro in asyncio.as_completed(tasks):
                            pnum, rows = await coro
                            for row in rows:
                                outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
                                total_written += 1
                            logging.info(f"   Wrote {len(rows)} rows from page {pnum}. Total so far: {total_written:,}")
                    else:
                        logging.info("Total unknown; page size unknown. Falling back to sequential page-by-page fetching.")
                        nxt = page + 1
                        while True:
                            pl = await fetch_page(client, nxt, limit=LIMIT, scroll=False)
                            if pl is None:
                                await asyncio.sleep(5)
                                continue
                            items = _extract_items(pl)
                            if not items:
                                logging.info("No more results or empty response. Download complete.")
                                break
                            rows: List[Dict[str, Optional[str]]] = []
                            for it in items:
                                if not isinstance(it, dict):
                                    continue
                                norm = _normalize_item(it)
                                if not any(norm.values()):
                                    continue
                                rows.append(norm)
                            for row in rows:
                                outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
                                total_written += 1
                            logging.info(f"   Wrote {len(rows)} rows from page {nxt}. Total so far: {total_written:,}")
                            nxt += 1

            elif total and effective_page_size > 0:
                total_pages = ceil(total / effective_page_size)
                remaining_pages = list(range(page + 1, total_pages + 1))
                logging.info(f"Detected total={total} items (~{total_pages} pages at ~{effective_page_size}/page). Fetching remaining {len(remaining_pages)} pages concurrently (max={MAX_CONCURRENCY}).")

                sem = asyncio.Semaphore(MAX_CONCURRENCY)

                async def fetch_and_normalize(p: int) -> Tuple[int, List[Dict[str, Optional[str]]]]:
                    async with sem:
                        pl = await fetch_page(client, p, limit=LIMIT, scroll=False)
                    if pl is None:
                        return p, []
                    its = _extract_items(pl)
                    rows: List[Dict[str, Optional[str]]] = []
                    for it in its:
                        if not isinstance(it, dict):
                            continue
                        norm = _normalize_item(it)
                        if not any(norm.values()):
                            continue
                        rows.append(norm)
                    return p, rows

                tasks = [asyncio.create_task(fetch_and_normalize(p)) for p in remaining_pages]
                for coro in asyncio.as_completed(tasks):
                    pnum, rows = await coro
                    for row in rows:
                        outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
                        total_written += 1
                    logging.info(f"   Wrote {len(rows)} rows from page {pnum}. Total so far: {total_written:,}")

            else:
                logging.info("Total unknown; continuing sequentially page-by-page until empty page is returned.")
                page += 1
                while True:
                    pl = await fetch_page(client, page, limit=LIMIT, scroll=False)
                    if pl is None:
                        await asyncio.sleep(10)
                        continue
                    items = _extract_items(pl)
                    if not items:
                        logging.info("No more results or empty response. Download complete.")
                        break
                    rows: List[Dict[str, Optional[str]]] = []
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        norm = _normalize_item(it)
                        if not any(norm.values()):
                            continue
                        rows.append(norm)
                    for row in rows:
                        outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
                        total_written += 1
                    logging.info(f"   Wrote {len(rows)} rows from page {page}. Total so far: {total_written:,}")
                    page += 1

    logging.info(f"All data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main_async())
