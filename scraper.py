from __future__ import annotations

from dataclasses import dataclass
import re
import time
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

MYINSTANTS_INDEX_URL = "https://www.myinstants.com/en/index/in/"
_PLAY_RE = re.compile(r"play\(\s*'(?P<url>[^']+)'\s*,")
_BLOCK_MARKERS = (
    "cf-chl",
    "attention required",
    "captcha",
    "verify you are human",
)
_HEADER_PROFILES: tuple[dict[str, str], ...] = (
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "close",
    },
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) "
            "Gecko/20100101 Firefox/124.0"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
        "Cache-Control": "max-age=0",
        "Pragma": "no-cache",
        "Connection": "close",
    },
)


def _build_http_session(extra_headers: dict[str, str] | None = None) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    base_headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "close",
    }
    if extra_headers:
        base_headers.update(extra_headers)
    session.headers.update(base_headers)
    return session


def _candidate_feed_urls(url: str) -> list[str]:
    parsed = urlparse(url)
    path = parsed.path or "/"
    candidates = [url]

    if path.endswith("//"):
        candidates.append(urlunparse(parsed._replace(path=path.rstrip("/"))))

    dedup: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        dedup.append(candidate)
    return dedup


def _extract_sounds_from_html(html: str, base_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    sounds: list[tuple[str, str]] = []
    for instant in soup.select("div.instant"):
        link = instant.select_one("a.instant-link")
        button = instant.select_one("button.small-button")
        if link is None or button is None:
            continue

        name = " ".join(link.get_text(strip=True).split())
        onclick = button.get("onclick", "")
        match = _PLAY_RE.search(onclick)
        if not match:
            continue

        audio_url = urljoin(base_url, match.group("url"))
        sounds.append((name, audio_url))
    return sounds


def _looks_blocked_html(html: str) -> bool:
    lowered = html.lower()
    return any(marker in lowered for marker in _BLOCK_MARKERS)


def _fetch_page_html(url: str, timeout: float) -> tuple[str, str]:
    errors: list[str] = []
    timeout_pair: tuple[float, float] = (8.0, max(float(timeout), 8.0))
    attempt = 0

    for candidate_url in _candidate_feed_urls(url):
        for profile in _HEADER_PROFILES:
            attempt += 1
            session = _build_http_session(extra_headers=profile)
            try:
                response = session.get(candidate_url, timeout=timeout_pair, allow_redirects=True)
                response.raise_for_status()
                text = response.text or ""
                if _looks_blocked_html(text) and "class=\"instant\"" not in text:
                    errors.append(f"attempt {attempt}: blocked challenge page at {candidate_url}")
                    continue
                return text, response.url
            except requests.RequestException as exc:
                errors.append(f"attempt {attempt}: {exc}")
                time.sleep(min(1.4, 0.22 * attempt))
            finally:
                session.close()

    if not errors:
        raise RuntimeError("Unable to load myinstants feed: no response")

    detail = errors[-1]
    if len(detail) > 320:
        detail = detail[:320] + "..."
    raise RuntimeError(f"Unable to load myinstants feed: {detail}")


def fetch_myinstants_sounds(page_url: str = MYINSTANTS_INDEX_URL, timeout: float = 15.0) -> list[tuple[str, str]]:
    return fetch_myinstants_sounds_page(page_url=page_url, page=1, timeout=timeout)


def fetch_myinstants_sounds_page(
    page_url: str = MYINSTANTS_INDEX_URL, page: int = 1, timeout: float = 15.0
) -> list[tuple[str, str]]:
    if page < 1:
        page = 1
    page_url = _with_page_param(page_url, page)

    html, resolved_url = _fetch_page_html(page_url, timeout=timeout)
    return _extract_sounds_from_html(html, resolved_url)


@dataclass(frozen=True)
class RemoteSound:
    name: str
    url: str


def _with_page_param(url: str, page: int) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["page"] = str(page)
    return urlunparse(parsed._replace(query=urlencode(query)))
