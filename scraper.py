from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

MYINSTANTS_INDEX_URL = "https://www.myinstants.com/en/index/in/"
_PLAY_RE = re.compile(r"play\(\s*'(?P<url>[^']+)'\s*,")


@dataclass(frozen=True)
class RemoteSound:
    name: str
    url: str


def fetch_myinstants_sounds(page_url: str = MYINSTANTS_INDEX_URL, timeout: float = 15.0) -> list[tuple[str, str]]:
    return fetch_myinstants_sounds_page(page_url=page_url, page=1, timeout=timeout)


def fetch_myinstants_sounds_page(
    page_url: str = MYINSTANTS_INDEX_URL, page: int = 1, timeout: float = 15.0
) -> list[tuple[str, str]]:
    if page < 1:
        page = 1
    page_url = _with_page_param(page_url, page)

    response = requests.get(page_url, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
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

        audio_url = urljoin(response.url, match.group("url"))
        sounds.append((name, audio_url))

    return sounds


def _with_page_param(url: str, page: int) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["page"] = str(page)
    return urlunparse(parsed._replace(query=urlencode(query)))
