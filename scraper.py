from __future__ import annotations

from dataclasses import dataclass
import json
import re
import subprocess
import sys
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
    scheme = parsed.scheme or "https"
    host = parsed.netloc or "www.myinstants.com"
    path = parsed.path or "/"
    query = parsed.query
    fragment = parsed.fragment

    host_variants = [host]
    if host.startswith("www."):
        host_variants.append(host[4:])
    else:
        host_variants.append(f"www.{host}")

    path_variants = [path]
    trimmed = path.rstrip("/")
    if trimmed and trimmed != path:
        path_variants.append(trimmed)
    normalized = (trimmed or "/").lower()
    index_like = normalized in {"/", "/en", "/en/index", "/en/index/in"} or "/en/index/" in normalized
    if index_like:
        if "/en/index/in/" in path:
            path_variants.append(path.replace("/en/index/in/", "/en/index/"))
        elif "/en/index/" in path:
            path_variants.append("/en/")
        path_variants.append("/en/index/in/")
        path_variants.append("/en/index/")
        path_variants.append("/en/")

    candidates: list[str] = []
    for host_item in host_variants:
        for path_item in path_variants:
            normalized_path = path_item or "/"
            if not normalized_path.startswith("/"):
                normalized_path = f"/{normalized_path}"
            candidates.append(
                urlunparse((scheme, host_item, normalized_path, parsed.params, query, fragment))
            )

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


# Cache the resolved PowerShell path so we don't hit the PATH lookup on
# every call.  Populated lazily by _powershell_exe().
_POWERSHELL_EXE: str | None = None


def _powershell_exe() -> str:
    """Return the full path to powershell.exe (cached)."""
    global _POWERSHELL_EXE
    if _POWERSHELL_EXE is None:
        import shutil
        path = shutil.which("powershell")
        _POWERSHELL_EXE = path or "powershell"
    return _POWERSHELL_EXE


def _fetch_via_dotnet(url: str, timeout: float) -> tuple[str, str]:
    """Fetch a URL using PowerShell's Invoke-WebRequest (.NET HTTP stack).

    The .NET TLS implementation is accepted by Cloudflare-protected sites
    that reject Python's ssl module during the TLS handshake.
    Returns (html_text, final_url).
    """
    timeout_sec = max(int(timeout), 8)
    ps_script = (
        f'$ProgressPreference = "SilentlyContinue"; '
        f'$r = Invoke-WebRequest -Uri "{url}" -UseBasicParsing '
        f'-TimeoutSec {timeout_sec} '
        f'-UserAgent "Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        f'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"; '
        f'@{{ status = $r.StatusCode; url = $r.BaseResponse.ResponseUri.AbsoluteUri; '
        f'content = $r.Content }} | ConvertTo-Json -Compress'
    )
    result = subprocess.run(
        [_powershell_exe(), "-NoProfile", "-NonInteractive", "-Command", ps_script],
        capture_output=True,
        text=True,
        timeout=timeout_sec + 10,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise RuntimeError(f"PowerShell fetch failed (rc={result.returncode}): {result.stderr[:300]}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError("PowerShell returned non-JSON output")
    status = int(data.get("status", 0))
    if status < 200 or status >= 400:
        raise RuntimeError(f"HTTP {status}")
    content = str(data.get("content", ""))
    resolved_url = str(data.get("url", "") or url)
    return content, resolved_url


def download_via_dotnet(url: str, dst_path: str, timeout: float = 30.0) -> None:
    """Download a binary file using .NET WebClient (avoids TLS rejection)."""
    timeout_ms = int(max(timeout, 10) * 1000)
    # Use [System.Net.WebClient] for efficient binary downloads.
    ps_script = (
        f'$ProgressPreference = "SilentlyContinue"; '
        f'$wc = New-Object System.Net.WebClient; '
        f'$wc.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        f'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"); '
        f'$wc.DownloadFile("{url}", "{dst_path}")'
    )
    result = subprocess.run(
        [_powershell_exe(), "-NoProfile", "-NonInteractive", "-Command", ps_script],
        capture_output=True,
        text=True,
        timeout=int(timeout) + 15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr[:300]}")


def _fetch_page_html(url: str, timeout: float) -> tuple[str, str]:
    # Try .NET HTTP stack first (Windows) – its TLS implementation is accepted
    # by Cloudflare whereas Python's ssl/urllib3 gets connection-reset.
    # Only try the primary URL to avoid spawning many PowerShell processes.
    if sys.platform == "win32":
        try:
            text, resolved = _fetch_via_dotnet(url, timeout)
            if text and not (_looks_blocked_html(text) and 'class="instant"' not in text):
                return text, resolved
        except Exception:
            pass  # fall through to requests-based attempts

    errors: list[str] = []
    timeout_pair: tuple[float, float] = (8.0, max(float(timeout), 8.0))
    attempt = 0

    max_rounds = 3
    for _round_idx in range(max_rounds):
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
