"""Read the main text of a news article from a link.

Safety: the server fetches URLs that users type in, so we block links to
private/internal addresses (an attack called SSRF) and limit size and time.
"""
import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (FakeNewsChecker/2.0)"}
MAX_BYTES = 2_000_000
MAX_CHARS = 6000


class UnsafeURL(ValueError):
    pass


def check_url(url: str) -> str:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise UnsafeURL("Only http/https links are allowed.")
    try:
        infos = socket.getaddrinfo(p.hostname, None)
    except socket.gaierror:
        raise UnsafeURL("Could not find that website.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise UnsafeURL("Links to private or local addresses are not allowed.")
    return url


def fetch_html(url: str, max_redirects: int = 3) -> str:
    for _ in range(max_redirects + 1):
        check_url(url)
        r = requests.get(url, headers=UA, timeout=10, allow_redirects=False, stream=True)
        if r.is_redirect:
            url = urljoin(url, r.headers.get("location", ""))
            continue
        r.raise_for_status()
        body = r.raw.read(MAX_BYTES, decode_content=True)
        return body.decode(r.encoding or "utf-8", errors="replace")
    raise UnsafeURL("Too many redirects.")


def extract_text(html: str) -> tuple:
    """Return (title, main text) using simple rules: keep real paragraphs."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    title = (soup.find("h1") or soup.title or soup.new_tag("x")).get_text(" ", strip=True)
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paras = [p for p in paras if len(p.split()) >= 8]
    return title, " ".join(paras)[:MAX_CHARS]


def fetch_article(url: str) -> dict:
    title, text = extract_text(fetch_html(url))
    if len(text.split()) < 20:
        raise ValueError("Could not find article text on that page. Paste the text instead.")
    return {"url": url, "title": title, "text": text}
