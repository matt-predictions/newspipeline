from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


def canonicalize_url(url: str) -> str:
    p = urlparse(url.strip())
    netloc = (p.hostname or "").lower()
    if p.port and p.port not in (80, 443):
        netloc = f"{netloc}:{p.port}"
    q = parse_qsl(p.query, keep_blank_values=True)
    drop = {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
    }
    q = [(k, v) for k, v in q if k.lower() not in drop]
    q.sort(key=lambda x: x[0].lower())
    new_query = urlencode(q, doseq=True)
    return urlunparse(
        (p.scheme.lower() or "https", netloc, p.path or "/", "", new_query, "")
    )


def url_hash(url: str) -> str:
    return hashlib.sha256(canonicalize_url(url).encode()).hexdigest()[:16]
