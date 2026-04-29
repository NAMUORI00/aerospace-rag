from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable


TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(str(text or "").lower()):
        if not raw:
            continue
        tokens.append(raw)
        if re.search(r"[가-힣]", raw) and len(raw) >= 3:
            for n in (2, 3):
                tokens.extend(raw[i : i + n] for i in range(0, len(raw) - n + 1))
    return tokens


def hash_embedding(text: str, *, dim: int = 384) -> list[float]:
    vec = [0.0] * dim
    counts = Counter(tokenize(text))
    for token, count in counts.items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign * (1.0 + math.log(float(count)))
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def excerpt(text: str, *, max_chars: int = 360) -> str:
    cleaned = normalize_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def unique_ordered(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = str(value).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out
