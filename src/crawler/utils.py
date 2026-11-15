from datetime import datetime, timezone
import re
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse


def extract_product_id(value: str):
    value = value.strip()
    if value.isdigit():
        return int(value)

    parsed = urlparse(value)
    match = re.search(r"-p(\d+)\.html", parsed.path)
    if match:
        return int(match.group(1))

    query = parse_qs(parsed.query)
    for key in ("product_id", "pid", "spid"):
        if query.get(key):
            return int(query[key][0])

    raise ValueError(f"Cannot extract Tiki product id from: {value}")


def extract_lazada_product_id(value: str | int) -> str:
    val = str(value).strip()
    if val.isdigit():
        return val

    parsed = urlparse(val)
    query = parse_qs(parsed.query)
    if "item_id" in query and query["item_id"]:
        return str(query["item_id"][0])

    m = re.search(r"[-_]i(\d+)", parsed.path)
    if m:
        return m.group(1)

    m_generic = re.search(r"(\d{9,})", val)
    if m_generic:
        return m_generic.group(1)

    raise ValueError(f"Cannot extract Lazada product id from: {value}")


def safe_int(value: Any):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_datetime(value: Any):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        timestamp = value / 1000 if value > 10_000_000_000 else value
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
    if isinstance(value, str):
        val = value.strip()
        # Try parsing Lazada dates like "May 09, 2026" or "09 May 2026"
        for fmt in ("%b %d, %Y", "%d %b %Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(val, fmt).replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                pass
        return val
    return str(value)


def source_to_arguments(source: str | int | Iterable[str | int]):
    values = as_list(source)
    if all(isinstance(value, int) for value in values):
        return [int(value) for value in values], None, None
    if all(isinstance(value, str) and value.strip().isdigit() for value in values):
        return [int(str(value).strip()) for value in values], None, None
    if all(isinstance(value, str) and looks_like_url(value) for value in values):
        return None, [str(value) for value in values], None
    if len(values) == 1 and isinstance(values[0], str):
        return None, None, values[0]
    raise ValueError("source must be product id(s), URL(s), or a single keyword.")


def as_list(value: Any):
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def looks_like_url(value: str):
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
