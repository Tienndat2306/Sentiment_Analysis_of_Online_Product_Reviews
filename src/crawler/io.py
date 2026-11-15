from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

import json
from src.crawler.model import LazadaReview, ProductSeed, Review, TikiReview


def reviews_to_frame(reviews: Iterable[Review | Any]):
    return pd.DataFrame([asdict(review) for review in reviews])


def get_existing_review_stats(output: str | Path) -> tuple[set[str], dict[str, int]]:
    """
    Trích xuất danh sách review_id đã có và số lượng review theo từng product_id
    từ file dữ liệu (CSV, JSONL, JSON, Parquet) phục vụ Resume & Khử trùng lặp.
    Trả về:
        (existing_review_ids: set[str], product_counts: dict[str, int])
    """
    output_path = Path(output)
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set(), {}

    fmt = output_path.suffix.lstrip(".").lower()
    try:
        if fmt == "csv":
            df = pd.read_csv(
                output_path,
                dtype={"review_id": str, "product_id": str},
                usecols=lambda c: c in {"review_id", "product_id"},
            )
        elif fmt == "jsonl":
            df = pd.read_json(
                output_path,
                lines=True,
                dtype={"review_id": str, "product_id": str},
            )
        elif fmt == "json":
            df = pd.read_json(output_path, dtype={"review_id": str, "product_id": str})
        elif fmt == "parquet":
            df = pd.read_parquet(output_path, columns=["review_id", "product_id"])
        else:
            return set(), {}

        existing_ids = set()
        if "review_id" in df.columns:
            cleaned_rids = df["review_id"].dropna().astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
            existing_ids = set(cleaned_rids)
            existing_ids.discard("")

        product_counts = {}
        if "product_id" in df.columns:
            cleaned_pids = df["product_id"].dropna().astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
            product_counts = cleaned_pids.value_counts().to_dict()

        return existing_ids, product_counts
    except Exception:
        return set(), {}


def save_reviews(
    reviews: Iterable[Review | Any],
    output: str | Path,
    file_format: str | None = None,
    append: bool = True,
    dedup_key: str | None = "review_id",
) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = reviews_to_frame(reviews)
    if df.empty:
        return output_path

    fmt = (file_format or output_path.suffix.lstrip(".") or "csv").lower()
    has_existing_data = output_path.exists() and output_path.stat().st_size > 0

    # 1. Khử trùng lặp nội bộ trong chính batch đang lưu
    if dedup_key and dedup_key in df.columns:
        df = df.drop_duplicates(subset=[dedup_key])
        if df.empty:
            return output_path

    # 2. Khử trùng lặp đối chiếu với dữ liệu đã có trong file (khi append)
    if append and has_existing_data and dedup_key and dedup_key in df.columns:
        existing_ids, _ = get_existing_review_stats(output_path)
        if existing_ids:
            col_str = df[dedup_key].dropna().astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
            df = df[~col_str.isin(existing_ids)]
            if df.empty:
                return output_path

    if fmt == "csv":
        write_mode = "a" if append else "w"
        write_header = not (append and has_existing_data)
        encoding = "utf-8" if append and has_existing_data else "utf-8-sig"
        df.to_csv(output_path, mode=write_mode, header=write_header, index=False, encoding=encoding)
    elif fmt == "jsonl":
        with output_path.open("a" if append else "w", encoding="utf-8") as file:
            df.to_json(file, orient="records", lines=True, force_ascii=False)
    elif fmt == "json":
        if append and has_existing_data:
            existing = pd.read_json(output_path)
            df = pd.concat([existing, df], ignore_index=True)
            if dedup_key and dedup_key in df.columns:
                df = df.drop_duplicates(subset=[dedup_key])
        df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    elif fmt == "parquet":
        if append and has_existing_data:
            existing = pd.read_parquet(output_path)
            df = pd.concat([existing, df], ignore_index=True)
            if dedup_key and dedup_key in df.columns:
                df = df.drop_duplicates(subset=[dedup_key])
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")

    return output_path


def save_product_seeds(
    seeds: Iterable[ProductSeed],
    output: str | Path,
    append: bool = True,
) -> tuple[Path, int]:
    """Save product seeds to JSONL file with automatic deduplication by (source, product_id)."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_keys = set()
    if append and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        item = json.loads(line_str)
                        existing_keys.add((item.get("source"), str(item.get("product_id"))))
                    except json.JSONDecodeError:
                        continue

    added_count = 0
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        for seed in seeds:
            key = (seed.source, str(seed.product_id))
            if not append or key not in existing_keys:
                f.write(json.dumps(asdict(seed), ensure_ascii=False) + "\n")
                existing_keys.add(key)
                added_count += 1

    return output_path, added_count


def load_product_seeds(input_path: str | Path) -> list[ProductSeed]:
    """Load product seeds from JSONL file."""
    path = Path(input_path)
    if not path.exists():
        return []

    seeds = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                data = json.loads(line_str)
                seeds.append(ProductSeed(**data))
    return seeds


def update_product_seeds_file(
    seeds_path: str | Path,
    updates: dict[tuple[str, str], dict[str, Any]],
) -> Path:
    """
    Cập nhật các trường (như status, crawled_at, ...) cho các sản phẩm trong file JSONL.
    updates có dạng: {(source, product_id): {"status": "completed", ...}}
    Ghi an toàn qua file tạm (.tmp) rồi thay thế nguyên tử (atomic replace).
    """
    path = Path(seeds_path)
    if not path.exists() or not updates:
        return path

    temp_path = path.with_suffix(".tmp")
    with path.open("r", encoding="utf-8") as f_in, temp_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                item = json.loads(line_str)
                src = str(item.get("source", "")).lower()
                pid = str(item.get("product_id", "")).strip()
                key = (src, pid)
                if key in updates:
                    item.update(updates[key])
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                f_out.write(line_str + "\n")

    temp_path.replace(path)
    return path


def mark_product_seed_completed(
    seeds_path: str | Path,
    product_id: str | int,
    source: str,
    status: str = "completed",
    crawled_at: str | None = None,
) -> None:
    """
    Đánh dấu một sản phẩm trong file seeds là đã hoàn thành (status='completed').
    """
    now_ts = crawled_at or datetime.now(timezone.utc).isoformat()
    key = (str(source).lower(), str(product_id).strip())
    update_product_seeds_file(
        seeds_path,
        {
            key: {
                "status": status,
                "crawled_at": now_ts,
            }
        },
    )
