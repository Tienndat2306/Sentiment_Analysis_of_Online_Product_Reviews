from datetime import datetime, timezone
import argparse
import sys
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crawler.io import (
    get_existing_review_stats,
    mark_product_seed_completed,
    save_reviews,
    update_product_seeds_file,
)
from src.crawler.lazada_review_crawler import crawl_lazada_reviews_to_file
from src.crawler.tiki_review_crawler import DEFAULT_SORT, TikiReviewCrawler
from src.crawler.utils import as_list, extract_product_id, safe_int, source_to_arguments


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Crawl product reviews from Tiki or Lazada.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--product-id", action="append", help="Product ID (Tiki or Lazada). Can be repeated.")
    source.add_argument("--url", action="append", help="Product URL (Tiki or Lazada). Can be repeated.")
    source.add_argument("--keyword", help="Search keyword (Tiki), then crawl reviews from found products.")
    source.add_argument("--from-seeds", help="Path to products.jsonl to crawl reviews for discovered products.")

    parser.add_argument(
        "--platform",
        choices=["auto", "tiki", "lazada"],
        default="auto",
        help="E-commerce platform to crawl (default: auto detect from URL or seeds).",
    )
    parser.add_argument("--search-pages", type=int, default=1, help="Number of search pages for --keyword.")
    parser.add_argument("--max-products", type=int, help="Max products to crawl after keyword search.")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="Max review pages per product (set 0 or negative to crawl ALL pages).",
    )
    parser.add_argument("--limit", type=int, default=20, help="Reviews per API page.")
    parser.add_argument("--sleep", type=float, default=0.4, help="Delay between requests in seconds.")
    parser.add_argument("--sort", default=DEFAULT_SORT, help="Tiki review sort expression.")
    parser.add_argument("--output", default="data/bronze/reviews.csv", help="Output file path.")
    parser.add_argument("--format", choices=["csv", "json", "jsonl", "parquet"], help="Output format.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file instead of appending.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip products that are already fully crawled in output file (default: True).",
    )
    return parser


def crawl_reviews_to_file(
    *,
    platform: str = "auto",
    product_ids: Iterable[str | int] | None = None,
    urls: Iterable[str] | None = None,
    keyword: str | None = None,
    from_seeds: str | Path | None = None,
    search_pages: int = 1,
    max_products: int | None = None,
    max_pages: int | None = 3,
    limit: int = 20,
    sleep: float = 0.4,
    sort: str = DEFAULT_SORT,
    output: str | Path = "data/bronze/reviews.csv",
    file_format: str | None = None,
    append: bool = True,
    resume: bool = True,
):
    # Detect platform
    is_lazada = platform.lower() == "lazada"
    if platform.lower() == "auto":
        if urls and any("lazada.vn" in str(u).lower() for u in urls):
            is_lazada = True
        elif from_seeds:
            from src.crawler.io import load_product_seeds
            seeds = load_product_seeds(from_seeds)
            if seeds and all(s.source.lower() == "lazada" for s in seeds):
                is_lazada = True

    if is_lazada:
        out_path = crawl_lazada_reviews_to_file(
            product_ids=product_ids,
            urls=urls,
            from_seeds=from_seeds,
            max_products=max_products,
            max_pages=max_pages,
            sleep=max(sleep, 0.8),
            output=output,
            file_format=file_format,
            append=append,
            resume=resume,
        )
        return out_path, 0, 0

    crawler = TikiReviewCrawler(sleep=sleep)
    provided_sources = sum(source is not None for source in (product_ids, urls, keyword, from_seeds))
    if provided_sources != 1:
        raise ValueError("Provide exactly one source: product_ids, urls, keyword, or from_seeds.")

    product_names = {}
    product_categories = {}
    product_target_counts = {}

    if product_ids is not None:
        resolved_product_ids = [int(p) for p in product_ids]
    elif urls is not None:
        resolved_product_ids = [extract_product_id(url) for url in urls]
    elif from_seeds is not None:
        from src.crawler.io import load_product_seeds
        seeds = load_product_seeds(from_seeds)
        resolved_product_ids = [int(s.product_id) for s in seeds if s.source == "tiki"]
        product_categories = {
            int(s.product_id): (s.category_id, s.category_name)
            for s in seeds
            if s.source == "tiki"
        }
        product_names = {
            int(s.product_id): s.product_name
            for s in seeds
            if s.source == "tiki" and s.product_name
        }
        product_target_counts = {
            int(s.product_id): safe_int(s.review_count) or 0
            for s in seeds
            if s.source == "tiki"
        }
    else:
        resolved_product_ids = crawler.search_product_ids(keyword or "", pages=search_pages)

    # Nạp danh mục và review_count từ seeds làm fallback nếu crawl theo id/url/keyword
    default_seeds_file = Path("data/seeds/products.jsonl")
    if default_seeds_file.exists():
        from src.crawler.io import load_product_seeds
        for s in load_product_seeds(default_seeds_file):
            if s.source == "tiki":
                try:
                    s_pid = int(s.product_id)
                    if s_pid not in product_categories:
                        product_categories[s_pid] = (s.category_id, s.category_name)
                    if s_pid not in product_names and s.product_name:
                        product_names[s_pid] = s.product_name
                    if s_pid not in product_target_counts:
                        product_target_counts[s_pid] = safe_int(s.review_count) or 0
                except (ValueError, TypeError):
                    pass

    if max_pages is not None and max_pages <= 0:
        max_pages = None

    unique_product_ids = list(dict.fromkeys(resolved_product_ids))
    if max_products is not None:
        unique_product_ids = unique_product_ids[:max_products]

    if not unique_product_ids:
        raise ValueError("No products found.")

    output_path = Path(output)
    existing_review_ids = set()
    product_review_counts = {}

    seeds_file = Path(from_seeds) if from_seeds else default_seeds_file

    if not append and output_path.exists():
        output_path.unlink()

    if append and output_path.exists() and output_path.stat().st_size > 0:
        existing_review_ids, product_review_counts = get_existing_review_stats(output_path)
        if existing_review_ids:
            print(
                f"[Resume] Đã nạp {len(existing_review_ids):,} reviews của "
                f"{len(product_review_counts)} sản phẩm từ {output_path}."
            )
            # Tự động đồng bộ status='completed' cho các sản phẩm đã cào đủ trước đó
            if seeds_file.exists():
                already_completed_updates = {}
                for pid_str, cnt in product_review_counts.items():
                    try:
                        pid_int = int(pid_str)
                    except ValueError:
                        continue
                    tgt = product_target_counts.get(pid_int)
                    if tgt is not None:
                        eff_tgt = min(tgt, max_pages * limit) if max_pages is not None else tgt
                        if eff_tgt <= 0 or cnt >= eff_tgt:
                            already_completed_updates[("tiki", pid_str)] = {"status": "completed"}
                    elif max_pages is not None and cnt >= max_pages * limit:
                        already_completed_updates[("tiki", pid_str)] = {"status": "completed"}
                if already_completed_updates:
                    update_product_seeds_file(seeds_file, already_completed_updates)

    total_new_reviews = 0
    crawled_products = 0
    skipped_products = 0
    batch_crawled_at = datetime.now(timezone.utc).isoformat()

    product_iter = tqdm(unique_product_ids, desc="products", unit="product") if tqdm else unique_product_ids

    for product_id in product_iter:
        current_count = product_review_counts.get(str(product_id), 0)
        target_count = product_target_counts.get(product_id)

        # Kiểm tra điều kiện bỏ qua nếu sản phẩm đã hoàn thành (Resume)
        if resume:
            if target_count is not None:
                effective_target = (
                    min(target_count, max_pages * limit) if max_pages is not None else target_count
                )
                if effective_target <= 0 or current_count >= effective_target:
                    skipped_products += 1
                    continue
            elif max_pages is not None and current_count >= max_pages * limit:
                skipped_products += 1
                continue

        product_name = product_names.get(product_id) or crawler.get_product_name(product_id)
        cat_id, cat_name = product_categories.get(product_id, (None, None))

        # Cào các review của sản phẩm
        product_reviews = crawler.crawl_product_reviews(
            product_id=product_id,
            product_name=product_name,
            max_pages=max_pages,
            limit=limit,
            sort=sort,
            category_id=cat_id,
            category_name=cat_name,
            crawled_at=batch_crawled_at,
        )

        # Lọc bỏ các review đã có trong file (deduplication)
        new_reviews = []
        for r in product_reviews:
            rid_str = str(r.review_id) if r.review_id is not None else None
            if rid_str and rid_str not in existing_review_ids:
                existing_review_ids.add(rid_str)
                new_reviews.append(r)
            elif not rid_str:
                new_reviews.append(r)

        # Ghi ngay lập tức ra đĩa (Per-product Flush - chống mất dữ liệu khi gặp sự cố)
        if new_reviews:
            save_reviews(
                new_reviews,
                output=output_path,
                file_format=file_format,
                append=True,
                dedup_key="review_id",
            )
            total_new_reviews += len(new_reviews)
            product_review_counts[str(product_id)] = current_count + len(new_reviews)

        # Cập nhật trạng thái status='completed' cho sản phẩm trong file seeds
        final_count = product_review_counts.get(str(product_id), 0)
        is_completed = False
        if target_count is not None:
            eff = min(target_count, max_pages * limit) if max_pages is not None else target_count
            if eff <= 0 or final_count >= eff:
                is_completed = True
        elif max_pages is not None:
            if final_count >= max_pages * limit or len(product_reviews) < limit:
                is_completed = True
        else:
            is_completed = True

        if is_completed and seeds_file.exists():
            mark_product_seed_completed(seeds_file, product_id=product_id, source="tiki", status="completed")

        crawled_products += 1

    print(
        f"Saved {total_new_reviews} new reviews from {crawled_products} products "
        f"(skipped {skipped_products} already completed) to {output_path}"
    )
    return output_path, total_new_reviews, crawled_products


def main(
    source: str | int | Iterable[str | int] | None = None,
    max_pages: int | None = 3,
    *,
    product_id: int | Iterable[int] | None = None,
    url: str | Iterable[str] | None = None,
    keyword: str | None = None,
    search_pages: int = 1,
    max_products: int | None = None,
    limit: int = 20,
    sleep: float = 0.4,
    sort: str = DEFAULT_SORT,
    output: str | Path = "data/bronze/tiki_reviews.csv",
    file_format: str | None = None,
    append: bool = True,
    resume: bool = True,
):
    if source is None and product_id is None and url is None and keyword is None:
        args = build_arg_parser().parse_args()
        output_path, review_count, product_count = crawl_reviews_to_file(
            platform=args.platform,
            product_ids=args.product_id,
            urls=args.url,
            keyword=args.keyword,
            from_seeds=args.from_seeds,
            search_pages=args.search_pages,
            max_products=args.max_products,
            max_pages=args.max_pages,
            limit=args.limit,
            sleep=args.sleep,
            sort=args.sort,
            output=args.output,
            file_format=args.format,
            append=not args.overwrite,
            resume=args.resume,
        )
        return output_path

    if source is not None:
        if any(value is not None for value in (product_id, url, keyword)):
            raise ValueError("Use either source or explicit product_id/url/keyword, not both.")
        product_id, url, keyword = source_to_arguments(source)

    product_ids = as_list(product_id) if product_id is not None else None
    urls = as_list(url) if url is not None else None

    output_path, review_count, product_count = crawl_reviews_to_file(
        product_ids=product_ids,
        urls=urls,
        keyword=keyword,
        search_pages=search_pages,
        max_products=max_products,
        max_pages=max_pages,
        limit=limit,
        sleep=sleep,
        sort=sort,
        output=output,
        file_format=file_format,
        append=append,
        resume=resume,
    )
    return output_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        main(
            "https://tiki.vn/binh-nuoc-giu-nhiet-500ml-bang-thep-khong-gi-304-hai-lop-p278528224.html?spid=278528234",
            max_pages=15,
        )
