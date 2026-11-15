from datetime import datetime, timezone
import argparse
import hashlib
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

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
from src.crawler.model import LazadaCrawlerError, LazadaReview
from src.crawler.utils import extract_lazada_product_id, parse_datetime, safe_int

logger = logging.getLogger("LazadaReviewCrawler")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

LAZADA_REVIEW_PAGE = "https://www.lazada.vn/product-reviews/?item_id={product_id}"


class LazadaReviewCrawler:
    """
    Crawler thu thập đánh giá sản phẩm từ Lazada sử dụng Playwright Chromium.
    Kết hợp giữa việc truy cập trực tiếp trang chuyên biệt /product-reviews/?item_id={id}
    và bóc tách DOM sạch kết hợp bắt gói tin API mtop.lazada.review.item.getpcreviewlist.
    """

    def __init__(
        self,
        headless: bool = True,
        timeout: int = 30000,
        sleep: float = 1.0,
        user_agent: str | None = None,
    ):
        self.headless = headless
        self.timeout = timeout
        self.sleep = sleep
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start(self):
        """Khởi động trình duyệt Playwright Chromium nếu chưa bật."""
        if self._playwright is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                ],
            )
            self._context = self._browser.new_context(
                user_agent=self.user_agent,
                viewport={"width": 1366, "height": 768},
                locale="vi-VN",
                timezone_id="Asia/Ho_Chi_Minh",
            )
            self._page = self._context.new_page()

    def close(self):
        """Giải phóng tài nguyên trình duyệt."""
        if self._page:
            try:
                self._page.close()
            except Exception:
                pass
            self._page = None
        if self._context:
            try:
                self._context.close()
            except Exception:
                pass
            self._context = None
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    def _ensure_page(self) -> Page:
        if self._page is None or self._page.is_closed():
            self.start()
        return self._page

    def crawl_product_reviews(
        self,
        product_id_or_url: str | int,
        product_name: str | None = None,
        max_pages: int | None = 2,
        max_reviews: int | None = None,
        category_id: str | None = None,
        category_name: str | None = None,
        crawled_at: str | None = None,
    ) -> list[LazadaReview]:
        """
        Crawl đánh giá cho một sản phẩm Lazada.
        """
        page = self._ensure_page()
        now_crawled_at = crawled_at or datetime.now(timezone.utc).isoformat()
        try:
            product_id = extract_lazada_product_id(product_id_or_url)
        except Exception as e:
            logger.warning(f"Không nhận diện được ID Lazada từ {product_id_or_url}: {e}")
            return []

        target_url = LAZADA_REVIEW_PAGE.format(product_id=product_id)
        logger.info(f"[Lazada] Đang cào review sản phẩm ID: {product_id} ({target_url})")

        intercepted_api_items: list[dict[str, Any]] = []

        def on_response(res):
            if "getpcreviewlist" in res.url.lower():
                try:
                    payload = res.json()
                    items = payload.get("data", {}).get("items", [])
                    if items:
                        intercepted_api_items.extend(items)
                except Exception:
                    pass

        page.on("response", on_response)

        try:
            response = page.goto(target_url, timeout=self.timeout, wait_until="domcontentloaded")
            page.wait_for_timeout(2500)
        except Exception as e:
            logger.error(f"[Lazada] Lỗi khi truy cập {target_url}: {e}")
            return []

        # Trích xuất tên sản phẩm từ tiêu đề trang nếu chưa có
        if not product_name:
            try:
                title = page.title()
                # Thường có dạng: "Đánh Giá [Tên sản phẩm] – Trải Nghiệm Người Dùng reviews"
                m = re.search(r"Đánh Giá\s+(.*?)\s+[–-]\s+Trải Nghiệm", title, re.IGNORECASE)
                if m:
                    product_name = m.group(1).strip()
                elif title and "reviews" in title.lower():
                    product_name = title.replace("reviews", "").replace("Đánh Giá", "").strip(" –-|")
                else:
                    product_name = f"Lazada Product {product_id}"
            except Exception:
                product_name = f"Lazada Product {product_id}"

        reviews: list[LazadaReview] = []
        seen_keys: set[str] = set()

        current_page = 1
        limit_pages = max_pages or 1

        while current_page <= limit_pages:
            # Cuộn xuống để kích hoạt toàn bộ thẻ review hiển thị
            page.mouse.wheel(0, 1500)
            page.wait_for_timeout(1000)

            # 1. Bóc tách trực tiếp từ DOM của trang
            html_content = page.content()
            dom_reviews = self._parse_reviews_from_html(
                html_content,
                product_id,
                product_name,
                category_id=category_id,
                category_name=category_name,
                crawled_at=now_crawled_at,
            )

            for rev in dom_reviews:
                key = f"{rev.customer_name}_{rev.created_at}_{rev.content}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    reviews.append(rev)
                    if max_reviews and len(reviews) >= max_reviews:
                        break

            if max_reviews and len(reviews) >= max_reviews:
                break

            # 2. Xử lý phân trang nếu cần lấy thêm trang
            if current_page < limit_pages:
                has_next = self._navigate_next_page(page)
                if not has_next:
                    break
                current_page += 1
                time.sleep(self.sleep)
            else:
                break

            # 3. Kết hợp thêm review bắt được từ API ngầm (nếu có bổ sung)
        if intercepted_api_items:
            for item in intercepted_api_items:
                api_rev = self._parse_review_from_api_item(
                    item,
                    product_id,
                    product_name,
                    category_id=category_id,
                    category_name=category_name,
                    crawled_at=now_crawled_at,
                )
                key = f"{api_rev.customer_name}_{api_rev.created_at}_{api_rev.content}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    reviews.append(api_rev)
                    if max_reviews and len(reviews) >= max_reviews:
                        break

        logger.info(f"[Lazada] Đã thu thập {len(reviews)} đánh giá cho sản phẩm {product_id}.")
        return reviews

    def crawl_many_products(
        self,
        product_sources: Iterable[str | int],
        product_names: dict[str, str] | None = None,
        product_categories: dict[str, tuple[str, str]] | None = None,
        max_pages: int | None = 2,
        max_reviews: int | None = None,
        crawled_at: str | None = None,
    ) -> list[LazadaReview]:
        """
        Crawl đánh giá cho nhiều sản phẩm Lazada trong cùng 1 phiên trình duyệt.
        """
        self.start()
        all_reviews: list[LazadaReview] = []
        product_list = list(dict.fromkeys(product_sources))
        product_names = product_names or {}
        product_categories = product_categories or {}
        batch_crawled_at = crawled_at or datetime.now(timezone.utc).isoformat()

        iterator = tqdm(product_list, desc="Lazada Products", unit="product") if tqdm else product_list

        for src in iterator:
            try:
                p_id = extract_lazada_product_id(src)
                name = product_names.get(str(p_id)) or product_names.get(str(src))
                cat_id, cat_name = product_categories.get(str(p_id), (None, None))
                revs = self.crawl_product_reviews(
                    product_id_or_url=p_id,
                    product_name=name,
                    max_pages=max_pages,
                    max_reviews=max_reviews,
                    category_id=cat_id,
                    category_name=cat_name,
                    crawled_at=batch_crawled_at,
                )
                all_reviews.extend(revs)
                time.sleep(self.sleep)
            except Exception as e:
                logger.error(f"[Lazada] Lỗi khi cào sản phẩm {src}: {e}")

        return all_reviews

    def _navigate_next_page(self, page: Page) -> bool:
        """Thử chuyển sang trang review tiếp theo."""
        try:
            # Tìm nút phân trang trang tiếp theo
            next_btn = page.query_selector(
                "button.next-pagination-item, button.ant-pagination-next, "
                ".next-btn, li.ant-pagination-next:not(.ant-pagination-disabled)"
            )
            if next_btn and next_btn.is_visible() and next_btn.is_enabled():
                next_btn.click()
                page.wait_for_timeout(2000)
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _parse_reviews_from_html(
        html: str,
        product_id: str | int,
        product_name: str | None,
        category_id: str | None = None,
        category_name: str | None = None,
        crawled_at: str | None = None,
    ) -> list[LazadaReview]:
        """Bóc tách các review items từ cấu trúc DOM HTML trang /product-reviews/."""
        if not BeautifulSoup:
            return []

        soup = BeautifulSoup(html, "html.parser")
        items = soup.find_all(
            lambda tag: tag.has_attr("class") and any("reviewItem" in c for c in tag["class"])
        )

        results: list[LazadaReview] = []
        now_crawled = crawled_at or datetime.now(timezone.utc).isoformat()
        for it in items:
            # 1. Tên người mua
            user_tag = it.find(
                lambda t: t.has_attr("class") and any("username" in c for c in t["class"])
            )
            customer_name = user_tag.get_text(strip=True) if user_tag else "Khách hàng Lazada"

            # 2. Ngày đánh giá
            date_tag = it.find(
                lambda t: t.has_attr("class") and any("dateLocation" in c for c in t["class"])
            )
            raw_date = date_tag.get_text(strip=True) if date_tag else None
            created_at = parse_datetime(raw_date) if raw_date else None

            # 3. Số sao đánh giá (đếm các sao màu vàng rgb(255, 200, 60))
            star_items = it.find_all("div", class_="i-rate-star-item")
            if star_items:
                rating = sum(
                    1
                    for s in star_items
                    if s.find(lambda p: p.name == "path" and "255, 200, 60" in p.get("style", ""))
                )
            else:
                rating = 5  # Mặc định

            # 4. Nội dung đánh giá
            text_tag = it.find(
                lambda t: t.has_attr("class") and any("reviewText" in c for c in t["class"])
            )
            if text_tag:
                content = text_tag.get_text(strip=True)
            else:
                # Fallback tìm các đoạn văn bản
                texts = []
                for elem in it.find_all(["div", "p", "span"]):
                    cls = " ".join(elem.get("class", []))
                    if any(k in cls for k in ["username", "dateLocation", "i-rate", "helpful"]):
                        continue
                    t = elem.get_text(strip=True)
                    if t and t not in [customer_name, raw_date] and len(t) > 3:
                        texts.append(t)
                content = " ".join(dict.fromkeys(texts)) if texts else ""

            # 5. Lượt hữu ích (Hữu ích (0))
            helpful_tag = it.find(
                lambda t: t.has_attr("class") and any("helpfulCount" in c for c in t["class"])
            )
            helpful_text = helpful_tag.get_text(strip=True) if helpful_tag else ""
            m_help = re.search(r"\((\d+)\)", helpful_text)
            thank_count = int(m_help.group(1)) if m_help else 0

            # 6. Có hình ảnh không
            has_photo = bool(it.find("img") or it.find(lambda t: t.has_attr("class") and any("mediaItem" in c for c in t["class"])))

            # Sinh ID review giả định nếu DOM không có ID
            hash_input = f"{product_id}_{customer_name}_{created_at}_{content}"
            generated_id = hashlib.md5(hash_input.encode("utf-8")).hexdigest()[:16]

            results.append(
                LazadaReview(
                    review_id=generated_id,
                    product_id=str(product_id),
                    product_name=product_name,
                    customer_id=None,
                    customer_name=customer_name,
                    rating=rating if rating > 0 else 5,
                    title=None,
                    content=content,
                    thank_count=thank_count,
                    comment_count=0,
                    created_at=created_at,
                    is_photo=has_photo,
                    seller_name=None,
                    category_id=category_id,
                    category_name=category_name,
                    source="lazada",
                    crawled_at=now_crawled,
                )
            )

        return results

    @staticmethod
    def _parse_review_from_api_item(
        item: dict[str, Any],
        product_id: str | int,
        product_name: str | None,
        category_id: str | None = None,
        category_name: str | None = None,
        crawled_at: str | None = None,
    ) -> LazadaReview:
        """Bóc tách từ JSON response mtop.lazada.review.item.getpcreviewlist."""
        buyer = item.get("buyerName") or item.get("userName") or "Khách hàng Lazada"
        raw_date = item.get("reviewTime") or item.get("createTime")
        created_at = parse_datetime(raw_date)
        content = item.get("reviewContent") or item.get("content") or ""
        rating = safe_int(item.get("rating")) or 5
        images = item.get("images") or item.get("photos") or []
        thank_count = safe_int(item.get("usefulCount") or item.get("likeCount")) or 0
        review_id = str(item.get("reviewId") or item.get("id") or "")
        now_crawled = crawled_at or datetime.now(timezone.utc).isoformat()

        return LazadaReview(
            review_id=review_id or None,
            product_id=str(product_id),
            product_name=product_name,
            customer_id=str(item.get("buyerId") or "") or None,
            customer_name=buyer,
            rating=rating,
            title=item.get("reviewTitle"),
            content=content,
            thank_count=thank_count,
            comment_count=0,
            created_at=created_at,
            is_photo=bool(images),
            seller_name=None,
            category_id=category_id,
            category_name=category_name,
            source="lazada",
            crawled_at=now_crawled,
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Crawl Lazada product reviews.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--product-id", action="append", help="Lazada product ID. Có thể lặp lại.")
    source.add_argument("--url", action="append", help="Lazada product URL. Có thể lặp lại.")
    source.add_argument("--from-seeds", help="Đường dẫn tới file seeds (JSONL) chứa danh sách sản phẩm.")

    parser.add_argument("--max-products", type=int, help="Số lượng sản phẩm tối đa cần crawl.")
    parser.add_argument("--max-pages", type=int, default=2, help="Số trang review mỗi sản phẩm.")
    parser.add_argument("--max-reviews", type=int, help="Số review tối đa mỗi sản phẩm.")
    parser.add_argument("--headless", action="store_true", default=True, help="Chạy browser ở chế độ headless.")
    parser.add_argument("--no-headless", action="store_false", dest="headless", help="Hiện giao diện browser để debug.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Độ trễ giữa các lượt cào (giây).")
    parser.add_argument("--output", default="data/bronze/lazada_reviews.csv", help="Đường dẫn file đầu ra.")
    parser.add_argument("--format", choices=["csv", "json", "jsonl", "parquet"], help="Định dạng file lưu.")
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè file nếu đã tồn tại.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bỏ qua các sản phẩm đã cào đủ trong file output (mặc định: True).",
    )
    return parser


def crawl_lazada_reviews_to_file(
    *,
    product_ids: Iterable[str | int] | None = None,
    urls: Iterable[str] | None = None,
    from_seeds: str | Path | None = None,
    max_products: int | None = None,
    max_pages: int | None = 2,
    max_reviews: int | None = None,
    headless: bool = True,
    sleep: float = 1.0,
    output: str | Path = "data/bronze/lazada_reviews.csv",
    file_format: str | None = None,
    append: bool = True,
    resume: bool = True,
):
    sources = []
    names = {}
    categories = {}

    if product_ids is not None:
        sources.extend(list(product_ids))
    elif urls is not None:
        sources.extend(list(urls))
    elif from_seeds is not None:
        from src.crawler.io import load_product_seeds
        seeds = load_product_seeds(from_seeds)
        lazada_seeds = [s for s in seeds if s.source.lower() == "lazada"]
        for s in lazada_seeds:
            sources.append(s.product_id)
            if s.product_name:
                names[str(s.product_id)] = s.product_name
            categories[str(s.product_id)] = (s.category_id, s.category_name)

    # Nạp danh mục từ seeds làm fallback nếu cào trực tiếp theo ID/URL
    default_seeds_file = Path("data/seeds/products.jsonl")
    if not categories and default_seeds_file.exists():
        from src.crawler.io import load_product_seeds
        for s in load_product_seeds(default_seeds_file):
            if s.source.lower() == "lazada":
                pid_str = str(s.product_id)
                if pid_str not in categories:
                    categories[pid_str] = (s.category_id, s.category_name)
                if pid_str not in names and s.product_name:
                    names[pid_str] = s.product_name

    if max_products is not None:
        sources = sources[:max_products]

    output_path = Path(output)
    existing_review_ids = set()
    product_review_counts = {}

    if not append and output_path.exists():
        output_path.unlink()

    if append and output_path.exists() and output_path.stat().st_size > 0:
        existing_review_ids, product_review_counts = get_existing_review_stats(output_path)
        if existing_review_ids:
            logger.info(
                f"[Resume] Đã nạp {len(existing_review_ids):,} reviews của "
                f"{len(product_review_counts)} sản phẩm từ {output_path}."
            )

    # Lọc bỏ các sản phẩm đã hoàn thành nếu resume=True
    if resume and sources:
        target_counts = {}
        target_seeds_file = Path(from_seeds) if from_seeds else default_seeds_file
        if target_seeds_file.exists():
            from src.crawler.io import load_product_seeds
            for s in load_product_seeds(target_seeds_file):
                if s.source.lower() == "lazada":
                    target_counts[str(s.product_id)] = safe_int(s.review_count) or 0

        filtered_sources = []
        for src in sources:
            p_id = str(extract_lazada_product_id(src))
            cur_cnt = product_review_counts.get(p_id, 0)
            target = target_counts.get(p_id)
            if target is not None:
                eff_target = min(target, (max_reviews or max_pages * 10))
                if eff_target <= 0 or cur_cnt >= eff_target:
                    continue
            elif max_reviews and cur_cnt >= max_reviews:
                continue
            filtered_sources.append(src)

        skipped_count = len(sources) - len(filtered_sources)
        if skipped_count > 0:
            logger.info(f"[Resume] Đã bỏ qua {skipped_count} sản phẩm đã cào đủ. Còn lại {len(filtered_sources)} sản phẩm.")
        sources = filtered_sources

    if not sources:
        logger.warning("Không có sản phẩm Lazada nào cần cào (đã hoàn thành hoặc danh sách rỗng).")
        return output_path

    total_new = 0
    with LazadaReviewCrawler(headless=headless, sleep=sleep) as crawler:
        iterator = tqdm(sources, desc="Lazada Products", unit="product") if tqdm else sources
        batch_crawled_at = datetime.now(timezone.utc).isoformat()
        for src in iterator:
            try:
                p_id = extract_lazada_product_id(src)
                name = names.get(str(p_id)) or names.get(str(src))
                cat_id, cat_name = categories.get(str(p_id), (None, None))
                revs = crawler.crawl_product_reviews(
                    product_id_or_url=p_id,
                    product_name=name,
                    max_pages=max_pages,
                    max_reviews=max_reviews,
                    category_id=cat_id,
                    category_name=cat_name,
                    crawled_at=batch_crawled_at,
                )
                new_revs = [r for r in revs if str(r.review_id) not in existing_review_ids]
                if new_revs:
                    save_reviews(
                        new_revs,
                        output=output_path,
                        file_format=file_format,
                        append=True,
                        dedup_key="review_id",
                    )
                    existing_review_ids.update(str(r.review_id) for r in new_revs)
                    total_new += len(new_revs)

                if target_seeds_file.exists():
                    mark_product_seed_completed(
                        target_seeds_file, product_id=p_id, source="lazada", status="completed"
                    )

                time.sleep(sleep)
            except Exception as e:
                logger.error(f"[Lazada] Lỗi khi cào sản phẩm {src}: {e}")

    logger.info(f"Đã lưu thành công {total_new} đánh giá Lazada mới vào {output_path}.")
    return output_path


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    crawl_lazada_reviews_to_file(
        product_ids=args.product_id,
        urls=args.url,
        from_seeds=args.from_seeds,
        max_products=args.max_products,
        max_pages=args.max_pages,
        max_reviews=args.max_reviews,
        headless=args.headless,
        sleep=args.sleep,
        output=args.output,
        file_format=args.format,
        append=not args.overwrite,
        resume=args.resume,
    )
