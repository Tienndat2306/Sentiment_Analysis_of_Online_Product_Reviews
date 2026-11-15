from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterable

import requests

from src.crawler.model import TikiCrawlerError, TikiReview
from src.crawler.utils import parse_datetime, safe_int

try:
    from fake_useragent import UserAgent
except ImportError:
    UserAgent = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


TIKI_REVIEW_API = "https://tiki.vn/api/v2/reviews"
TIKI_PRODUCT_API = "https://tiki.vn/api/v2/products/{product_id}"
TIKI_SEARCH_API = "https://tiki.vn/api/v2/products"
DEFAULT_SORT = "score|desc,id|desc,stars|all"

# 7 danh mục cấp cao nhất của Tiki (theo cấu hình hạt giống categories.yaml)
TIKI_ROOT_CATEGORIES = {
    "1789": "Điện Thoại - Máy Tính Bảng",
    "1882": "Điện Gia Dụng",
    "2549": "Đồ Chơi - Mẹ & Bé",
    "1520": "Làm Đẹp - Sức Khỏe",
    "1883": "Nhà Cửa - Đời Sống",
    "8322": "Nhà Sách Tiki",
    "4384": "Bách Hóa Online",
}

# Bản đồ ánh xạ danh mục cấp con về danh mục cấp cao nhất
SUBCATEGORY_TO_ROOT = {
    "4386": ("1883", "Nhà Cửa - Đời Sống"),   # Vệ sinh nhà cửa -> Nhà Cửa - Đời Sống
    "4399": ("1883", "Nhà Cửa - Đời Sống"),   # Vệ sinh nhà bếp -> Nhà Cửa - Đời Sống
    "15078": ("1883", "Nhà Cửa - Đời Sống"),  # Chăm sóc nhà cửa -> Nhà Cửa - Đời Sống
}


class TikiReviewCrawler:
    def __init__(self, timeout: int = 20, sleep: float = 0.3):
        self.timeout = timeout
        self.sleep = sleep
        self.session = requests.Session()
        self.session.headers.update(
            {
                "accept": "application/json, text/plain, */*",
                "referer": "https://tiki.vn/",
                "user-agent": self._user_agent(),
                "x-guest-token": "",
            }
        )
        self._seeds_cache: dict[int, tuple[str, str, str]] | None = None

    def _get_seeds_cache(self) -> dict[int, tuple[str, str, str]]:
        if self._seeds_cache is not None:
            return self._seeds_cache

        cache = {}
        seeds_path = Path("data/seeds/products.jsonl")
        if seeds_path.exists():
            try:
                with seeds_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if line_str:
                            item = json.loads(line_str)
                            if item.get("source") == "tiki":
                                pid = int(item["product_id"])
                                cache[pid] = (
                                    item.get("product_name"),
                                    str(item.get("category_id")),
                                    item.get("category_name"),
                                )
            except Exception:
                pass
        self._seeds_cache = cache
        return cache

    def resolve_category(
        self,
        product_id: int,
        category_id: str | None = None,
        category_name: str | None = None,
    ) -> tuple[str | None, str | None]:
        """
        Đảm bảo danh mục luôn là danh mục cấp cao nhất (root category)
        1. Nếu là danh mục con đã biết (4386, 4399, 15078...) -> ánh xạ về root (1883 Nhà Cửa - Đời Sống).
        2. Nếu category_id nằm trong 7 root categories -> giữ nguyên.
        3. Nếu thiếu -> tra cứu trong seed products.jsonl.
        4. Nếu seed chưa có -> tra cứu breadcrumbs từ TIKI_PRODUCT_API.
        """
        cid_str = str(category_id) if category_id is not None else None

        # 1. Ánh xạ danh mục con đã biết
        if cid_str in SUBCATEGORY_TO_ROOT:
            return SUBCATEGORY_TO_ROOT[cid_str]

        # 2. Nếu đã là root category hợp lệ
        if cid_str in TIKI_ROOT_CATEGORIES:
            return cid_str, TIKI_ROOT_CATEGORIES[cid_str]

        # 3. Tra cứu từ seed cache
        seeds_cache = self._get_seeds_cache()
        if product_id in seeds_cache:
            _, s_cid, s_cname = seeds_cache[product_id]
            if s_cid in SUBCATEGORY_TO_ROOT:
                return SUBCATEGORY_TO_ROOT[s_cid]
            if s_cid in TIKI_ROOT_CATEGORIES:
                return s_cid, TIKI_ROOT_CATEGORIES[s_cid]
            return s_cid, s_cname

        # 4. Tra cứu API chi tiết sản phẩm qua breadcrumbs
        try:
            payload = self._get_json(TIKI_PRODUCT_API.format(product_id=product_id))
            breadcrumbs = payload.get("breadcrumbs") or []
            for bc in breadcrumbs:
                bc_id = str(bc.get("category_id"))
                if bc_id in TIKI_ROOT_CATEGORIES:
                    return bc_id, TIKI_ROOT_CATEGORIES[bc_id]
                if bc_id in SUBCATEGORY_TO_ROOT:
                    return SUBCATEGORY_TO_ROOT[bc_id]
        except Exception:
            pass

        return category_id, category_name

    def crawl_product_reviews(
        self,
        product_id: int,
        product_name: str | None = None,
        max_pages: int | None = None,
        limit: int = 20,
        sort: str = DEFAULT_SORT,
        category_id: str | None = None,
        category_name: str | None = None,
        crawled_at: str | None = None,
    ):
        reviews: list[TikiReview] = []
        page = 1
        total_pages = max_pages
        now_crawled_at = crawled_at or datetime.now(timezone.utc).isoformat()

        # Chuẩn hóa danh mục về cấp cao nhất
        category_id, category_name = self.resolve_category(
            product_id=product_id,
            category_id=category_id,
            category_name=category_name,
        )

        progress = tqdm(desc=f"product {product_id}", unit="page") if tqdm else None
        try:
            while total_pages is None or page <= total_pages:
                payload = self.fetch_reviews_page(product_id, page, limit, sort)
                items = payload.get("data") or []
                paging = payload.get("paging") or {}

                if total_pages is None:
                    last_page = paging.get("last_page") or paging.get("total_page")
                    total_pages = int(last_page) if last_page else page
                    if max_pages is not None:
                        total_pages = min(total_pages, max_pages)

                if not items:
                    break

                for item in items:
                    reviews.append(
                        self._parse_review(
                            item,
                            product_id,
                            product_name,
                            category_id=category_id,
                            category_name=category_name,
                            crawled_at=now_crawled_at,
                        )
                    )

                if progress is not None:
                    progress.update(1)
                page += 1
                time.sleep(self.sleep)
        finally:
            if progress is not None:
                progress.close()

        return reviews

    def crawl_many_products(
        self,
        product_ids: Iterable[int],
        max_pages: int | None = None,
        limit: int = 20,
        sort: str = DEFAULT_SORT,
        product_names: dict[int, str] | None = None,
        product_categories: dict[int, tuple[str, str]] | None = None,
        crawled_at: str | None = None,
    ):
        all_reviews: list[TikiReview] = []
        unique_product_ids = list(dict.fromkeys(product_ids))
        product_iter = tqdm(unique_product_ids, desc="products", unit="product") if tqdm else unique_product_ids
        product_names = product_names or {}
        product_categories = product_categories or {}
        batch_crawled_at = crawled_at or datetime.now(timezone.utc).isoformat()

        for product_id in product_iter:
            product_name = product_names.get(product_id) or self.get_product_name(product_id)
            cat_id, cat_name = product_categories.get(product_id, (None, None))
            all_reviews.extend(
                self.crawl_product_reviews(
                    product_id=product_id,
                    product_name=product_name,
                    max_pages=max_pages,
                    limit=limit,
                    sort=sort,
                    category_id=cat_id,
                    category_name=cat_name,
                    crawled_at=batch_crawled_at,
                )
            )
        return all_reviews

    def search_product_ids(self, keyword: str, pages: int = 1, limit: int = 40):
        ids: list[int] = []
        for page in range(1, pages + 1):
            params = {
                "limit": limit,
                "page": page,
                "q": keyword,
                "include": "advertisement",
            }
            payload = self._get_json(TIKI_SEARCH_API, params=params)
            for item in payload.get("data") or []:
                product_id = item.get("id")
                if product_id is not None:
                    ids.append(int(product_id))
            time.sleep(self.sleep)
        return list(dict.fromkeys(ids))

    def get_product_name(self, product_id: int):
        try:
            payload = self._get_json(TIKI_PRODUCT_API.format(product_id=product_id))
        except TikiCrawlerError:
            return None
        return payload.get("name")

    def fetch_reviews_page(
        self,
        product_id: int,
        page: int = 1,
        limit: int = 20,
        sort: str = DEFAULT_SORT,
    ):
        params = {
            "product_id": product_id,
            "limit": limit,
            "page": page,
            "include": "comments,contribute_info",
            "sort": sort,
        }
        return self._get_json(TIKI_REVIEW_API, params=params)

    def _get_json(self, url: str, params: dict[str, Any] | None = None):
        last_error = None
        for attempt in range(1, 5):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code in {403, 429, 500, 502, 503, 504}:
                    raise TikiCrawlerError(f"Tiki returned HTTP {response.status_code}: {response.url}")
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, TikiCrawlerError) as exc:
                last_error = exc
                if attempt == 4:
                    break
                time.sleep(min(2 ** (attempt - 1), 10))
        raise TikiCrawlerError(str(last_error)) from last_error

    @staticmethod
    def _parse_review(
        item: dict[str, Any],
        product_id: int,
        product_name: str | None,
        category_id: str | None = None,
        category_name: str | None = None,
        crawled_at: str | None = None,
    ):
        created_at = item.get("created_at")
        customer = item.get("created_by") or item.get("customer") or {}
        seller = item.get("seller") or {}
        images = item.get("images") or []

        return TikiReview(
            review_id=safe_int(item.get("id")),
            product_id=product_id,
            product_name=product_name or item.get("product_name"),
            customer_id=safe_int(customer.get("id")),
            customer_name=customer.get("name") or customer.get("full_name"),
            rating=safe_int(item.get("rating")),
            title=item.get("title"),
            content=item.get("content"),
            thank_count=safe_int(item.get("thank_count")),
            comment_count=safe_int(item.get("comment_count")),
            created_at=parse_datetime(created_at),
            is_photo=bool(images),
            seller_name=seller.get("name") or item.get("seller_name"),
            category_id=category_id,
            category_name=category_name,
            source="tiki",
            crawled_at=crawled_at or datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _user_agent():
        try:
            if UserAgent is not None:
                return UserAgent().random
        except Exception:
            pass
        return (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
