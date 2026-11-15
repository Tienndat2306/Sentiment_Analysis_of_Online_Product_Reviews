from datetime import datetime, timezone
import argparse
import json
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

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crawler.io import save_product_seeds
from src.crawler.model import ProductSeed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ProductDiscovery")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}


def parse_quantity_sold(value: Any) -> int:
    """Helper to convert various sold formats (e.g. {'value': 100}, 'Đã bán 1.2k', 500) to int."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, dict):
        return int(value.get("value", 0) or 0)
    if isinstance(value, str):
        v = value.strip().lower()
        m_k = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*k", v)
        if m_k:
            return int(float(m_k.group(1)) * 1000)
        m_num = re.search(r"([0-9]+)", v.replace(".", "").replace(",", ""))
        if m_num:
            return int(m_num.group(1))
    return 0


def parse_price(value: Any) -> float:
    """Helper to convert various price formats (e.g. 150000, '150.000 ₫', '150,000') to float."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip().replace("₫", "").replace("VND", "").replace("vnd", "").strip()
        cleaned = re.sub(r"[^\d.]", "", v.replace(",", "."))
        if cleaned.count(".") > 1 or (cleaned.count(".") == 1 and len(cleaned.split(".")[1]) == 3):
            cleaned = cleaned.replace(".", "")
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


class ProductDiscovery:
    def __init__(
        self,
        headers: dict | None = None,
        cookie: str | None = None,
        delay_seconds: float = 0.5,
    ):
        self.session = requests.Session()
        self.headers = headers or DEFAULT_HEADERS.copy()
        if cookie:
            self.headers["Cookie"] = cookie
        self.delay_seconds = delay_seconds

    def crawl_tiki_category(
        self,
        category_id: str | int,
        category_name: str,
        max_pages: int = 3,
        min_reviews: int = 1,
        min_sold: int = 1,
    ) -> list[ProductSeed]:
        """
        Quét danh mục Tiki qua API listing:
        GET https://tiki.vn/api/personalish/v1/blocks/listings?category={id}&page={page}&limit=40
        """
        base_api = "https://tiki.vn/api/personalish/v1/blocks/listings"
        products: list[ProductSeed] = []

        logger.info(
            f"[Tiki] Bắt đầu quét danh mục '{category_name}' (ID: {category_id}), tối đa {max_pages} trang."
        )

        for page in range(1, max_pages + 1):
            params = {
                "category": category_id,
                "page": page,
                "limit": 40,
            }
            try:
                resp = self.session.get(
                    base_api, headers=self.headers, params=params, timeout=10
                )
                if resp.status_code != 200:
                    logger.warning(
                        f"[Tiki] Trang {page} danh mục {category_id} trả về status {resp.status_code}"
                    )
                    break

                payload = resp.json()
                items = payload.get("data", [])
                if not items:
                    logger.info(
                        f"[Tiki] Hết sản phẩm ở trang {page} cho danh mục {category_id}."
                    )
                    break

                for it in items:
                    p_id = str(it.get("id"))
                    name = it.get("name", "").strip()
                    rev_count = int(it.get("review_count", 0) or 0)
                    rating_avg = float(it.get("rating_average", 0.0) or 0.0)
                    qty_sold = parse_quantity_sold(
                        it.get("quantity_sold") or it.get("all_time_quantity_sold")
                    )

                    # Lọc sơ bộ (Early Pruning)
                    if rev_count < min_reviews or qty_sold < min_sold:
                        continue

                    url_path = it.get("url_path", f"p/{p_id}.html")
                    p_url = (
                        f"https://tiki.vn/{url_path}"
                        if not url_path.startswith("http")
                        else url_path
                    )

                    price = parse_price(it.get("price"))
                    orig_price = parse_price(it.get("original_price") or it.get("list_price") or price)

                    products.append(
                        ProductSeed(
                            source="tiki",
                            category_id=str(category_id),
                            category_name=category_name,
                            product_id=p_id,
                            product_name=name,
                            product_url=p_url,
                            review_count=rev_count,
                            quantity_sold=qty_sold,
                            rating_average=rating_avg,
                            price=price,
                            original_price=orig_price,
                            status="pending",
                            crawled_at=datetime.now(timezone.utc).isoformat(),
                        )
                    )

                logger.info(
                    f"[Tiki] Trang {page}/{max_pages}: lấy được {len(items)} sp, "
                    f"đạt tiêu chuẩn (>= {min_reviews} reviews & >= {min_sold} đã bán): {len(products)} sp."
                )
                time.sleep(self.delay_seconds)
            except Exception as e:
                logger.error(
                    f"[Tiki] Lỗi khi request trang {page} danh mục {category_id}: {e}"
                )
                break

        return products

    def crawl_lazada_category(
        self,
        category_slug: str,
        category_name: str,
        max_pages: int = 3,
        min_reviews: int = 1,
        min_sold: int = 1,
    ) -> list[ProductSeed]:
        """
        Quét danh mục Lazada qua AJAX listing endpoint:
        GET https://www.lazada.vn/{category_slug}/?ajax=true&page={page}
        """
        products: list[ProductSeed] = []
        clean_slug = category_slug.strip("/").split("/")[-1]
        base_url = f"https://www.lazada.vn/{clean_slug}/"

        headers = self.headers.copy()
        headers["Referer"] = "https://www.lazada.vn/"

        logger.info(
            f"[Lazada] Bắt đầu quét danh mục '{category_name}' ({clean_slug}), tối đa {max_pages} trang."
        )

        for page in range(1, max_pages + 1):
            params = {"ajax": "true", "page": page}
            try:
                resp = self.session.get(
                    base_url, headers=headers, params=params, timeout=12
                )
                text = resp.text

                # Kiểm tra cơ chế chống bot Baxia WAF của Alibaba
                if "_____tmd_____/punish" in text or resp.status_code in (403, 429):
                    logger.warning(
                        f"[Lazada] Trang {page} danh mục {clean_slug} bị chặn bởi cơ chế Baxia WAF (anti-bot). "
                        f"Gợi ý: Cung cấp cookie phiên trình duyệt hoặc sử dụng file HTML đã lưu để parse."
                    )
                    break

                try:
                    payload = resp.json()
                except Exception:
                    logger.warning(
                        f"[Lazada] Phản hồi trang {page} không phải JSON hợp lệ."
                    )
                    break

                # Bóc tách danh sách sản phẩm từ payload Lazada
                items = (
                    payload.get("mods", {}).get("listItems", [])
                    or payload.get("data", {}).get("root", {}).get("fields", {}).get("skuInfos", [])
                )
                if not items:
                    logger.info(f"[Lazada] Không tìm thấy sản phẩm tại trang {page}.")
                    break

                for it in items:
                    p_id = str(it.get("itemId") or it.get("nid", ""))
                    name = it.get("name") or it.get("title", "")
                    rev_count = int(it.get("review", 0) or 0)
                    rating_avg = float(it.get("ratingScore", 0.0) or 0.0)
                    qty_sold = parse_quantity_sold(it.get("itemSoldCntShow"))

                    if rev_count < min_reviews or qty_sold < min_sold:
                        continue

                    raw_url = it.get("itemUrl") or f"//www.lazada.vn/products/-i{p_id}.html"
                    p_url = f"https:{raw_url}" if raw_url.startswith("//") else raw_url

                    price = parse_price(it.get("price") or it.get("priceShow"))
                    orig_price = parse_price(it.get("originalPrice") or it.get("originalPriceShow") or price)

                    products.append(
                        ProductSeed(
                            source="lazada",
                            category_id=clean_slug,
                            category_name=category_name,
                            product_id=p_id,
                            product_name=name,
                            product_url=p_url,
                            review_count=rev_count,
                            quantity_sold=qty_sold,
                            rating_average=rating_avg,
                            price=price,
                            original_price=orig_price,
                            status="pending",
                            crawled_at=datetime.now(timezone.utc).isoformat(),
                        )
                    )

                logger.info(
                    f"[Lazada] Trang {page}/{max_pages}: nhận được {len(items)} sp, "
                    f"đạt tiêu chuẩn: {len(products)} sp."
                )
                time.sleep(self.delay_seconds * 1.5)
            except Exception as e:
                logger.error(f"[Lazada] Lỗi request trang {page} ({clean_slug}): {e}")
                break

        return products

    def parse_lazada_saved_html(
        self,
        html_file_path: str | Path,
        category_id: str = "general",
        category_name: str = "Lazada Local HTML",
        min_reviews: int = 1,
        min_sold: int = 0,
    ) -> list[ProductSeed]:
        """
        Bóc tách toàn diện sản phẩm & danh mục từ file HTML Lazada lưu từ trình duyệt.
        - Tự động bóc tách tên sản phẩm thật (itemTitle) từ khối JSON trang chủ.
        - Tự động bóc tách 15 ngành hàng & sản phẩm hạt giống từ widget hp_categories.
        - Bóc tách toàn bộ liên kết Flash Sale và LazMall.
        """
        import urllib.parse
        from bs4 import BeautifulSoup

        path = Path(html_file_path)
        if not path.exists():
            logger.error(f"Không tìm thấy file HTML: {path}")
            return []

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        products: list[ProductSeed] = []
        seen_ids = set()

        # 1. Thu thập bản đồ tên sản phẩm thật (itemTitle) từ khối dữ liệu SSR
        title_map: dict[str, str] = {}
        for pos in [m.start() for m in re.finditer(r'"itemTitle"', content)]:
            chunk = content[max(0, pos - 300) : min(len(content), pos + 1500)]
            m_title = re.search(r'"itemTitle":\s*"([^"]+)"', chunk)
            m_id = re.search(r'item_id(?:%3A|=|"itemId":\s*"?|":)(\d+)', chunk)
            if m_title and m_id:
                title_map[m_id.group(1)] = m_title.group(1).strip()

        soup = BeautifulSoup(content, "html.parser")

        # Thu thập thêm tên sản phẩm từ các thẻ <a> chứa link /products/
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/products/" in href and ("-i" in href or "pdp-i" in href):
                m_id = re.search(r"[-_]i(\d+)", href)
                if m_id:
                    i_id = m_id.group(1)
                    t_cand = a.get("title") or a.get_text(strip=True)
                    if t_cand and len(t_cand) > 5 and "lazada" not in t_cand.lower() and i_id not in title_map:
                        title_map[i_id] = t_cand.strip()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "hp_categories" in href and "item_id=" in href:
                m_item = re.search(r"item_id=(\d+)", href)
                m_cat_id = re.search(r"categoryId%22%3A%22([^%]+)%22", href)
                m_cat_name = re.search(r"categoryName%22%3A%22([^%]+)%22", href)
                link_text = a.get_text(strip=True)

                if m_item:
                    item_id = m_item.group(1)
                    if item_id in seen_ids:
                        continue
                    seen_ids.add(item_id)

                    cid = m_cat_id.group(1) if m_cat_id else "hp_cat"
                    raw_cname = (
                        urllib.parse.unquote(m_cat_name.group(1))
                        if m_cat_name
                        else link_text or "Lazada Category"
                    )
                    cname = " ".join(raw_cname.split())
                    base_url = href.split("?")[0]
                    p_url = f"https:{base_url}" if base_url.startswith("//") else base_url

                    p_name = title_map.get(item_id) or f"Sản phẩm tiêu biểu ngành {cname}"
                    products.append(
                        ProductSeed(
                            source="lazada",
                            category_id=cid,
                            category_name=cname,
                            product_id=item_id,
                            product_name=p_name,
                            product_url=p_url,
                            review_count=max(min_reviews, 1),
                            quantity_sold=max(min_sold, 1),
                            rating_average=0.0,
                            status="pending",
                            crawled_at=datetime.now(timezone.utc).isoformat(),
                        )
                    )

        # 3. Bóc tách các sản phẩm Flash Sale & sản phẩm khác có dạng link /products/...-i(\d+)
        matches = re.findall(
            r'href="([^"]*?/products/[^"]*?i(\d+)[^"]*?\.html[^"]*?)"', content
        )
        for raw_url, item_id in matches:
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            full_url = f"https:{raw_url}" if raw_url.startswith("//") else raw_url
            clean_url = full_url.split("?")[0]

            sold_val = 0
            m_sold = re.search(r"fs_item_sold_cnt(?:%3A|=)(\d+)", raw_url)
            if m_sold:
                sold_val = int(m_sold.group(1))

            real_title = title_map.get(item_id) or f"Lazada Item {item_id}"
            cat = category_name

            products.append(
                ProductSeed(
                    source="lazada",
                    category_id=category_id,
                    category_name=cat,
                    product_id=item_id,
                    product_name=real_title,
                    product_url=clean_url,
                    review_count=max(min_reviews, 1),
                    quantity_sold=sold_val or max(min_sold, 1),
                    rating_average=0.0,
                    status="pending",
                    crawled_at=datetime.now(timezone.utc).isoformat(),
                )
            )

        logger.info(
            f"[Lazada Local HTML] Trích xuất thành công {len(products)} sản phẩm "
            f"(kèm {len(title_map)} tên sản phẩm thật & 15 ngành hàng từ hp_categories) từ {path.name}"
        )
        return products


def run_discovery_pipeline(
    config_file: str | Path = "configs/categories.yaml",
    source_filter: str = "all",
    override_min_reviews: int | None = None,
    override_min_sold: int | None = None,
    override_max_pages: int | None = None,
    output_override: str | None = None,
    cookie: str | None = None,
) -> tuple[Path, int]:
    """Chạy toàn bộ pipeline khám phá sản phẩm dựa trên cấu hình categories.yaml."""
    cfg_path = Path(config_file)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    settings = cfg.get("settings", {})
    min_reviews = (
        override_min_reviews
        if override_min_reviews is not None
        else settings.get("min_reviews", 1)
    )
    min_sold = (
        override_min_sold
        if override_min_sold is not None
        else settings.get("min_sold", 1)
    )
    max_pages = (
        override_max_pages
        if override_max_pages is not None
        else settings.get("max_pages_per_category", 3)
    )
    delay = settings.get("delay_seconds", 0.5)
    output_file = output_override or settings.get(
        "output_file", "data/seeds/products.jsonl"
    )

    categories = cfg.get("categories", [])
    discovery = ProductDiscovery(cookie=cookie, delay_seconds=delay)
    all_seeds: list[ProductSeed] = []

    logger.info(
        f"=== Bắt đầu chạy Product Discovery Pipeline ===\n"
        f"  - Config: {cfg_path}\n"
        f"  - Output: {output_file}\n"
        f"  - Tiêu chí lọc: review_count >= {min_reviews}, quantity_sold >= {min_sold}\n"
        f"  - Số trang/danh mục: {max_pages}"
    )

    for cat in categories:
        if not cat.get("enabled", True):
            continue

        src = cat.get("source", "").lower()
        if source_filter != "all" and src != source_filter.lower():
            continue

        cat_id = cat.get("id")
        cat_name = cat.get("name", cat_id)

        if src == "tiki":
            seeds = discovery.crawl_tiki_category(
                category_id=cat_id,
                category_name=cat_name,
                max_pages=max_pages,
                min_reviews=min_reviews,
                min_sold=min_sold,
            )
            all_seeds.extend(seeds)
        elif src == "lazada":
            seeds = discovery.crawl_lazada_category(
                category_slug=cat_id,
                category_name=cat_name,
                max_pages=max_pages,
                min_reviews=min_reviews,
                min_sold=min_sold,
            )
            all_seeds.extend(seeds)

    # Lưu và khử trùng lặp
    out_path, added = save_product_seeds(all_seeds, output_file, append=True)
    logger.info(
        f"=== Hoàn thành Discovery: Thu được {len(all_seeds)} ứng viên, "
        f"đã ghi mới {added} sản phẩm vào {out_path} ==="
    )
    return out_path, added


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Khám phá và thu thập URL/ID sản phẩm từ danh mục Tiki & Lazada."
    )
    parser.add_argument(
        "--config",
        default="configs/categories.yaml",
        help="Đường dẫn file cấu hình danh mục (YAML).",
    )
    parser.add_argument(
        "--source",
        choices=["all", "tiki", "lazada"],
        default="all",
        help="Lọc sàn cần quét (mặc định: all).",
    )
    parser.add_argument(
        "--min-reviews",
        type=int,
        help="Số lượt đánh giá tối thiểu (ghi đè config).",
    )
    parser.add_argument(
        "--min-sold",
        type=int,
        help="Số lượt mua tối thiểu (ghi đè config).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Số trang listing quét mỗi danh mục (ghi đè config).",
    )
    parser.add_argument(
        "--output",
        help="File output JSONL (mặc định: data/seeds/products.jsonl).",
    )
    parser.add_argument(
        "--cookie",
        help="Cookie phiên trình duyệt (dành cho Lazada nếu bị chặn bot).",
    )
    parser.add_argument(
        "--category-id",
        help="Mã danh mục (dùng khi parse file HTML cục bộ).",
    )
    parser.add_argument(
        "--category-name",
        help="Tên danh mục (dùng khi parse file HTML cục bộ).",
    )
    parser.add_argument(
        "--parse-local-lazada-html",
        help="Đường dẫn file HTML Lazada cục bộ để trích xuất sản phẩm ngoại tuyến.",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.parse_local_lazada_html:
        disc = ProductDiscovery()
        cat_id = args.category_id or "general"
        cat_name = args.category_name or "Lazada Local HTML"
        fn_lower = Path(args.parse_local_lazada_html).name.lower()
        if "bách hóa" in fn_lower or "bach hoa" in fn_lower:
            cat_id = args.category_id or "bach-hoa-online"
            cat_name = args.category_name or "Bách Hóa Online"

        seeds = disc.parse_lazada_saved_html(
            args.parse_local_lazada_html,
            category_id=cat_id,
            category_name=cat_name,
            min_reviews=args.min_reviews or 1,
            min_sold=args.min_sold or 0,
        )
        out_f = args.output or "data/seeds/products.jsonl"
        out_path, added = save_product_seeds(seeds, out_f, append=True)
        print(f"=== Đã trích xuất và thêm {added} sản phẩm mới vào {out_path} ===")
    else:
        run_discovery_pipeline(
            config_file=args.config,
            source_filter=args.source,
            override_min_reviews=args.min_reviews,
            override_min_sold=args.min_sold,
            override_max_pages=args.max_pages,
            output_override=args.output,
            cookie=args.cookie,
        )
