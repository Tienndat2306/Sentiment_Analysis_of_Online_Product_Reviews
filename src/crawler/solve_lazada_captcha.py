import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playwright.sync_api import sync_playwright
from src.crawler.update_lazada_prices import update_lazada_prices_in_seeds


def capture_cookie_and_update(test_product_id: str = "2017340529", max_wait_seconds: int = 45):
    target_url = f"https://www.lazada.vn/products/pdp-i{test_product_id}.html"
    print(f"\n[Lazada WAF Solver] Đang mở trình duyệt để xác thực phiên tại: {target_url}")
    print("[Hướng dẫn] Nếu có thanh trượt CAPTCHA xuất hiện trên màn hình, bạn hãy kéo thanh trượt từ trái sang phải.")
    print(f"[Đang chờ] Hệ thống sẽ tự động phát hiện khi bạn hoàn tất (tối đa {max_wait_seconds} giây)...\n")

    captured_cookie = None

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="vi-VN",
        )
        page = context.new_page()

        try:
            page.goto(target_url, timeout=35000, wait_until="domcontentloaded")
        except Exception as e:
            print(f"Lỗi truy cập trang: {e}")

        start = time.time()
        while time.time() - start < max_wait_seconds:
            curr_url = page.url
            cookies = context.cookies()
            has_x5sec = any(c.get("name") == "x5sec" for c in cookies)

            if ("punish" not in curr_url and ("pdp-i" in curr_url or "products" in curr_url)) or has_x5sec:
                print("\n[Thành công] Đã vượt qua kiểm tra bảo mật của Lazada!")
                # Lấy toàn bộ cookie phiên
                cookie_parts = [f"{c['name']}={c['value']}" for c in cookies]
                captured_cookie = "; ".join(cookie_parts)
                page.wait_for_timeout(1000)
                break

            time.sleep(1)

        browser.close()

    if not captured_cookie:
        print("[Thất bại] Chưa nhận được cookie hợp lệ trong thời gian chờ.")
        return False

    # Lưu cookie vào file .env để tái sử dụng
    env_file = Path(".env")
    env_content = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    if "LAZADA_COOKIE=" in env_content:
        import re
        env_content = re.sub(r"LAZADA_COOKIE=.*", f"LAZADA_COOKIE={captured_cookie}", env_content)
    else:
        env_content += f"\nLAZADA_COOKIE={captured_cookie}\n"
    env_file.write_text(env_content.strip() + "\n", encoding="utf-8")
    print(f"[Đã lưu] Cookie phiên đã được lưu vào file .env để tái sử dụng lâu dài.")

    # Tiến hành cập nhật giá cho các sản phẩm còn lại
    print("\n[Bắt đầu] Đang tiến hành cập nhật giá cho các sản phẩm Lazada còn lại...")
    res = update_lazada_prices_in_seeds(
        seeds_file="data/seeds/products.jsonl",
        delay_seconds=0.6,
        timeout=10,
        only_missing=True,
        cookie=captured_cookie,
    )
    print(f"\n[Hoàn tất] Kết quả cập nhật: {res}")
    return True


if __name__ == "__main__":
    capture_cookie_and_update()
