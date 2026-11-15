"""
Module Gemini Verifier: Thẩm định và Làm sạch các Đánh giá Nghi vấn (is_noisy == True)
======================================================================================
1. Đọc trực tiếp các đánh giá bị gắn cờ `is_noisy == True` từ tầng Silver (data/silver/reviews.parquet).
2. Sử dụng mô hình Google Gemini (mặc định gemini-1.5-flash) để phân tích ngữ nghĩa sâu,
   giải quyết dứt điểm mâu thuẫn giữa số sao và văn bản phức tạp.
3. Cập nhật nhãn chuẩn:
   - `label_final` = nhãn do Gemini thẩm định ('positive', 'neutral', 'negative')
   - `is_noisy` = False (đã giải phóng khỏi diện nghi vấn)
   - `is_trainable` = True (sạch hoàn toàn, sẵn sàng đưa vào huấn luyện NLP)
4. Đồng bộ in-place vào cả 2 tầng Silver và Gold mà không làm phân mảnh dữ liệu.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi cp1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo import được module từ thư mục gốc
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Tự động nạp biến môi trường từ .env
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GeminiVerifier")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

SYSTEM_INSTRUCTION = """Bạn là một chuyên gia ngôn ngữ học và chuyên gia phân tích cảm xúc đánh giá sản phẩm thương mại điện tử tiếng Việt (Tiki, Shopee, Lazada).
Nhiệm vụ của bạn là thẩm định nhãn cảm xúc thực sự cho các câu đánh giá phức tạp, phủ định kép, từ lóng hoặc mâu thuẫn với số sao.

Quy tắc phân loại:
- 'positive': Khen ngợi, hài lòng, hết lỗi, hết ngứa/gàu, đúng mô tả, chất lượng tốt, ủng hộ tiếp.
- 'negative': Chê bai, hỏng hóc, cong vênh, giao sai, lừa đảo, thất vọng, không dùng được.
- 'neutral': Hoàn toàn khách quan, thuần mô tả sự việc/thông số kỹ thuật, không khen không chê rõ ràng ("đã nhận hàng", "sách 200 trang bìa mềm").

Định dạng trả về: Mảng JSON thuần túy (Array of Objects), mỗi phần tử gồm:
[
  {"id": 0, "sentiment": "positive"|"neutral"|"negative", "reason": "Giải thích ngắn gọn"}
]
"""


def call_gemini_batch(
    reviews: List[Dict[str, Any]],
    api_key: str,
    model: str = "gemini-3.5-flash-lite",
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """Gửi một batch đánh giá qua Gemini REST API."""
    items_to_send = [{"id": r["idx"], "text": r["content_clean"]} for r in reviews]
    user_prompt = f"Phân loại cảm xúc cho danh sách đánh giá sau:\n{json.dumps(items_to_send, ensure_ascii=False, indent=2)}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{SYSTEM_INSTRUCTION}\n\n{user_prompt}"}],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }

    fallback_models = [model, "gemini-3-flash-preview", "gemini-3.5-flash"]

    for attempt in range(1, max_retries + 1):
        # Chọn model phù hợp nếu lần trước bị 503 hoặc timeout
        current_model = fallback_models[(attempt - 1) % len(fallback_models)]
        current_url = GEMINI_API_URL.format(model=current_model, api_key=api_key)

        try:
            resp = requests.post(
                current_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=25,
            )
            if resp.status_code == 200:
                data = resp.json()
                text_content = data["candidates"][0]["content"]["parts"][0]["text"]
                # Parse JSON array
                parsed = json.loads(text_content)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict) and "reviews" in parsed:
                    return parsed["reviews"]
                logger.warning("Định dạng JSON trả về không phải List: %s", text_content[:150])
            elif resp.status_code == 429:
                wait_sec = 6 * attempt
                logger.warning("Gemini API Rate Limit (429). Chờ %d giây trước khi thử lại...", wait_sec)
                time.sleep(wait_sec)
            elif resp.status_code == 503:
                logger.warning("Gemini Model %s quá tải (503). Chuyển sang model dự phòng...", current_model)
                time.sleep(3)
            else:
                logger.error("Lỗi HTTP %d từ Gemini API: %s", resp.status_code, resp.text[:200])
                time.sleep(2)
        except Exception as e:
            logger.error("Lỗi kết nối Gemini (lần thử %d/%d): %s", attempt, max_retries, e)
            time.sleep(3)

    return []


def run_gemini_verification(
    silver_path: str | Path = "data/silver/reviews.parquet",
    gold_path: str | Path = "data/gold/gold_reviews.parquet",
    api_key: str | None = None,
    model: str = "gemini-3.5-flash-lite",
    sample_size: int = 50,
    batch_size: int = 10,
    dry_run: bool = False,
) -> None:
    s_path = Path(silver_path)
    g_path = Path(gold_path)

    if not s_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file Silver: {s_path.resolve()}")

    # 1. Đọc dữ liệu Silver
    logger.info("Đang đọc dữ liệu Silver: %s", s_path)
    df = pd.read_parquet(s_path)

    # 2. Lọc các bản ghi is_noisy == True
    if "is_noisy" not in df.columns:
        raise ValueError("Cột 'is_noisy' chưa tồn tại. Vui lòng chạy sync_all_labels.py trước.")

    noisy_mask = df["is_noisy"] == True
    total_noisy = int(noisy_mask.sum())
    logger.info("Tổng số bản ghi bị gắn cờ is_noisy == True: %d", total_noisy)

    if total_noisy == 0:
        logger.info("🎉 Tuyệt vời! Không còn bản ghi nào bị gắn cờ is_noisy.")
        return

    # Xác định số lượng cần thẩm định
    if sample_size > 0 and sample_size < total_noisy:
        target_indices = df[noisy_mask].head(sample_size).index
        logger.info("Chế độ lấy mẫu thẩm định: %d / %d bản ghi", sample_size, total_noisy)
    else:
        target_indices = df[noisy_mask].index
        logger.info("Chế độ thẩm định toàn bộ: %d bản ghi", len(target_indices))

    # 3. Kiểm tra API Key nếu không phải dry-run
    final_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not dry_run and not final_api_key:
        logger.error(
            "❌ Không tìm thấy GEMINI_API_KEY!\n"
            "Vui lòng thiết lập biến môi trường GEMINI_API_KEY trong file .env hoặc truyền tham số --api-key.\n"
            "Ví dụ trong .env: GEMINI_API_KEY=AIzaSy...\n"
            "Hoặc chạy thử nghiệm cấu trúc với: python src/spark_jobs/gemini_verifier.py --dry-run"
        )
        return

    if dry_run:
        logger.info("⚙️ Đang chạy ở chế độ DRY-RUN (không gọi API thực tế)...")
        print("\n" + "=" * 75)
        print(" DANH SÁCH MẪU NGHI VẤN is_noisy == True CHỜ GEMINI THẨM ĐỊNH:")
        print("=" * 75)
        sample_view = df.loc[target_indices, ["rating", "label_weak", "label_text", "confidence_score", "content_clean"]].head(10)
        for _, row in sample_view.iterrows():
            print(f"- Sao: {row['rating']}★ ({row['label_weak']}) | PhoBERT: {row['label_text']} (conf: {row['confidence_score']:.2f})")
            print(f"  Nội dung: {row['content_clean']}\n")
        print(f"✅ Đã xác nhận: Có {len(target_indices)} dòng sẵn sàng để Gemini tiếp nhận trực tiếp.")
        return

    # 4. Thực thi gọi Gemini theo batch
    logger.info("Bắt đầu gửi %d bản ghi sang Gemini (%s) theo batch %d...", len(target_indices), model, batch_size)
    t0 = time.time()
    verified_count = 0

    indices_list = list(target_indices)
    for i in range(0, len(indices_list), batch_size):
        chunk_indices = indices_list[i : i + batch_size]
        batch_items = []
        for local_idx, real_idx in enumerate(chunk_indices):
            batch_items.append({
                "idx": local_idx,
                "real_idx": real_idx,
                "content_clean": str(df.loc[real_idx, "content_clean"]),
            })

        logger.info("Đang xử lý batch %d - %d / %d...", i + 1, min(i + batch_size, len(indices_list)), len(indices_list))
        gemini_results = call_gemini_batch(batch_items, api_key=final_api_key, model=model)

        # Cập nhật kết quả
        for res in gemini_results:
            local_id = res.get("id")
            sentiment = str(res.get("sentiment", "")).lower().strip()
            if sentiment in {"positive", "neutral", "negative"} and 0 <= local_id < len(batch_items):
                r_idx = batch_items[local_id]["real_idx"]
                df.loc[r_idx, "label_final"] = sentiment
                df.loc[r_idx, "is_noisy"] = False
                df.loc[r_idx, "is_trainable"] = True
                verified_count += 1

        # Tự động lưu checkpoint định kỳ mỗi 5 batch để bảo toàn tiến độ
        batch_num = (i // batch_size) + 1
        if batch_num % 5 == 0 and verified_count > 0:
            df.to_parquet(s_path, index=False, compression="snappy")
            logger.info("💾 [Auto-Save Checkpoint] Đã lưu an toàn %d bản ghi đã thẩm định vào tầng Silver!", verified_count)

        # Nghỉ 2.0s để tránh dồn dập vượt quá RPM gói Free Tier
        time.sleep(2.0)

    elapsed = time.time() - t0
    logger.info("Hoàn tất thẩm định %d / %d bản ghi trong %.2f giây!", verified_count, len(target_indices), elapsed)

    # 5. Lưu lại dữ liệu Silver và Gold in-place
    logger.info("Đang ghi đè tầng Silver: %s...", s_path)
    df.to_parquet(s_path, index=False, compression="snappy")
    logger.info("✅ Tầng Silver đã cập nhật hoàn tất!")

    if g_path.exists():
        logger.info("Đang đồng bộ sang tầng Gold: %s...", g_path)
        df_gold = pd.read_parquet(g_path)
        # Đồng bộ nhãn các dòng đã thẩm định sang Gold
        common_cols = ["label_final", "is_noisy", "is_trainable"]
        for c in common_cols:
            df_gold.loc[target_indices, c] = df.loc[target_indices, c]
        df_gold.to_parquet(g_path, index=False, compression="snappy")
        logger.info("✅ Tầng Gold đã đồng bộ hoàn tất!")

    # 6. Báo cáo kết quả
    remaining_noisy = int((df["is_noisy"] == True).sum())
    total_trainable = int((df["is_trainable"] == True).sum())
    print("\n" + "=" * 75)
    print(" BÁO CÁO KẾT QUẢ THẨM ĐỊNH BẰNG GEMINI")
    print("=" * 75)
    print(f"- Số bản ghi đã được thẩm định & làm sạch: {verified_count:,}")
    print(f"- Số bản ghi còn lại bị gắn cờ is_noisy:     {remaining_noisy:,}")
    print(f"- Tổng số bản ghi đạt chuẩn train NLP:       {total_trainable:,}")
    print("=" * 75 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Thẩm định các đánh giá is_noisy == True bằng Gemini API.")
    parser.add_argument("--silver-path", default="data/silver/reviews.parquet", help="Đường dẫn Silver parquet")
    parser.add_argument("--gold-path", default="data/gold/gold_reviews.parquet", help="Đường dẫn Gold parquet")
    parser.add_argument("--api-key", default=None, help="Google Gemini API Key (mặc định lấy từ biến môi trường GEMINI_API_KEY)")
    parser.add_argument("--model", default="gemini-3.5-flash-lite", help="Model Gemini (mặc định gemini-3.5-flash-lite)")
    parser.add_argument("--sample-size", type=int, default=50, help="Số lượng mẫu cần thẩm định (mặc định 50, 0 để chạy hết)")
    parser.add_argument("--batch-size", type=int, default=10, help="Số lượng review mỗi lượt gọi API")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử nghiệm kiểm tra mẫu dữ liệu, không gọi API")
    args = parser.parse_args()

    run_gemini_verification(
        silver_path=args.silver_path,
        gold_path=args.gold_path,
        api_key=args.api_key,
        model=args.model,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
