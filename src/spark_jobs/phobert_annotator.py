"""
Module PhoBERT Annotator: Gán nhãn ngữ nghĩa & Cập nhật label_final qua Ma trận quyết định
========================================================================================
1. Tải mô hình PhoBERT tiền huấn luyện chuyên sâu cho cảm xúc tiếng Việt:
   `wonrax/phobert-base-vietnamese-sentiment`.
2. Chạy suy luận ngữ nghĩa (Teacher Model) trên tập dữ liệu đánh giá:
   - Sinh nhãn `label_text`: positive, neutral, negative
   - Sinh điểm độ tin cậy `confidence_score` (Softmax probability)
3. Áp dụng Ma trận quyết định (Decision Matrix) theo chiến lược Text-First:
   - Sửa sai cho các đánh giá mâu thuẫn giữa sao và lời nhận xét (confidence >= 0.50).
   - Quy các câu phân vân, cảm xúc pha trộn (confidence < 0.50) về `neutral`.
   - Giữ nguyên số sao (label_weak) cho các review không có nội dung text.
4. Cập nhật trực tiếp vào tầng Silver (data/silver/reviews.parquet).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

# Tối ưu hóa số luồng CPU cho PyTorch
cpu_threads = max(1, (os.cpu_count() or 4) - 1)
torch.set_num_threads(cpu_threads)

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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PhoBERT_Annotator")

# Từ khóa nghi vấn mâu thuẫn trong các review 4-5 sao
SUSPICIOUS_KEYWORDS = [
    "không", "k ", "ko ", "chưa", "hỏng", "lỗi", "chậm", "tệ", "kém",
    "thất vọng", "móp", "hạn sử dụng", "hsd", "rách", "thiếu", "đắt",
    "nhỏ", "bình thường", "tạm", "nhưng", "tuy nhiên", "mất", "bể",
    "vỡ", "trầy", "dơ", "hôi", "giả", "nhái", "lừa", "sai", "đổi", "trả"
]


def load_phobert_pipeline(model_name: str = "wonrax/phobert-base-vietnamese-sentiment") -> Any:
    """Khởi tạo pipeline PhoBERT."""
    logger.info("Đang nạp mô hình PhoBERT: %s...", model_name)
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (CUDA)" if device == 0 else "CPU"
    logger.info("Sử dụng thiết bị tính toán: %s", device_name)

    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device,
        truncation=True,
        max_length=64,
    )
    return classifier


def map_phobert_label(raw_label: str) -> str:
    """Ánh xạ nhãn PhoBERT (NEG, NEU, POS) sang (negative, neutral, positive)."""
    clean_label = str(raw_label).strip().upper()
    if "NEG" in clean_label:
        return "negative"
    if "NEU" in clean_label:
        return "neutral"
    if "POS" in clean_label:
        return "positive"
    return "neutral"


def select_target_reviews(
    df: pd.DataFrame,
    sample_size: int = 3000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Lọc các review mục tiêu cần gán nhãn ngữ nghĩa bằng PhoBERT.
    Ưu tiên tập trung vào các trường hợp có nguy cơ mâu thuẫn cao:
    - 100% review 1-2-3 sao có text
    - Các review 4-5 sao chứa từ khóa nghi vấn (phàn nàn, mâu thuẫn)
    - Kèm mẫu review 4-5 sao chuẩn
    """
    usable = df[df["is_text_usable"] == True].copy()
    logger.info("Tổng số review có text hợp lệ: %d", len(usable))

    # Nhóm 1: Tất cả 1, 2, 3 sao
    low_ratings = usable[usable["rating"].isin([1, 2, 3])]

    # Nhóm 2: 4, 5 sao có chứa từ khóa nghi vấn
    pattern = "|".join(SUSPICIOUS_KEYWORDS)
    high_ratings = usable[usable["rating"].isin([4, 5])]
    suspicious = high_ratings[high_ratings["content_clean"].str.lower().str.contains(pattern, regex=True, na=False)]

    # Nhóm 3: Mẫu chuẩn 4, 5 sao không nghi vấn
    clean_high = high_ratings[~high_ratings.index.isin(suspicious.index)]
    
    # Kết hợp các nhóm ưu tiên
    priority_pool = pd.concat([low_ratings, suspicious]).drop_duplicates(subset=["review_id"])
    logger.info("Số mẫu nghi vấn mâu thuẫn cao được chọn ưu tiên: %d", len(priority_pool))

    if len(priority_pool) >= sample_size:
        target = priority_pool.sample(n=sample_size, random_state=random_state)
    else:
        remaining_needed = sample_size - len(priority_pool)
        high_sample = clean_high.sample(n=min(remaining_needed, len(clean_high)), random_state=random_state)
        target = pd.concat([priority_pool, high_sample]).drop_duplicates(subset=["review_id"])

    logger.info("Tập mục tiêu PhoBERT thẩm định: %d bản ghi", len(target))
    return target


def run_phobert_annotation(
    silver_path: str | Path = "data/silver/reviews.parquet",
    sample_size: int = 2500,
    batch_size: int = 32,
    output_silver_path: str | Path | None = None,
) -> pd.DataFrame:
    """Thực thi toàn bộ quy trình gán nhãn PhoBERT và ma trận quyết định."""
    s_path = Path(silver_path)
    if not s_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file Silver: {s_path.resolve()}")

    logger.info("Đang đọc dữ liệu Silver: %s", s_path)
    df = pd.read_parquet(s_path)

    # 1. Chọn tập review cần thẩm định
    target_df = select_target_reviews(df, sample_size=sample_size)
    target_indices = target_df.index
    target_texts = target_df["content_clean"].tolist()

    # 2. Nạp mô hình PhoBERT
    classifier = load_phobert_pipeline()

    # 3. Chạy suy luận theo batch
    logger.info("Bắt đầu chạy suy luận PhoBERT trên %d mẫu (batch_size=%d)...", len(target_texts), batch_size)
    t0 = time.time()
    
    phobert_labels = []
    phobert_scores = []

    with torch.inference_mode():
        for i in tqdm(range(0, len(target_texts), batch_size), desc="PhoBERT Annotating"):
            batch = target_texts[i : i + batch_size]
            results = classifier(batch, batch_size=batch_size, truncation=True, max_length=64)
            for r in results:
                mapped_lbl = map_phobert_label(r["label"])
                score = round(float(r["score"]), 4)
                phobert_labels.append(mapped_lbl)
                phobert_scores.append(score)

    elapsed = time.time() - t0
    logger.info("Hoàn tất PhoBERT trong %.2f giây (Tốc độ: %.1f câu/giây)", elapsed, len(target_texts) / max(elapsed, 0.001))

    # 4. Gán kết quả vào DataFrame
    # Đảm bảo các cột có kiểu dữ liệu phù hợp
    if "label_text" not in df.columns or df["label_text"].dtype != "object":
        df["label_text"] = df["label_text"].astype("object")
    if "confidence_score" not in df.columns:
        df["confidence_score"] = np.nan

    df.loc[target_indices, "label_text"] = phobert_labels
    df.loc[target_indices, "confidence_score"] = phobert_scores

    # 5. Áp dụng Ma trận quyết định Text-First để sinh label_final
    logger.info("Áp dụng Ma trận quyết định Text-First để cập nhật label_final...")
    
    # Mặc định: label_final = label_weak
    df["label_final"] = df["label_weak"]
    if "is_noisy" not in df.columns:
        df["is_noisy"] = False
    if "is_trainable" not in df.columns:
        df["is_trainable"] = df["is_text_usable"] == True

    # Có label_text từ PhoBERT:
    has_text_label = df["label_text"].notna()
    
    # Nhóm 1: Tự tin cao (>= 0.60) -> Ưu tiên tuyệt đối label_text (sửa sai cho số sao)
    high_conf = has_text_label & (df["confidence_score"] >= 0.60)
    df.loc[high_conf, "label_final"] = df.loc[high_conf, "label_text"]
    df.loc[high_conf, "is_noisy"] = False
    df.loc[high_conf, "is_trainable"] = True

    # Nhóm 2: Tự tin thấp (< 0.60) nhưng đồng thuận (label_weak == label_text) -> An toàn
    low_conf_agree = has_text_label & (df["confidence_score"] < 0.60) & (df["label_weak"] == df["label_text"])
    df.loc[low_conf_agree, "label_final"] = df.loc[low_conf_agree, "label_weak"]
    df.loc[low_conf_agree, "is_noisy"] = False
    df.loc[low_conf_agree, "is_trainable"] = True

    # Nhóm 3: Tự tin thấp (< 0.60) và mâu thuẫn (label_weak != label_text) -> Đánh dấu nhiễu để Gemini thẩm định
    low_conf_conflict = has_text_label & (df["confidence_score"] < 0.60) & (df["label_weak"] != df["label_text"])
    df.loc[low_conf_conflict, "label_final"] = df.loc[low_conf_conflict, "label_weak"]
    df.loc[low_conf_conflict, "is_noisy"] = True
    df.loc[low_conf_conflict, "is_trainable"] = False

    # Cập nhật lại cờ is_conflict
    df["is_conflict"] = has_text_label & (df["label_weak"] != df["label_text"])
    conflict_count = int(df["is_conflict"].sum())
    noisy_count = int(df["is_noisy"].sum())
    
    logger.info("Phát hiện %d trường hợp mâu thuẫn giữa số sao và ngữ nghĩa PhoBERT", conflict_count)
    logger.info("Gắn cờ is_noisy = True cho %d trường hợp nghi vấn chờ Gemini thẩm định", noisy_count)

    # 6. Thống kê kết quả
    print("\n" + "=" * 75)
    print(" BÁO CÁO KẾT QUẢ GÁN NHÃN NGỮ NGHĨA PHOBERT & MA TRẬN QUYẾT ĐỊNH")
    print("=" * 75)
    print("1. Phân bố nhãn ngữ nghĩa PhoBERT (label_text trên tập thẩm định):")
    print(df.loc[target_indices, "label_text"].value_counts().to_string())
    print("\n2. Phân bố điểm tự tin trung bình (confidence_score):")
    print(df.loc[target_indices, "confidence_score"].describe().to_string())
    print("\n3. Số lượng mâu thuẫn sao vs chữ đã được sửa:")
    print(f" - Tổng số ca xung đột: {conflict_count} / {len(target_df)} ({conflict_count / len(target_df) * 100:.2f}%)")
    print("\n4. Phân bố nhãn huấn luyện cuối cùng (label_final) toàn bộ dataset:")
    print(df["label_final"].value_counts().to_string())
    print("=" * 75 + "\n")

    # 7. Ghi đè vào file Silver Parquet
    out_path = Path(output_silver_path) if output_silver_path else s_path
    logger.info("Đang lưu kết quả cập nhật vào: %s", out_path)
    df.to_parquet(out_path, index=False, compression="snappy")
    logger.info("✅ Cập nhật thành công tầng Silver với nhãn PhoBERT và label_final sạch!")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Gán nhãn ngữ nghĩa bằng PhoBERT và ma trận quyết định.")
    parser.add_argument("--silver-path", default="data/silver/reviews.parquet", help="Đường dẫn file Silver parquet")
    parser.add_argument("--sample-size", type=int, default=2500, help="Số lượng mẫu PhoBERT thẩm định")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước batch suy luận PhoBERT")
    parser.add_argument("--output", default=None, help="Đường dẫn lưu file mới (mặc định ghi đè Silver)")
    args = parser.parse_args()

    run_phobert_annotation(
        silver_path=args.silver_path,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        output_silver_path=args.output,
    )


if __name__ == "__main__":
    main()
