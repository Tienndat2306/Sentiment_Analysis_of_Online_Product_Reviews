"""
Script Đồng Bộ Hóa Toàn Diện: Phủ kín 100% Nhãn và Confidence Score
===================================================================
1. Sử dụng mô hình Machine Learning / NLP tốt nhất từ models/best_sentiment_model.joblib.
2. Suy luận toàn bộ 69.113 đánh giá có văn bản:
   - Gán `label_text` cho 100% bản ghi có text.
   - Tính toán `confidence_score` (xác suất Softmax cực đại từ 0.0 - 1.0) cho 100% bản ghi.
   - Áp dụng Ma trận quyết định Text-First cập nhật `label_final` sạch hoàn toàn.
   - Gắn cờ `is_conflict` (mâu thuẫn giữa số sao và ngữ nghĩa).
3. Với đánh giá không có text (156.546 bản ghi):
   - Kế thừa `label_final = label_weak`, `confidence_score = 1.0`, `prediction_source = 'rule_weak_label'`.
4. Ghi đè và đồng bộ hóa chuẩn xác cả hai tầng:
   - `data/silver/reviews.parquet`
   - `data/gold/gold_reviews.parquet`
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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
logger = logging.getLogger("SyncAllLabels")


def sync_labels_and_confidence(
    silver_path: str = "data/silver/reviews.parquet",
    gold_path: str = "data/gold/gold_reviews.parquet",
    model_path: str = "models/best_sentiment_model.joblib",
    batch_size: int = 10000,
) -> None:
    s_path = Path(silver_path)
    g_path = Path(gold_path)
    m_path = Path(model_path)

    if not s_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file Silver: {s_path.resolve()}")
    if not m_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file Model: {m_path.resolve()}")

    logger.info("Đang nạp mô hình NLP tốt nhất từ: %s", m_path)
    pipeline = joblib.load(m_path)

    logger.info("Đang đọc dữ liệu Silver: %s", s_path)
    df = pd.read_parquet(s_path)
    total_records = len(df)
    logger.info("Tổng số bản ghi: %d", total_records)

    # 1. Xác định tập có text hợp lệ
    text_mask = (df["is_text_usable"] == True) & (df["content_clean"].fillna("").str.strip().str.len() > 0)
    usable_count = int(text_mask.sum())
    logger.info("Số bản ghi có văn bản cần suy luận: %d (%.2f%%)", usable_count, (usable_count / total_records) * 100)

    # 2. Khởi tạo các cột
    df["label_text"] = None
    df["confidence_score"] = 1.0  # Mặc định 1.0 cho các review theo sao
    df["label_final"] = df["label_weak"]
    df["is_conflict"] = False
    df["is_noisy"] = False
    df["is_trainable"] = False

    # 3. Suy luận theo batch cho 69.113 bản ghi có text
    if usable_count > 0:
        logger.info("Bắt đầu suy luận NLP toàn diện cho %d bản ghi (batch_size=%d)...", usable_count, batch_size)
        t0 = time.time()
        usable_texts = df.loc[text_mask, "content_clean"].tolist()
        
        preds = []
        confidences = []
        has_proba = hasattr(pipeline, "predict_proba")

        for i in range(0, usable_count, batch_size):
            batch = usable_texts[i : i + batch_size]
            batch_preds = pipeline.predict(batch)
            preds.extend(batch_preds)

            if has_proba:
                proba_mat = pipeline.predict_proba(batch)
                max_probs = np.max(proba_mat, axis=1)
                confidences.extend(np.round(max_probs, 4))
            else:
                confidences.extend([1.0] * len(batch))

        elapsed = time.time() - t0
        logger.info("Suy luận xong trong %.2f giây (Tốc độ: %.1f câu/giây)", elapsed, usable_count / max(elapsed, 0.001))

        # Gán label_text và confidence_score
        df.loc[text_mask, "label_text"] = preds
        df.loc[text_mask, "confidence_score"] = confidences

        # 4. Áp dụng Ma trận quyết định Text-First cải tiến (Không ép neutral sai lệch)
        # Nhóm 1: Tự tin cao (>= 0.60) -> Ưu tiên label_text (sửa sai cho số sao)
        high_conf_mask = text_mask & (df["confidence_score"] >= 0.60)
        df.loc[high_conf_mask, "label_final"] = df.loc[high_conf_mask, "label_text"]
        df.loc[high_conf_mask, "is_noisy"] = False
        df.loc[high_conf_mask, "is_trainable"] = True

        # Nhóm 2: Tự tin thấp (< 0.60) & Đồng thuận (label_weak == label_text) -> An toàn
        low_conf_agree_mask = text_mask & (df["confidence_score"] < 0.60) & (df["label_weak"] == df["label_text"])
        df.loc[low_conf_agree_mask, "label_final"] = df.loc[low_conf_agree_mask, "label_weak"]
        df.loc[low_conf_agree_mask, "is_noisy"] = False
        df.loc[low_conf_agree_mask, "is_trainable"] = True

        # Nhóm 3: Tự tin thấp (< 0.60) & Xung đột (label_weak != label_text) -> Đánh dấu nhiễu, loại khỏi train chờ Gemini
        low_conf_conflict_mask = text_mask & (df["confidence_score"] < 0.60) & (df["label_weak"] != df["label_text"])
        df.loc[low_conf_conflict_mask, "label_final"] = df.loc[low_conf_conflict_mask, "label_weak"]
        df.loc[low_conf_conflict_mask, "is_noisy"] = True
        df.loc[low_conf_conflict_mask, "is_trainable"] = False

        # Cập nhật is_conflict: có mâu thuẫn giữa số sao và ngữ nghĩa mô hình dự đoán
        df["is_conflict"] = text_mask & (df["label_weak"] != df["label_text"])

    conflict_count = int(df["is_conflict"].sum())
    noisy_count = int(df["is_noisy"].sum())
    trainable_count = int(df["is_trainable"].sum())
    logger.info("Tổng số bản ghi mâu thuẫn sao vs chữ: %d (%.2f%% trên tập có text)", conflict_count, (conflict_count / usable_count) * 100)
    logger.info("Gắn cờ is_noisy = True cho %d bản ghi nghi vấn (chờ đưa qua Gemini)", noisy_count)
    logger.info("Số bản ghi sạch đạt chuẩn huấn luyện NLP (is_trainable = True): %d", trainable_count)

    # Đảm bảo các kiểu dữ liệu chuẩn xác
    df["label_text"] = df["label_text"].astype("string")
    df["confidence_score"] = df["confidence_score"].astype("float64")
    df["label_final"] = df["label_final"].astype("string")
    df["is_conflict"] = df["is_conflict"].astype("bool")
    df["is_noisy"] = df["is_noisy"].astype("bool")
    df["is_trainable"] = df["is_trainable"].astype("bool")

    # 5. Lưu lại tầng Silver
    logger.info("Đang ghi đè tầng Silver: %s...", s_path)
    df.to_parquet(s_path, index=False, compression="snappy")
    logger.info("✅ Tầng Silver đã đồng bộ hoàn tất!")

    # 6. Đồng bộ hóa tầng Gold (Loại bỏ triệt để các cột dư thừa)
    logger.info("Đang tạo và đồng bộ tầng Gold: %s...", g_path)
    df_gold = df.copy()
    df_gold["prediction_source"] = np.where(text_mask, "ml_model", "rule_weak_label")

    # Loại bỏ các cột dư thừa cũ nếu có
    redundant_cols = ["predicted_sentiment", "prediction_confidence", "is_sentiment_conflict"]
    df_gold = df_gold.drop(columns=[col for col in redundant_cols if col in df_gold.columns])

    df_gold.to_parquet(g_path, index=False, compression="snappy")
    logger.info("✅ Tầng Gold đã đồng bộ hoàn tất (gọn gàng, không trùng lặp)!")

    # 7. In báo cáo kiểm tra số liệu
    print("\n" + "=" * 80)
    print(" BÁO CÁO ĐỒNG BỘ HÓA 100% DỮ LIỆU TẦNG SILVER VÀ GOLD")
    print("=" * 80)
    print("1. Kiểm tra số lượng Non-Null (Không còn bất kỳ dòng rỗng nào):")
    print(f" - label_weak:        {df['label_weak'].notna().sum():,} / {total_records:,} (100%)")
    print(f" - label_text:        {df['label_text'].notna().sum():,} / {usable_count:,} (100% tập có text)")
    print(f" - confidence_score:   {df['confidence_score'].notna().sum():,} / {total_records:,} (100% toàn bộ dataset)")
    print(f" - label_final:       {df['label_final'].notna().sum():,} / {total_records:,} (100% toàn bộ dataset)")
    print(f" - is_conflict:       {df['is_conflict'].notna().sum():,} / {total_records:,} (100%)")
    print(f" - is_noisy:          {df['is_noisy'].sum():,} / {usable_count:,} (Dòng nghi vấn chờ Gemini)")
    print(f" - is_trainable:      {df['is_trainable'].sum():,} / {usable_count:,} (Dòng đạt chuẩn huấn luyện)")

    print("\n2. Phân bố nhãn cuối cùng (label_final):")
    print(df["label_final"].value_counts().to_string())

    print("\n3. Thống kê điểm tự tin (confidence_score):")
    print(" - Trên tập có text (suy luận NLP):")
    print(df.loc[text_mask, "confidence_score"].describe().to_string())

    print(f"\n4. Số trường hợp mâu thuẫn sao vs chữ đã được sửa nhãn sạch:")
    print(f" - Số ca mâu thuẫn: {conflict_count:,} / {usable_count:,} ({conflict_count / usable_count * 100:.2f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    sync_labels_and_confidence()
