"""
Inference Pipeline: Suy luận Cảm xúc & Xuất bản Tầng Gold
=========================================================
Thực hiện:
1. Đọc dữ liệu từ tầng Silver (data/silver/reviews.parquet).
2. Tải active model từ Model Registry (models/best_sentiment_model.joblib).
3. Dự đoán nhãn cảm xúc (predicted_sentiment) và độ tin cậy (prediction_confidence).
   - Với đánh giá có văn bản (is_text_usable == True): Suy luận bằng mô hình ML.
   - Với đánh giá không có văn bản hoặc quá ngắn: Kế thừa nhãn yếu từ số sao (label_weak).
4. Xuất dữ liệu tầng Gold ra Parquet (data/gold/gold_reviews.parquet)
   và tùy chọn đồng bộ vào MongoDB (collection: gold_reviews).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

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

import joblib
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("InferencePipeline")


# =============================================================================
# Danh sách cột chuẩn mực (Contract Schema) của Tầng Gold (32 cột)
# Tách bạch hoàn toàn dữ liệu kinh doanh/AI khỏi cờ kỹ thuật cào (như in_seed)
# =============================================================================
GOLD_COLUMNS = [
    # 1. Định danh & thông tin thực thể
    "review_id",
    "product_id",
    "product_name",
    "customer_id",
    "customer_name",
    # 2. Đánh giá số sao & tính hợp lệ
    "rating",
    "is_valid_rating",
    # 3. Tiêu đề & nội dung đánh giá đã làm sạch
    "title",
    "title_is_template",
    "content",
    "content_clean",
    "has_content",
    "word_count",
    "is_text_usable",
    "reject_reason",
    # 4. Nhãn cảm xúc, độ tin cậy & cờ mâu thuẫn AI
    "label_weak",
    "label_text",
    "confidence_score",
    "label_final",
    "is_conflict",
    # 5. Metadata tương tác & nguồn dữ liệu
    "is_photo",
    "thank_count",
    "comment_count",
    "created_at",
    "seller_name",
    "category_id",
    "category_name",
    "source",
    "crawled_at",
    # 6. Cờ chất lượng & truy xuất nguồn gốc nhãn
    "is_noisy",
    "is_trainable",
    "prediction_source",
]


def load_settings(config_path: str = "configs/settings.yaml") -> dict[str, Any]:
    cfg_file = Path(config_path)
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def run_inference(
    silver_path: str | Path = "data/silver/reviews.parquet",
    model_path: str | Path = "models/best_sentiment_model.joblib",
    output_gold_path: str | Path = "data/gold/gold_reviews.parquet",
    teacher_cache_path: str | Path = "data/silver/teacher_annotations.parquet",
    force_student: bool = False,
    batch_size: int = 10000,
    sync_to_mongo: bool = False,
    mongo_uri: str | None = None,
) -> pd.DataFrame:
    """Chạy toàn bộ luồng suy luận và tạo bảng dữ liệu tầng Gold.
    
    Bảo toàn nhãn Teacher (PhoBERT + Gemini) từ teacher_annotations.parquet cho các đánh giá đã thẩm định,
    và chỉ chạy suy luận bằng mô hình Student (LinearSVC) cho các đánh giá mới chưa có nhãn Teacher
    (hoặc khi bật cờ force_student=True).
    """
    s_path = Path(silver_path)
    m_path = Path(model_path)
    g_path = Path(output_gold_path)
    t_path = Path(teacher_cache_path)

    if not s_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file Silver tại {s_path.resolve()}")
    if not m_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file Model tại {m_path.resolve()}")

    logger.info("Đang nạp mô hình từ: %s", m_path)
    pipeline = joblib.load(m_path)

    # Đọc dữ liệu Silver
    logger.info("Đang nạp dữ liệu Silver: %s", s_path)
    df = pd.read_parquet(s_path)
    logger.info("Tổng số bản ghi Silver cần xử lý: %d", len(df))

    # Khởi tạo/chuẩn hóa các cột AI suy luận cho tầng Gold
    # Lưu ý: các đánh giá không có văn bản (rule_weak_label) để confidence_score = NaN và is_trainable = False
    df["label_text"] = None
    df["label_final"] = df["label_weak"]
    df["confidence_score"] = np.nan
    df["prediction_source"] = "rule_weak_label"
    df["is_conflict"] = False
    df["is_noisy"] = False
    df["is_trainable"] = False

    # Lọc các dòng có text hợp lệ để suy luận bằng Machine Learning
    text_mask = (df["is_text_usable"] == True) & (df["content_clean"].fillna("").str.strip().str.len() > 0)
    usable_count = int(text_mask.sum())
    logger.info("Số bản ghi đủ điều kiện suy luận NLP: %d (%.2f%%)", usable_count, (usable_count / len(df)) * 100)

    # Ưu tiên kế thừa nhãn Teacher (PhoBERT + Gemini) nếu đã có trong kho lưu trữ
    unannotated_mask = text_mask.copy()
    if not force_student and t_path.exists():
        logger.info("Đang nạp kho nhãn Teacher (PhoBERT + Gemini) từ: %s", t_path)
        df_teacher = pd.read_parquet(t_path)
        df_teacher = df_teacher.drop_duplicates(subset=["source", "review_id"], keep="last")
        merged = df[["source", "review_id"]].merge(
            df_teacher, on=["source", "review_id"], how="left", indicator=True
        )
        matched_mask = text_mask & (merged["_merge"] == "both").values
        matched_count = int(matched_mask.sum())
        if matched_count > 0:
            for col in [
                "label_text",
                "confidence_score",
                "label_final",
                "prediction_source",
            ]:
                if col in merged.columns:
                    df.loc[matched_mask, col] = merged.loc[matched_mask, col].values
            for bcol in ["is_conflict", "is_noisy", "is_trainable"]:
                if bcol in merged.columns:
                    df.loc[matched_mask, bcol] = merged.loc[matched_mask, bcol].fillna(False).astype(bool).values
            logger.info("Đã khôi phục %d nhãn Teacher (PhoBERT + Gemini) đã thẩm định.", matched_count)
        unannotated_mask = text_mask & ~(merged["_merge"] == "both").values

    student_target_count = int(unannotated_mask.sum())
    if student_target_count > 0:
        usable_texts = df.loc[unannotated_mask, "content_clean"].tolist()

        logger.info("Đang chạy suy luận Student ML cho %d bản ghi mới (batch_size=%d)...", student_target_count, batch_size)
        preds = []
        confidences = []

        has_proba = hasattr(pipeline, "predict_proba")

        for i in range(0, student_target_count, batch_size):
            batch = usable_texts[i : i + batch_size]
            batch_preds = pipeline.predict(batch)
            preds.extend(batch_preds)

            if has_proba:
                proba_matrix = pipeline.predict_proba(batch)
                max_probs = np.max(proba_matrix, axis=1)
                confidences.extend(np.round(max_probs, 4))
            else:
                confidences.extend([1.0] * len(batch))

        df.loc[unannotated_mask, "label_text"] = preds
        df.loc[unannotated_mask, "confidence_score"] = confidences
        df.loc[unannotated_mask, "prediction_source"] = "student_ml_model"

        # Áp dụng Ma trận quyết định Text-First cải tiến cho các bản ghi mới
        # Nhóm 1: Tự tin cao (>= 0.60) -> Ưu tiên label_text (sửa sai cho số sao)
        high_conf = unannotated_mask & (df["confidence_score"] >= 0.60)
        df.loc[high_conf, "label_final"] = df.loc[high_conf, "label_text"]
        df.loc[high_conf, "is_noisy"] = False
        df.loc[high_conf, "is_trainable"] = True

        # Nhóm 2: Tự tin thấp (< 0.60) & Đồng thuận -> An toàn
        low_conf_agree = unannotated_mask & (df["confidence_score"] < 0.60) & (df["label_weak"] == df["label_text"])
        df.loc[low_conf_agree, "label_final"] = df.loc[low_conf_agree, "label_weak"]
        df.loc[low_conf_agree, "is_noisy"] = False
        df.loc[low_conf_agree, "is_trainable"] = True

        # Nhóm 3: Tự tin thấp (< 0.60) & Xung đột -> Đánh dấu nhiễu chờ Gemini
        low_conf_conflict = unannotated_mask & (df["confidence_score"] < 0.60) & (df["label_weak"] != df["label_text"])
        df.loc[low_conf_conflict, "label_final"] = df.loc[low_conf_conflict, "label_weak"]
        df.loc[low_conf_conflict, "is_noisy"] = True
        df.loc[low_conf_conflict, "is_trainable"] = False

        # Gắn cờ mâu thuẫn giữa số sao và nhãn cuối cùng
        df.loc[unannotated_mask, "is_conflict"] = (
            df.loc[unannotated_mask, "label_weak"] != df.loc[unannotated_mask, "label_final"]
        )
    else:
        logger.info("Toàn bộ %d bản ghi có văn bản đã có nhãn Teacher chuẩn xác, không cần ghi đè.", usable_count)

    conflict_count = int(df["is_conflict"].sum())
    logger.info("Phát hiện mâu thuẫn giữa sao và cảm xúc thực tế: %d bản ghi (%.2f%%)", conflict_count, (conflict_count / max(usable_count, 1)) * 100)

    # 5. Khai báo tường minh Schema chuẩn của Gold (Explicit Contract Schema)
    # Tự động lọc chính xác 32 cột phục vụ kinh doanh/AI, triệt để loại bỏ in_seed và mọi cờ kỹ thuật rác
    selected_gold_cols = [col for col in GOLD_COLUMNS if col in df.columns]
    df_gold = df[selected_gold_cols].copy()

    # Xuất dữ liệu tầng Gold ra Parquet
    g_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Đang ghi dữ liệu tầng Gold ra: %s", g_path)
    df_gold.to_parquet(g_path, index=False, compression="snappy")
    logger.info("✅ Đã ghi thành công tầng Gold: %s (%d dòng, %d cột)", g_path, len(df_gold), len(df_gold.columns))

    # Tùy chọn đồng bộ vào MongoDB
    if sync_to_mongo:
        _sync_gold_to_mongodb(df_gold, mongo_uri)

    return df_gold


def _sync_gold_to_mongodb(df: pd.DataFrame, mongo_uri: str | None = None) -> None:
    """Đồng bộ tầng Gold vào MongoDB."""
    try:
        from pymongo import MongoClient
    except ImportError:
        logger.warning("Thư viện pymongo chưa cài đặt, bỏ qua đồng bộ MongoDB.")
        return

    settings = load_settings()
    uri = mongo_uri or settings.get("mongodb", {}).get("uri", "mongodb://localhost:27017")
    db_name = settings.get("mongodb", {}).get("database", "ecommerce_sentiment")
    coll_name = settings.get("mongodb", {}).get("collections", {}).get("reviews_gold", "gold_reviews")

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.server_info()  # Kiểm tra kết nối
        db = client[db_name]
        coll = db[coll_name]

        logger.info("Đang nạp %d bản ghi vào MongoDB [%s.%s]...", len(df), db_name, coll_name)
        # Chuyển đổi timestamp sang string hoặc datetime ISO
        records = df.to_dict(orient="records")
        for r in records:
            if isinstance(r.get("created_at"), pd.Timestamp):
                r["created_at"] = r["created_at"].isoformat()
            if isinstance(r.get("crawled_at"), pd.Timestamp):
                r["crawled_at"] = r["crawled_at"].isoformat()

        coll.delete_many({})  # Làm mới collection
        coll.insert_many(records, ordered=False)
        logger.info("✅ Đồng bộ MongoDB thành công!")
    except Exception as e:
        logger.warning("Không thể kết nối hoặc ghi vào MongoDB: %s. Tiếp tục lưu tại file Parquet.", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline suy luận cảm xúc và xuất tầng Gold.")
    parser.add_argument("--silver-path", default="data/silver/reviews.parquet", help="Đường dẫn file Silver")
    parser.add_argument("--model-path", default="models/best_sentiment_model.joblib", help="Đường dẫn file Model")
    parser.add_argument("--output-gold", default="data/gold/gold_reviews.parquet", help="Đường dẫn file Gold parquet")
    parser.add_argument("--batch-size", type=int, default=10000, help="Kích thước batch suy luận")
    parser.add_argument("--sync-mongo", action="store_true", help="Đồng bộ kết quả vào MongoDB")
    args = parser.parse_args()

    run_inference(
        silver_path=args.silver_path,
        model_path=args.model_path,
        output_gold_path=args.output_gold,
        batch_size=args.batch_size,
        sync_to_mongo=args.sync_mongo,
    )


if __name__ == "__main__":
    main()
