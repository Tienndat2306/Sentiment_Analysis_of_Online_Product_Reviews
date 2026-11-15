"""
Module Feature Engineering: Trích xuất đặc trưng văn bản cho phân tích cảm xúc
==============================================================================
Hỗ trợ nạp dữ liệu sạch từ tầng Silver (data/silver/reviews.parquet),
chuẩn bị tập dữ liệu huấn luyện và xây dựng bộ trích xuất đặc trưng TF-IDF
(hỗ trợ n-gram, sublinear TF) phục vụ các mô hình học máy phân loại cảm xúc.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Tuple

# Đảm bảo import được module từ thư mục gốc
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FeatureEngineering")


def load_training_data(
    parquet_path: str | Path | None = None,
    text_usable_only: bool = True,
    trainable_only: bool = True,
    deduplicate_text: bool = True,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Nạp dữ liệu đánh giá phục vụ trích xuất đặc trưng và huấn luyện mô hình.
    
    Mặc định ưu tiên nạp từ data/gold/gold_reviews.parquet (chứa nhãn AI label_final và cờ lọc is_trainable).
    Nếu không tìm thấy, dự phòng nạp từ data/silver/reviews.parquet (dùng label_weak).
    
    Args:
        parquet_path: Đường dẫn tới file Parquet (Gold hoặc Silver).
        text_usable_only: Nếu True, chỉ lấy các bản ghi có is_text_usable == True.
        trainable_only: Nếu True và có cột is_trainable, loại bỏ các mẫu nhiễu (is_noisy == True).
        deduplicate_text: Nếu True, khử trùng lặp nội dung chuẩn hóa chữ thường để chống rò rỉ dữ liệu (Data Leakage) giữa Train và Test.
        sample_size: Nếu cần lấy mẫu nhỏ hơn để thử nghiệm nhanh.
        random_state: Seed ngẫu nhiên khi lấy mẫu.
    """
    if parquet_path is None:
        gold_file = Path("data/gold/gold_reviews.parquet")
        silver_file = Path("data/silver/reviews.parquet")
        path = gold_file if gold_file.exists() else silver_file
    else:
        path = Path(parquet_path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {path.resolve()}")

    logger.info("Đang đọc dữ liệu từ: %s", path)
    import pyarrow.parquet as pq
    schema_cols = set(pq.read_schema(path).names)

    base_cols = [
        "review_id",
        "product_id",
        "product_name",
        "rating",
        "content_clean",
        "is_text_usable",
        "label_weak",
        "source",
        "created_at",
    ]
    columns_to_load = [c for c in base_cols if c in schema_cols]
    if "label_final" in schema_cols:
        columns_to_load.append("label_final")
    if "is_trainable" in schema_cols:
        columns_to_load.append("is_trainable")
    if "is_noisy" in schema_cols:
        columns_to_load.append("is_noisy")

    df = pd.read_parquet(path, columns=columns_to_load)
    logger.info("Tổng số bản ghi ban đầu: %d", len(df))

    if text_usable_only:
        df = df[df["is_text_usable"] == True].copy()
        logger.info("Số bản ghi hợp lệ cho NLP (is_text_usable == True): %d", len(df))

    if trainable_only and "is_trainable" in df.columns:
        df = df[df["is_trainable"] == True].copy()
        logger.info("Số bản ghi đạt chuẩn huấn luyện (is_trainable == True): %d", len(df))

    # Loại bỏ các chuỗi trống hoặc toàn khoảng trắng
    df["content_clean"] = df["content_clean"].fillna("").astype(str).str.strip()
    df = df[df["content_clean"].str.len() > 0].copy()

    # Nhãn mục tiêu: ưu tiên label_final, nếu trống dùng label_weak
    if "label_final" in df.columns and df["label_final"].notna().any():
        df["target_label"] = df["label_final"].fillna(df["label_weak"])
    else:
        df["target_label"] = df["label_weak"]

    # Lọc bỏ các dòng có nhãn unknown hoặc không hợp lệ
    valid_labels = {"positive", "neutral", "negative"}
    df = df[df["target_label"].isin(valid_labels)].copy()

    # Khử trùng lặp văn bản chuẩn hóa chữ thường để ngăn chặn Data Leakage giữa Train và Test
    if deduplicate_text:
        before_dedup = len(df)
        df["_norm_text"] = df["content_clean"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        df = df.drop_duplicates(subset=["_norm_text"], keep="first").drop(columns=["_norm_text"]).reset_index(drop=True)
        logger.info(
            "Đã khử trùng lặp văn bản chống Data Leakage: giữ lại %d câu duy nhất (loại bỏ %d câu trùng lặp).",
            len(df),
            before_dedup - len(df),
        )

    if sample_size and sample_size < len(df):
        logger.info("Lấy mẫu ngẫu nhiên %d bản ghi (seed=%d)", sample_size, random_state)
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    logger.info("Số bản ghi sẵn sàng huấn luyện: %d", len(df))
    logger.info("Phân bố nhãn:\n%s", df["target_label"].value_counts().to_string())
    return df


def get_fixed_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    split_path: str | Path = "data/gold/train_test_split.parquet",
    force_recreate: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tạo hoặc nạp tập chia Train/Test cố định (Fixed Split) đã khử trùng lặp,
    đảm bảo Scikit-learn và Spark MLlib đánh giá trên đúng một tập kiểm thử duy nhất.
    """
    from sklearn.model_selection import train_test_split

    s_path = Path(split_path)
    if not force_recreate and s_path.exists() and "review_id" in df.columns:
        split_df = pd.read_parquet(s_path)
        split_map = dict(zip(split_df["review_id"].astype(str), split_df["split"]))
        mapped_splits = df["review_id"].astype(str).map(split_map)
        if mapped_splits.notna().all():
            train_df = df[mapped_splits == "train"].reset_index(drop=True)
            test_df = df[mapped_splits == "test"].reset_index(drop=True)
            logger.info("Đã nạp tập chia cố định từ %s (Train: %d, Test: %d)", s_path, len(train_df), len(test_df))
            return train_df, test_df

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target_label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if "review_id" in df.columns and len(df) > 10000:
        s_path.parent.mkdir(parents=True, exist_ok=True)
        out_split = pd.concat(
            [
                pd.DataFrame({"review_id": train_df["review_id"].astype(str), "split": "train"}),
                pd.DataFrame({"review_id": test_df["review_id"].astype(str), "split": "test"}),
            ],
            ignore_index=True,
        )
        out_split.to_parquet(s_path, index=False, compression="snappy")
        logger.info("Đã lưu cố định tập chia Train/Test tại: %s", s_path)

    return train_df, test_df


# Tương thích ngược: Alias trỏ về tên hàm mới
load_silver_data = load_training_data


def create_tfidf_vectorizer(
    max_features: int = 15000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 3,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Khởi tạo TfidfVectorizer tối ưu cho tiếng Việt thương mại điện tử.
    
    Args:
        max_features: Số lượng từ/cụm từ tối đa giữ lại.
        ngram_range: Dải n-gram (1-gram và 2-gram để bắt cụm từ cảm xúc như 'không_thích', 'rất_tốt').
        min_df: Số tài liệu tối thiểu từ phải xuất hiện.
        max_df: Tỷ lệ tài liệu tối đa từ xuất hiện (loại bỏ từ quá phổ biến).
        sublinear_tf: Áp dụng tỉ lệ logarithmic 1 + log(tf) để giảm ảnh hưởng của từ lặp lại nhiều.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    return vectorizer


def save_vectorizer(vectorizer: TfidfVectorizer, output_path: str | Path) -> None:
    """Lưu TfidfVectorizer đã fit vào đĩa."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)
    logger.info("Đã lưu TF-IDF Vectorizer tại: %s", path)


def load_vectorizer(input_path: str | Path) -> TfidfVectorizer:
    """Nạp TfidfVectorizer từ đĩa."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy vectorizer tại: {path.resolve()}")
    return joblib.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trích xuất đặc trưng TF-IDF từ dữ liệu Silver.")
    parser.add_argument("--silver-path", default="data/silver/reviews.parquet", help="Đường dẫn Silver parquet")
    parser.add_argument("--output-vec", default="models/tfidf_vectorizer.joblib", help="Đường dẫn lưu Vectorizer")
    parser.add_argument("--max-features", type=int, default=15000, help="Số feature tối đa")
    parser.add_argument("--sample-size", type=int, default=None, help="Lấy mẫu bản ghi nếu cần")
    args = parser.parse_args()

    df = load_silver_data(parquet_path=args.silver_path, sample_size=args.sample_size)
    vectorizer = create_tfidf_vectorizer(max_features=args.max_features)

    logger.info("Bắt đầu fit TF-IDF vectorizer...")
    X = vectorizer.fit_transform(df["content_clean"])
    logger.info("Kích thước ma trận đặc trưng: %s (Bộ nhớ: ~%.2f MB)", X.shape, (X.data.nbytes + X.indptr.nbytes + X.indices.nbytes) / 1024 / 1024)

    save_vectorizer(vectorizer, args.output_vec)
    logger.info("Hoàn tất Feature Engineering!")


if __name__ == "__main__":
    main()
