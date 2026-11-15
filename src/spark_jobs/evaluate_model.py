"""
Evaluation Module: Đánh giá chuyên sâu và Phân tích lỗi (Error Analysis)
========================================================================
Nạp mô hình đã huấn luyện từ Model Registry (models/best_sentiment_model.joblib),
đánh giá hiệu năng tổng thể và phân tích các trường hợp dự đoán nhầm (misclassifications)
phục vụ báo cáo chuyên môn.
"""

from __future__ import annotations

import argparse
import json
import logging
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
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.spark_jobs.feature_engineering import get_fixed_train_test_split, load_training_data

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EvaluateModel")


def evaluate_pipeline(
    model_path: str | Path = "models/best_sentiment_model.joblib",
    data_path: str | Path | None = None,
    silver_path: str | Path | None = None,
    sample_size: int | None = None,
    test_only: bool = True,
    output_report_path: str | Path | None = "models/evaluation_report.json",
) -> dict[str, Any]:
    """Đánh giá toàn diện mô hình trên tập kiểm thử độc lập (Test set)."""
    m_path = Path(model_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model tại {m_path.resolve()}")

    logger.info("Đang nạp mô hình từ: %s", m_path)
    pipeline = joblib.load(m_path)

    actual_path = silver_path or data_path
    df = load_training_data(parquet_path=actual_path, deduplicate_text=True, sample_size=sample_size)
    if test_only and sample_size is None:
        _, df = get_fixed_train_test_split(df)
        logger.info("Chỉ đánh giá trên tập Test độc lập (Held-out Test Set): %d bản ghi", len(df))
    labels = ["negative", "neutral", "positive"]

    X = df["content_clean"].values
    y_true = df["target_label"].values

    logger.info("Bắt đầu suy luận để đánh giá trên %d bản ghi...", len(X))
    y_pred = pipeline.predict(X)

    # Báo cáo chi tiết
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    print("\n" + "=" * 70)
    print(" BÁO CÁO ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI CẢM XÚC")
    print("=" * 70)
    print(report_str)
    print("-" * 70)
    print("MA TRẬN NHẦM LẪN (CONFUSION MATRIX) [negative, neutral, positive]:")
    for row_idx, row in enumerate(cm):
        print(f"  Thực tế {labels[row_idx]:<9}: {row}")
    print("=" * 70 + "\n")

    # Phân tích các trường hợp nhầm lẫn (Error Analysis)
    df["predicted_label"] = y_pred
    df["is_correct"] = df["target_label"] == df["predicted_label"]
    errors = df[~df["is_correct"]]

    logger.info("Tổng số lỗi dự đoán: %d (%.2f%%)", len(errors), (len(errors) / len(df)) * 100)

    # Thu thập một số mẫu nhầm lẫn điển hình (ví dụ: Thực tế Tiêu cực nhưng đoán Tích cực)
    error_samples = []
    severe_errors = errors[(errors["target_label"] == "negative") & (errors["predicted_label"] == "positive")].head(5)
    for _, row in severe_errors.iterrows():
        error_samples.append({
            "review_id": str(row.get("review_id", "")),
            "text": str(row["content_clean"])[:150],
            "true_label": str(row["target_label"]),
            "predicted_label": str(row["predicted_label"]),
            "rating": int(row.get("rating", 0)),
        })

    eval_result = {
        "model_path": str(m_path.as_posix()),
        "total_evaluated": len(df),
        "accuracy": round(float(report_dict["accuracy"]), 4),
        "macro_f1": round(float(report_dict["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report_dict["weighted avg"]["f1-score"]), 4),
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "labels": labels,
        "sample_misclassifications": error_samples,
    }

    if output_report_path:
        out_p = Path(output_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)
        logger.info("Đã lưu báo cáo đánh giá tại: %s", out_p)

    return eval_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá mô hình phân loại cảm xúc.")
    parser.add_argument("--model-path", default="models/best_sentiment_model.joblib", help="Đường dẫn file model .joblib")
    parser.add_argument("--data-path", "--silver-path", default=None, dest="data_path", help="Đường dẫn file Parquet (Gold hoặc Silver)")
    parser.add_argument("--output-report", default="models/evaluation_report.json", help="Đường dẫn file json báo cáo")
    parser.add_argument("--sample-size", type=int, default=None, help="Lấy mẫu đánh giá nếu muốn test nhanh")
    args = parser.parse_args()

    evaluate_pipeline(
        model_path=args.model_path,
        data_path=args.data_path,
        sample_size=args.sample_size,
        output_report_path=args.output_report,
    )


if __name__ == "__main__":
    main()
