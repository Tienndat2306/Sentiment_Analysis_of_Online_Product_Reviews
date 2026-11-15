"""
Training Pipeline: Huấn luyện và So sánh Đa Mô hình Phân loại Cảm xúc
===================================================================
Thực hiện:
1. Phân chia Train / Test phân tầng (Stratified 80/20).
2. Xử lý mất cân bằng nhãn bằng class_weight='balanced' & ComplementNB.
3. Huấn luyện và đối chiếu 4 mô hình:
   - Baseline: Multinomial Naive Bayes
   - Tối ưu mất cân bằng: Complement Naive Bayes
   - Mô hình tuyến tính: Logistic Regression (Balanced)
   - Tối ưu siêu phẳng: Linear SVM (Calibrated with Probabilities)
4. Đánh giá chi tiết (Accuracy, F1-macro, Recall từng lớp, Confusion Matrix).
5. Tự động đóng gói Pipeline (TF-IDF + Classifier), lưu vào Model Registry (models/).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.spark_jobs.feature_engineering import (
    create_tfidf_vectorizer,
    get_fixed_train_test_split,
    load_training_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainModel")


def evaluate_classifier(
    clf: Any,
    X_test: Any,
    y_test: pd.Series,
    labels: list[str],
) -> Dict[str, Any]:
    """Đánh giá chi tiết mô hình trên tập kiểm thử."""
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1_mac = float(f1_score(y_test, y_pred, average="macro"))
    f1_weight = float(f1_score(y_test, y_pred, average="weighted"))

    p, r, f, s = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)
    per_class = {}
    for idx, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(p[idx]), 4),
            "recall": round(float(r[idx]), 4),
            "f1_score": round(float(f[idx]), 4),
            "support": int(s[idx]),
        }

    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_weight, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def update_model_registry(
    metadata_path: str | Path,
    new_model_info: Dict[str, Any],
) -> None:
    """Cập nhật metadata.json của Model Registry."""
    meta_file = Path(metadata_path)
    data: Dict[str, Any] = {
        "project": "Sentiment Analysis of Online Product Reviews",
        "version": "1.0.0",
        "active_model": None,
        "models": [],
    }

    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    data = loaded
        except Exception as e:
            logger.warning("Không thể đọc metadata.json cũ (%s), khởi tạo lại.", e)

    data["active_model"] = new_model_info["model_id"]
    if "models" not in data or not isinstance(data["models"], list):
        data["models"] = []

    # Thêm model mới vào danh sách
    data["models"].append(new_model_info)

    meta_file.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Đã cập nhật Model Registry tại: %s", meta_file)


def train_and_compare(
    data_path: str | None = None,
    output_dir: str = "models",
    test_size: float = 0.2,
    max_features: int = 15000,
    sample_size: int | None = None,
    random_state: int = 42,
    silver_path: str | None = None,
) -> Dict[str, Any]:
    """Huấn luyện, so sánh các mô hình và lưu lại mô hình tốt nhất."""
    actual_path = silver_path or data_path
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Nạp dữ liệu huấn luyện (ưu tiên tầng Gold, dự phòng tầng Silver, khử trùng lặp chống Data Leakage)
    df = load_training_data(
        parquet_path=actual_path,
        deduplicate_text=True,
        sample_size=sample_size,
        random_state=random_state,
    )
    labels = ["negative", "neutral", "positive"]
    y = df["target_label"].values

    # 2. Phân chia Stratified Train/Test cố định (Fixed Split không rò rỉ dữ liệu)
    logger.info("Phân chia Stratified Train/Test cố định (Test size = %.1f%%)...", test_size * 100)
    train_df, test_df = get_fixed_train_test_split(
        df, test_size=test_size, random_state=random_state, force_recreate=True
    )
    X_train_raw = train_df["content_clean"].values
    y_train = train_df["target_label"].values
    X_test_raw = test_df["content_clean"].values
    y_test = test_df["target_label"].values
    logger.info("Train set: %d mẫu | Test set: %d mẫu", len(X_train_raw), len(X_test_raw))

    # 3. Trích xuất đặc trưng TF-IDF (Fit duy nhất trên Train)
    logger.info("Khởi tạo và fit TF-IDF Vectorizer (max_features=%d, n-gram=(1,2))...", max_features)
    vectorizer = create_tfidf_vectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    logger.info("Kích thước ma trận Train: %s | Test: %s", X_train.shape, X_test.shape)

    # 4. Định nghĩa danh sách các ứng viên mô hình
    candidates: dict[str, Any] = {
        "MultinomialNB (Baseline)": MultinomialNB(alpha=0.5),
        "ComplementNB (Imbalanced-optimized)": ComplementNB(alpha=0.5),
        "LogisticRegression (Balanced)": LogisticRegression(
            class_weight="balanced",
            max_iter=300,
            C=1.0,
            solver="lbfgs",
            random_state=random_state,
        ),
        "LinearSVC (Balanced, Calibrated)": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=random_state, max_iter=2000),
            cv=3,
        ),
    }

    results: dict[str, Dict[str, Any]] = {}
    best_model_name = ""
    best_f1_macro = -1.0
    best_clf = None

    print("\n" + "=" * 80)
    print(" BẮT ĐẦU HUẤN LUYỆN VÀ SO SÁNH CÁC MÔ HÌNH PHÂN LOẠI CẢM XÚC")
    print("=" * 80)

    for name, clf in candidates.items():
        logger.info("Đang huấn luyện mô hình: %s...", name)
        clf.fit(X_train, y_train)

        metrics = evaluate_classifier(clf, X_test, y_test, labels=labels)
        results[name] = metrics

        logger.info(
            "[%s] Accuracy: %.4f | F1-Macro: %.4f | F1-Weighted: %.4f",
            name,
            metrics["accuracy"],
            metrics["f1_macro"],
            metrics["f1_weighted"],
        )

        if metrics["f1_macro"] > best_f1_macro:
            best_f1_macro = metrics["f1_macro"]
            best_model_name = name
            best_clf = clf

    # 5. In bảng tổng hợp so sánh trực quan
    print("\n" + "-" * 88)
    print(f"{'Mô hình':<38} | {'Accuracy':<10} | {'F1-Macro':<10} | {'F1-Weight':<10} | {'Rec(Neg)':<9} | {'Rec(Neu)':<9}")
    print("-" * 88)
    for name, m in results.items():
        rec_neg = m["per_class"]["negative"]["recall"]
        rec_neu = m["per_class"]["neutral"]["recall"]
        print(
            f"{name:<38} | {m['accuracy']:<10.4f} | {m['f1_macro']:<10.4f} | {m['f1_weighted']:<10.4f} | {rec_neg:<9.4f} | {rec_neu:<9.4f}"
        )
    print("-" * 88)
    print(f"🏆 Mô hình chiến thắng (F1-macro cao nhất): {best_model_name} (F1-macro = {best_f1_macro:.4f})\n")

    # 6. Đóng gói hoàn chỉnh Pipeline (TF-IDF + Best Classifier)
    full_pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", best_clf),
    ])

    # 7. Lưu Artifacts
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_id = f"sentiment_pipeline_{timestamp}"
    versioned_filename = f"{model_id}.joblib"
    best_symlink_filename = "best_sentiment_model.joblib"

    versioned_path = out_path / versioned_filename
    best_path = out_path / best_symlink_filename

    logger.info("Đang lưu Pipeline vào: %s và %s", versioned_path, best_path)
    joblib.dump(full_pipeline, versioned_path)
    joblib.dump(full_pipeline, best_path)

    # Lưu confusion matrix và báo cáo của mô hình tốt nhất
    cm_info = {
        "model_name": best_model_name,
        "labels": labels,
        "confusion_matrix": results[best_model_name]["confusion_matrix"],
        "per_class_metrics": results[best_model_name]["per_class"],
    }
    with open(out_path / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(cm_info, f, ensure_ascii=False, indent=2)

    # 8. Cập nhật Model Registry
    class_dist = {str(k): int(v) for k, v in pd.Series(y).value_counts().items()}
    new_model_record = {
        "model_id": model_id,
        "model_name": best_model_name,
        "algorithm": best_clf.__class__.__name__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(X_train_raw),
        "test_samples": len(X_test_raw),
        "feature_extractor": "TfidfVectorizer",
        "max_features": max_features,
        "ngram_range": [1, 2],
        "class_distribution": class_dist,
        "metrics": {
            "accuracy": results[best_model_name]["accuracy"],
            "f1_macro": results[best_model_name]["f1_macro"],
            "f1_weighted": results[best_model_name]["f1_weighted"],
            "per_class": results[best_model_name]["per_class"],
        },
        "all_candidate_metrics": {
            k: {
                "accuracy": v["accuracy"],
                "f1_macro": v["f1_macro"],
                "f1_weighted": v["f1_weighted"],
            }
            for k, v in results.items()
        },
        "artifact_path": str(best_path.as_posix()),
    }

    update_model_registry(out_path / "metadata.json", new_model_record)

    logger.info("✅ Pipeline huấn luyện mô hình đã hoàn thành xuất sắc!")
    return new_model_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện và so sánh mô hình phân tích cảm xúc.")
    parser.add_argument("--data-path", "--silver-path", default=None, dest="data_path", help="Đường dẫn file Parquet (Gold hoặc Silver)")
    parser.add_argument("--output-dir", default="models", help="Thư mục lưu trữ Model Registry")
    parser.add_argument("--max-features", type=int, default=15000, help="Số feature TF-IDF tối đa")
    parser.add_argument("--test-size", type=float, default=0.2, help="Tỷ lệ tập kiểm thử")
    parser.add_argument("--sample-size", type=int, default=None, help="Lấy mẫu nếu muốn test nhanh")
    args = parser.parse_args()

    train_and_compare(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        max_features=args.max_features,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
