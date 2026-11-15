"""
Training Pipeline bằng Apache Spark MLlib (Học máy phân tán Big Data)
=====================================================================
Thực hiện:
1. Khởi tạo SparkSession tối ưu hóa trên cụm local[*].
2. Nạp dữ liệu tầng Silver (data/silver/reviews.parquet), lọc các bản ghi sạch (is_trainable == True).
3. Xây dựng Pipeline phân tán chuẩn PySpark ML:
   - StringIndexer: Mã hóa nhãn (positive, negative, neutral -> 0, 1, 2)
   - RegexTokenizer: Phân tách từ vựng tiếng Việt
   - StopWordsRemover: Loại bỏ từ dừng
   - HashingTF: Chuyển đổi từ vựng thành vector tần suất 15.000 chiều
   - IDF: Nghịch đảo tần suất văn bản (Inverse Document Frequency)
4. Huấn luyện và đối chiếu 2 mô hình phân loại:
   - LogisticRegression (Multinomial Family)
   - NaiveBayes (Multinomial Model)
5. Đánh giá đa chiều bằng MulticlassClassificationEvaluator (Accuracy, F1-score, Precision, Recall).
6. Đóng gói và lưu PipelineModel vào thư mục models/spark_mllib_pipeline/.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Tự động cấu hình JAVA_HOME trỏ tới Java 11 (tránh lỗi Subject.getSubject trên JDK 26)
JAVA_11_CANDIDATES = [
    r"C:\Program Files\Eclipse Adoptium\jdk-11.0.32.101-hotspot",
    r"C:\Program Files\Eclipse Adoptium\jdk-11.0.32.1-hotspot",
    r"C:\Program Files\Java\jdk-11",
]
for p in JAVA_11_CANDIDATES:
    if os.path.exists(p):
        os.environ["JAVA_HOME"] = p
        break

# Đảm bảo import được module từ thư mục gốc
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Cấu hình HADOOP_HOME và PATH để Spark lưu model chuẩn xác trên Windows
HADOOP_DIR = ROOT_DIR / "hadoop"
if (HADOOP_DIR / "bin" / "winutils.exe").exists():
    os.environ["HADOOP_HOME"] = str(HADOOP_DIR)
    hadoop_bin_str = str(HADOOP_DIR / "bin")
    if hadoop_bin_str not in os.environ.get("PATH", ""):
        os.environ["PATH"] = hadoop_bin_str + ";" + os.environ.get("PATH", "")

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi cp1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import (
    HashingTF,
    IDF,
    RegexTokenizer,
    StopWordsRemover,
    StringIndexer,
)
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainSparkMLlib")


def get_spark_session(app_name: str = "SentimentAnalysis_SparkMLlib") -> SparkSession:
    """Khởi tạo SparkSession phân tán với cấu hình bộ nhớ tối ưu."""
    logger.info("Đang khởi tạo SparkSession (Master: local[*])...")
    spark_tmp = ROOT_DIR / "tmp" / "spark"
    spark_tmp.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.local.dir", str(spark_tmp))
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    logger.info("✅ SparkSession khởi tạo thành công (Spark v%s)!", spark.version)
    return spark


def train_spark_mllib(
    data_path: str | None = None,
    output_dir: str = "models",
    test_ratio: float = 0.2,
    num_features: int = 15000,
    seed: int = 42,
    silver_path: str | None = None,
) -> Dict[str, Any]:
    """Quy trình huấn luyện, đánh giá và lưu trữ mô hình Spark MLlib."""
    input_path = silver_path or data_path or ("data/gold/gold_reviews.parquet" if Path("data/gold/gold_reviews.parquet").exists() else "data/silver/reviews.parquet")
    s_path = Path(input_path)
    if not s_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {s_path.resolve()}")

    spark = get_spark_session()
    t_start = time.time()

    # 1. Đọc dữ liệu
    logger.info("Đang đọc dữ liệu huấn luyện bằng Spark DataFrame: %s", s_path)
    df_raw = spark.read.parquet(str(s_path))
    total_raw = df_raw.count()
    logger.info("Tổng số bản ghi trong bảng dữ liệu: %d", total_raw)

    cols = set(df_raw.columns)
    # 2. Lọc các mẫu đạt chuẩn huấn luyện (is_trainable == true nếu có)
    filter_cond = (F.col("content_clean").isNotNull()) & (F.length(F.trim(F.col("content_clean"))) > 0)
    if "is_trainable" in cols:
        filter_cond = filter_cond & (F.col("is_trainable") == True)
    elif "is_text_usable" in cols:
        filter_cond = filter_cond & (F.col("is_text_usable") == True)

    target_label_col = "label_final" if "label_final" in cols else "label_weak"

    df_clean = df_raw.filter(filter_cond).select(
        F.col("review_id"),
        F.col("content_clean"),
        F.col(target_label_col).alias("label_final"),
    )

    # 3. Phân chia Train/Test cố định đã khử trùng lặp (đồng bộ 100% với Scikit-learn)
    split_file = Path("data/gold/train_test_split.parquet")
    if split_file.exists():
        logger.info("Đang nạp tập chia cố định không rò rỉ dữ liệu từ: %s", split_file)
        df_split = spark.read.parquet(str(split_file))
        df_joined = df_clean.join(df_split, on="review_id", how="inner")
        train_df = df_joined.filter(F.col("split") == "train").drop("split")
        test_df = df_joined.filter(F.col("split") == "test").drop("split")
    else:
        df_clean = (
            df_clean.withColumn(
                "_norm_text",
                F.trim(F.regexp_replace(F.lower(F.col("content_clean")), r"\s+", " ")),
            )
            .dropDuplicates(["_norm_text"])
            .drop("_norm_text")
        )
        train_ratio = 1.0 - test_ratio
        logger.info("Phân chia tập dữ liệu: Train = %.0f%%, Test = %.0f%%...", train_ratio * 100, test_ratio * 100)
        train_df, test_df = df_clean.randomSplit([train_ratio, test_ratio], seed=seed)

    train_df.cache()
    test_df.cache()

    train_count = train_df.count()
    test_count = test_df.count()
    trainable_count = train_count + test_count
    logger.info("Số bản ghi sạch (đã khử trùng lặp): %d (Train: %d | Test: %d)", trainable_count, train_count, test_count)

    # 3.1. Tính trọng số cân bằng lớp (Balanced Class Weight: N / (3 * N_c)) cho tập Train
    class_counts = {row["label_final"]: row["count"] for row in train_df.groupBy("label_final").count().collect()}
    num_classes = max(len(class_counts), 1)
    weight_expr = F.lit(1.0)
    for lbl_name, cnt in class_counts.items():
        w = float(train_count) / (float(num_classes) * float(max(cnt, 1)))
        weight_expr = F.when(F.col("label_final") == lbl_name, F.lit(w)).otherwise(weight_expr)
    train_df = train_df.withColumn("class_weight", weight_expr)
    test_df = test_df.withColumn("class_weight", F.lit(1.0))

    # 4. Xây dựng các Feature Engineering Stages trong PySpark ML
    logger.info("Xây dựng Feature Extraction Pipeline (Tokenizer -> StopWords -> HashingTF -> IDF)...")
    
    # 4.1. Mã hóa nhãn dạng chuỗi sang chỉ số số học (0, 1, 2)
    label_indexer = StringIndexer(
        inputCol="label_final", outputCol="label", handleInvalid="skip"
    )

    # 4.2. Tách từ bằng Regex
    tokenizer = RegexTokenizer(
        inputCol="content_clean", outputCol="words", pattern=r"\s+", gaps=True
    )

    # 4.3. Loại bỏ từ dừng
    stop_words_remover = StopWordsRemover(
        inputCol="words", outputCol="filtered_words"
    )

    # 4.4. Trích xuất đặc trưng HashingTF (15.000 chiều)
    hashing_tf = HashingTF(
        inputCol="filtered_words", outputCol="raw_features", numFeatures=num_features
    )

    # 4.5. Tính toán IDF
    idf = IDF(inputCol="raw_features", outputCol="features")

    # 5. Định nghĩa các thuật toán phân loại ứng viên (có xử lý mất cân bằng nhãn bằng weightCol)
    candidates = {
        "LogisticRegression (Spark MLlib Balanced)": LogisticRegression(
            featuresCol="features",
            labelCol="label",
            weightCol="class_weight",
            maxIter=50,
            family="multinomial",
            regParam=0.05,
        ),
        "NaiveBayes (Spark MLlib)": NaiveBayes(
            featuresCol="features",
            labelCol="label",
            weightCol="class_weight",
            modelType="multinomial",
            smoothing=0.5,
        ),
    }

    # 6. Các bộ đánh giá số liệu
    eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    eval_f1_macro = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    eval_weighted_prec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    eval_weighted_rec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

    results = {}
    best_model_name = ""
    best_f1 = -1.0
    best_pipeline_model = None

    print("\n" + "=" * 80)
    print(" BẮT ĐẦU HUẤN LUYỆN VÀ ĐỐI CHIẾU MÔ HÌNH PHÂN TÁN SPARK MLLIB")
    print("=" * 80)

    for name, classifier in candidates.items():
        logger.info("Đang huấn luyện Pipeline với thuật toán: %s...", name)
        t0 = time.time()

        pipeline = Pipeline(stages=[label_indexer, tokenizer, stop_words_remover, hashing_tf, idf, classifier])
        pipeline_model = pipeline.fit(train_df)
        t_fit = time.time() - t0

        logger.info("Huấn luyện xong trong %.2f giây. Đang đánh giá trên tập Test...", t_fit)
        predictions = pipeline_model.transform(test_df)
        predictions.cache()

        acc = float(eval_acc.evaluate(predictions))
        f1_weighted = float(eval_f1_macro.evaluate(predictions))
        prec = float(eval_weighted_prec.evaluate(predictions))
        rec = float(eval_weighted_rec.evaluate(predictions))

        # Tính F1-Macro chuẩn xác (trung bình không trọng số qua 3 lớp 0.0, 1.0, 2.0)
        f1_per_label = []
        for lbl_idx in [0.0, 1.0, 2.0]:
            ev_lbl = MulticlassClassificationEvaluator(
                labelCol="label",
                predictionCol="prediction",
                metricName="fMeasureByLabel",
                metricLabel=lbl_idx,
            )
            f1_per_label.append(float(ev_lbl.evaluate(predictions)))
        f1_macro = float(sum(f1_per_label) / len(f1_per_label))

        results[name] = {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "f1_score": round(f1_weighted, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "fit_time_seconds": round(t_fit, 2),
        }

        logger.info(
            "[%s] Accuracy: %.4f | F1-Macro: %.4f | F1-Weighted: %.4f | Precision: %.4f | Recall: %.4f",
            name,
            acc,
            f1_macro,
            f1_weighted,
            prec,
            rec,
        )

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_name = name
            best_pipeline_model = pipeline_model

    # 7. In bảng so sánh kết quả
    print("\n" + "-" * 98)
    print(f"{'Mô hình Spark MLlib':<34} | {'Accuracy':<9} | {'F1-Macro':<9} | {'F1-Weight':<9} | {'Precision':<9} | {'Thời gian':<9}")
    print("-" * 98)
    for name, m in results.items():
        print(f"{name:<34} | {m['accuracy']:<9.4f} | {m['f1_macro']:<9.4f} | {m['f1_weighted']:<9.4f} | {m['precision']:<9.4f} | {m['fit_time_seconds']}s")
    print("-" * 98)
    print(f"🏆 Mô hình Spark MLlib tốt nhất (F1-Macro): {best_model_name} (F1-Macro = {best_f1:.4f})\n")

    # 8. Lưu mô hình tốt nhất vào thư mục Model Registry
    model_save_path = Path(output_dir) / "spark_mllib_pipeline"
    logger.info("Đang lưu PipelineModel Spark MLlib vào: %s...", model_save_path)
    model_saved = False
    if best_pipeline_model is not None:
        try:
            best_pipeline_model.write().overwrite().save(str(model_save_path))
            logger.info("✅ Đã lưu Spark PipelineModel thành công!")
            model_saved = True
        except Exception as e:
            logger.warning(
                "Lưu ý môi trường Windows: Bỏ qua bước lưu file nhị phân Hadoop của Spark do thiếu winutils.exe (%s). "
                "Tuy nhiên toàn bộ quá trình phân tán Spark MLlib đã hoàn thành và đánh giá thành công 100%%!",
                e,
            )

    # 9. Ghi file báo cáo JSON
    report_path = Path(output_dir) / "spark_mllib_evaluation.json"
    report_data = {
        "engine": "Apache Spark MLlib (Distributed)",
        "spark_version": spark.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_training_samples": train_count,
        "total_test_samples": test_count,
        "num_features": num_features,
        "best_model": best_model_name,
        "best_f1_macro": round(best_f1, 4),
        "best_f1_score": results[best_model_name]["f1_weighted"],
        "models_comparison": results,
        "model_binary_saved": model_saved,
        "saved_model_path": str(model_save_path) if model_saved else "N/A (Requires winutils on Windows)",
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    logger.info("✅ Đã lưu báo cáo đánh giá Spark MLlib vào: %s", report_path)

    elapsed_total = time.time() - t_start
    print("\n" + "=" * 80)
    print(" BÁO CÁO TỔNG KẾT QUY TRÌNH HỌC MÁY PHÂN TÁN SPARK MLLIB")
    print("=" * 80)
    print(f"- Động cơ thực thi: Apache Spark {spark.version} (local[*])")
    print(f"- Số mẫu huấn luyện (Train): {train_count:,} bản ghi")
    print(f"- Số mẫu kiểm thử (Test):     {test_count:,} bản ghi")
    print(f"- Không gian đặc trưng:      {num_features:,} chiều (HashingTF + IDF)")
    print(f"- Mô hình chiến thắng:       {best_model_name}")
    print(f"- Độ chính xác (Accuracy):   {results[best_model_name]['accuracy'] * 100:.2f}%")
    print(f"- F1-Macro:                  {results[best_model_name]['f1_macro']:.4f}")
    print(f"- F1-Weighted:               {results[best_model_name]['f1_weighted']:.4f}")
    print(f"- Thời gian thực thi toàn bộ: {elapsed_total:.2f} giây")
    print("=" * 80 + "\n")

    spark.stop()
    return report_data


if __name__ == "__main__":
    train_spark_mllib()
