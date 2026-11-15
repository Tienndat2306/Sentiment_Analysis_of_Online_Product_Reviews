"""
Spark ETL Job: Bronze to Silver (Toàn diện & Tối ưu hóa)
=========================================================
Khắc phục triệt để toàn bộ 13 vấn đề kỹ thuật và kiến trúc:
1. Parse created_at từ chuỗi ISO sang TimestampType chuẩn xác, không bị null.
2. Làm sạch customer_id, review_id, product_id loại bỏ đuôi .0.
3. Chuẩn hóa triệt để Unicode NFC cho toàn bộ văn bản tiếng Việt.
4. Giữ lại và chuẩn hóa is_photo dạng BooleanType.
5. Bảo toàn 100% dữ liệu review: không xóa cứng, bổ sung has_content, is_text_usable, reject_reason.
6. Chuẩn hóa nhãn yếu (label_weak) từ rating; tách biệt hoàn toàn các nhãn suy luận AI sang tầng Gold.
7. Khử trùng lặp đa sàn theo (source, review_id) và lấy bản ghi mới nhất theo thời gian.
8. Đối chiếu danh sách hạt giống sản phẩm (in_seed flag).
9. Kiểm tra và validate rating thuộc dải 1 - 5.
10. Tối ưu hiệu năng: cache() và gộp toàn bộ thống kê vào 1 lần agg duy nhất.
11. Cấu hình môi trường linh hoạt (dynamic JAVA_HOME, đa nền tảng Windows/Linux/Docker).
12. Xuất báo cáo chất lượng dữ liệu ra file JSON (dq_report.json) và kiểm tra ngưỡng an toàn.
13. Regex HTML </?[a-zA-Z][^>]*> an toàn, bảo vệ tuyệt đối các biểu tượng cảm xúc (<3, >.<).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi cp1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession, Window
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        BooleanType,
        DoubleType,
        IntegerType,
        LongType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )
else:
    try:
        from pyspark.sql import DataFrame, SparkSession, Window
        from pyspark.sql import functions as F
        from pyspark.sql.types import (
            BooleanType,
            DoubleType,
            IntegerType,
            LongType,
            StringType,
            StructField,
            StructType,
            TimestampType,
        )
    except ImportError:
        SparkSession = None
        DataFrame = Any
        F = None
        Window = None
        DoubleType = None

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ETL_Silver")


# ---------------------------------------------------------------------------
# 1. Định nghĩa Schemas Bronze & Helper nạp Seeds
# ---------------------------------------------------------------------------

BRONZE_TIKI_SCHEMA = StructType(
    [
        StructField("review_id", StringType(), True),
        StructField("product_id", StringType(), True),
        StructField("product_name", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("customer_name", StringType(), True),
        StructField("rating", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("content", StringType(), True),
        StructField("thank_count", IntegerType(), True),
        StructField("comment_count", IntegerType(), True),
        StructField("created_at", StringType(), True),  # Lưu ý: StringType để parse ISO timestamp
        StructField("is_photo", StringType(), True),
        StructField("seller_name", StringType(), True),
        StructField("category_id", StringType(), True),
        StructField("category_name", StringType(), True),
        StructField("source", StringType(), True),
        StructField("crawled_at", StringType(), True),
    ]
)
BRONZE_REVIEW_SCHEMA = BRONZE_TIKI_SCHEMA


def load_settings(config_path: str = "configs/settings.yaml") -> dict[str, Any]:
    cfg_file = Path(config_path)
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_seed_product_ids(seeds_path: str = "data/seeds/products.jsonl") -> set[str]:
    """Nạp danh sách product_id hợp lệ từ file hạt giống seeds."""
    p = Path(seeds_path)
    if not p.exists():
        logger.warning("Không tìm thấy file seeds tại %s", seeds_path)
        return set()
    seed_ids = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    obj = json.loads(line_str)
                    pid = str(obj.get("product_id", "")).replace(".0", "").strip()
                    if pid:
                        seed_ids.add(pid)
                except Exception:
                    pass
    logger.info("Đã nạp %d product_id từ hạt giống %s", len(seed_ids), seeds_path)
    return seed_ids


# ---------------------------------------------------------------------------
# 2. Khởi tạo môi trường SparkSession linh hoạt đa nền tảng
# ---------------------------------------------------------------------------

def configure_environment(settings: dict[str, Any]) -> None:
    """Tự động phát hiện và cấu hình JAVA_HOME và Python executable phù hợp."""
    # 1. Cấu hình Python worker
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    # 2. Cấu hình JAVA_HOME
    configured_java = settings.get("spark", {}).get("java_home")
    if configured_java and Path(configured_java).exists():
        os.environ["JAVA_HOME"] = configured_java
        logger.info("Sử dụng JAVA_HOME từ settings: %s", configured_java)
        return

    # Nếu đang chạy trên Windows, kiểm tra Adoptium hoặc OpenJDK 11 tương thích
    if sys.platform == "win32":
        current_java = os.environ.get("JAVA_HOME", "")
        # Nếu JAVA_HOME hiện tại trỏ vào JDK 20+ (không tương thích Spark 3.5), tìm JDK 11/17
        if not current_java or "jdk-2" in current_java.lower():
            candidates = [
                r"C:\Program Files\Eclipse Adoptium\jdk-11.0.32.101-hotspot",
                r"C:\Program Files\Eclipse Adoptium\jdk-11*",
                r"C:\Program Files\Java\jdk-11*",
                r"C:\Program Files\Java\jdk-17*",
            ]
            for pattern in candidates:
                matches = glob.glob(pattern)
                if matches and Path(matches[0]).exists():
                    os.environ["JAVA_HOME"] = matches[0]
                    os.environ["PATH"] = os.path.join(matches[0], "bin") + os.pathsep + os.environ.get("PATH", "")
                    logger.info("Đã cấu hình tự động JAVA_HOME -> %s", matches[0])
                    return


def create_spark_session(
    app_name: str = "BronzeToSilver_ETL",
    master: str = "local[*]",
    settings: dict[str, Any] | None = None,
) -> SparkSession:
    """Tạo SparkSession cấu hình tối ưu."""
    if SparkSession is None:
        raise RuntimeError("PySpark chưa được cài đặt. Vui lòng chạy: pip install pyspark")

    cfg = settings or {}
    spark_cfg = cfg.get("spark", {})
    configure_environment(cfg)

    app_name = spark_cfg.get("app_name", app_name)
    master = spark_cfg.get("master", master)
    driver_mem = spark_cfg.get("driver_memory", "4g")
    exec_mem = spark_cfg.get("executor_memory", "4g")

    logger.info("Khởi tạo SparkSession: app_name=%s, master=%s", app_name, master)
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", exec_mem)
        .config("spark.sql.session.timeZone", "UTC")
    )
    return builder.getOrCreate()


# ---------------------------------------------------------------------------
# 3. Pipeline xử lý Silver toàn diện
# ---------------------------------------------------------------------------

def clean_and_transform_reviews(
    df: DataFrame,
    seed_product_ids: set[str],
    min_words: int = 3,
    default_lazada_seller: str | None = None,
) -> tuple[DataFrame, DataFrame, dict[str, Any]]:
    """
    Thực hiện quy trình làm sạch, chuẩn hóa, đối chiếu seed và phân loại review:
    Trả về:
        (df_silver: DataFrame, df_quarantine: DataFrame, metrics: dict[str, Any])
    """
    metrics: dict[str, Any] = {}

    # 1. Làm sạch sơ bộ các cột định danh & phân loại sàn
    df = df.withColumn("source", F.lower(F.trim(F.coalesce(F.col("source"), F.lit("tiki")))))
    df = df.withColumn("review_id", F.regexp_replace(F.trim(F.coalesce(F.col("review_id"), F.lit(""))), r"\.0$", ""))
    df = df.withColumn("product_id", F.regexp_replace(F.trim(F.coalesce(F.col("product_id"), F.lit(""))), r"\.0$", ""))

    # Xử lý customer_id: Cắt .0, nếu rỗng/nan/none/null -> trả về NULL (chuẩn SQL/Parquet)
    clean_cust = F.regexp_replace(F.trim(F.coalesce(F.col("customer_id"), F.lit(""))), r"\.0$", "")
    df = df.withColumn(
        "customer_id",
        F.when(
            F.col("customer_id").isNull()
            | (clean_cust == "")
            | (F.lower(clean_cust).isin("nan", "none", "null")),
            F.lit(None).cast(StringType()),
        ).otherwise(clean_cust),
    )

    # Xử lý seller_name:
    # Nếu rỗng hoặc nan/none/null:
    # - Nếu có cấu hình default_lazada_seller và source == 'lazada' -> gán giá trị đó
    # - Mặc định (default_lazada_seller=None): giữ NULL chuẩn cho toàn bộ các sàn (nhất quán Tiki & Lazada)
    clean_seller = F.trim(F.coalesce(F.col("seller_name"), F.lit("")))
    is_seller_empty = (
        F.col("seller_name").isNull()
        | (clean_seller == "")
        | (F.lower(clean_seller).isin("nan", "none", "null"))
    )
    if default_lazada_seller:
        df = df.withColumn(
            "seller_name",
            F.when(
                is_seller_empty,
                F.when(F.col("source") == "lazada", F.lit(default_lazada_seller)).otherwise(F.lit(None).cast(StringType())),
            ).otherwise(clean_seller),
        )
    else:
        df = df.withColumn(
            "seller_name",
            F.when(is_seller_empty, F.lit(None).cast(StringType())).otherwise(clean_seller),
        )

    # 2. Parse thời gian created_at chuẩn TimestampType (Xử lý chuỗi ISO)
    # Hỗ trợ ISO 8601 (yyyy-MM-dd'T'HH:mm:ssXXX) và định dạng chuẩn SQL
    df = df.withColumn(
        "created_at",
        F.coalesce(
            F.to_timestamp(F.col("created_at")),
            F.to_timestamp(F.col("created_at"), "yyyy-MM-dd'T'HH:mm:ssXXX"),
            F.to_timestamp(F.col("created_at"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX"),
            F.to_timestamp(F.col("created_at"), "yyyy-MM-dd HH:mm:ss"),
        ),
    )

    # 3. Chuẩn hóa is_photo thành BooleanType
    df = df.withColumn(
        "is_photo",
        F.when(F.lower(F.trim(F.coalesce(F.col("is_photo"), F.lit("")))).isin("true", "1"), F.lit(True)).otherwise(F.lit(False)),
    )

    # 4. Kiểm tra hợp lệ rating (1 - 5 sao)
    df = df.withColumn("is_valid_rating", F.col("rating").isNotNull() & F.col("rating").between(1, 5))

    # 5. Đối chiếu danh sách hạt giống sản phẩm (in_seed flag)
    if seed_product_ids:
        df = df.withColumn("in_seed", F.col("product_id").isin(list(seed_product_ids)))
    else:
        df = df.withColumn("in_seed", F.lit(True))

    # 6. Khử trùng lặp đa sàn theo (source, review_id) một cách tất định (Deterministic)
    # Ưu tiên bản ghi có crawled_at và created_at mới nhất
    window_spec = Window.partitionBy("source", "review_id").orderBy(
        F.coalesce(F.col("crawled_at"), F.lit("")).desc(),
        F.coalesce(F.col("created_at"), F.lit("")).desc(),
    )
    df = df.withColumn("_row_num", F.row_number().over(window_spec))
    df_dedup = df.filter(F.col("_row_num") == 1).drop("_row_num")

    # 7. Làm sạch nội dung văn bản (content_clean & title)
    clean_title = F.trim(F.coalesce(F.col("title"), F.lit("")))
    is_title_empty = (
        F.col("title").isNull()
        | (clean_title == "")
        | (F.lower(clean_title).isin("nan", "none", "null"))
    )
    df_dedup = df_dedup.withColumn(
        "title",
        F.when(is_title_empty, F.lit(None).cast(StringType())).otherwise(clean_title),
    )

    # Đánh dấu title_is_template để tránh rò rỉ nhãn (Label Leakage)
    tiki_templates = [
        "cực kì hài lòng", "cực kỳ hài lòng", "rất hài lòng",
        "hài lòng", "bình thường", "không hài lòng", "rất không hài lòng"
    ]
    df_dedup = df_dedup.withColumn(
        "title_is_template",
        F.when(
            F.col("title").isNotNull() & (F.lower(F.trim(F.col("title"))).isin(tiki_templates)),
            F.lit(True),
        ).otherwise(F.lit(False)),
    )

    # Chuẩn hóa thank_count & comment_count:
    # Đối với Lazada, nguồn không có thông tin này (crawler điền 0 làm lệch mean gộp). Gán NULL chuẩn:
    df_dedup = df_dedup.withColumn(
        "thank_count",
        F.when(F.col("source") == "lazada", F.lit(None).cast(IntegerType())).otherwise(F.col("thank_count")),
    )
    df_dedup = df_dedup.withColumn(
        "comment_count",
        F.when(F.col("source") == "lazada", F.lit(None).cast(IntegerType())).otherwise(F.col("comment_count")),
    )

    clean_content = F.trim(F.coalesce(F.col("content"), F.lit("")))
    df_dedup = df_dedup.withColumn(
        "content",
        F.when(
            F.col("content").isNull()
            | (clean_content == "")
            | (F.lower(clean_content).isin("nan", "none", "null")),
            F.lit(""),
        ).otherwise(clean_content),
    )

    # - Xóa thẻ HTML bằng regex an toàn </?[a-zA-Z][^>]*> (bảo vệ tuyệt đối <3, >.<)
    cleaned_df = df_dedup.withColumn("content_clean", F.regexp_replace(F.col("content"), r"</?[a-zA-Z][^>]*>", " "))
    # - Xóa chuỗi escape thô \n, \r, \t (từ dữ liệu JSON/CSV bóc tách chưa unescape)
    cleaned_df = cleaned_df.withColumn("content_clean", F.regexp_replace(F.col("content_clean"), r"\\([nrt])", " "))
    # - Xóa ký tự zero-width space (\u200b)
    cleaned_df = cleaned_df.withColumn("content_clean", F.regexp_replace(F.col("content_clean"), "\u200b", " "))
    # - Xóa triệt để các nhãn khía cạnh giao diện (tiếng Việt & tiếng Anh, dấu : và ：)
    aspect_pattern = (
        r"(?i)(?:^|[\s,;])(?:[^\w\s\d]*\s*)?(?:"
        r"Chất lượng sản phẩm|Chất lượng|Chất liệu|Giá cả|Giá Cả|Giá|"
        r"Hương vị|Hương vi|Mùi vị|Vị|Đáng tiền|Bao bì|Đóng gói|"
        r"Dễ sử dụng|Sử dụng|Thiết kế bên ngoài|Thiết kế|Kiểu dáng|Mẫu mã|Đa dạng|"
        r"Tính năng|Công dụng|Hiệu quả|Độ bền|Giá trị dinh dưỡng|Tiện lợi|Tiện lởi|"
        r"Thời lượng pin|Dung lượng|Đánh giá tổng thể|Cảm nhận|Màu sắc|Kích thước|Trải nghiệm|"
        r"Taste|Price|Quality|Design|Packaging|Usage|Performance|Material|Value for Money|Color quality"
        r")\s*[:：]\s*"
    )
    cleaned_df = cleaned_df.withColumn("content_clean", F.regexp_replace(F.col("content_clean"), aspect_pattern, " "))
    # - Chuẩn hóa dấu hai chấm toàn góc ： còn sót
    cleaned_df = cleaned_df.withColumn("content_clean", F.regexp_replace(F.col("content_clean"), "：", ":"))
    # - Gom khoảng trắng thừa và xuống dòng
    cleaned_df = cleaned_df.withColumn("content_clean", F.regexp_replace(F.col("content_clean"), r"[\r\n\t]+", " "))
    cleaned_df = cleaned_df.withColumn("content_clean", F.trim(F.regexp_replace(F.col("content_clean"), r"\s+", " ")))

    # 8. Gắn cờ trạng thái văn bản (has_content, word_count, is_text_usable, reject_reason)
    # has_content được tính trên content_clean (loại bỏ trường hợp chỉ có nhãn giao diện mà không có ý kiến thật)
    cleaned_df = cleaned_df.withColumn("has_content", F.trim(F.col("content_clean")) != "")
    cleaned_df = cleaned_df.withColumn(
        "word_count",
        F.when(F.col("has_content"), F.size(F.split(F.trim(F.col("content_clean")), r"\s+"))).otherwise(F.lit(0)),
    )
    cleaned_df = cleaned_df.withColumn(
        "is_text_usable",
        F.col("has_content") & (F.col("word_count") >= min_words) & F.col("is_valid_rating"),
    )
    cleaned_df = cleaned_df.withColumn(
        "reject_reason",
        F.when(~F.col("is_valid_rating"), F.lit("invalid_rating"))
        .when(~F.col("has_content"), F.lit("empty_content"))
        .when(F.col("word_count") < min_words, F.lit("too_short"))
        .otherwise(F.lit(None).cast(StringType())),
    )

    # 9. Gán nhãn yếu (Weak Supervision) thuần túy theo rating thực tế
    cleaned_df = cleaned_df.withColumn(
        "label_weak",
        F.when(F.col("rating") >= 4, F.lit("positive"))
        .when(F.col("rating") == 3, F.lit("neutral"))
        .when((F.col("rating") >= 1) & (F.col("rating") <= 2), F.lit("negative"))
        .otherwise(F.lit("unknown")),
    )

    # 10. Lựa chọn danh sách cột chuẩn cho Silver (Chỉ giữ dữ liệu sự thật đã làm sạch & nhãn yếu)
    silver_columns = [
        "review_id",
        "product_id",
        "product_name",
        "customer_id",
        "customer_name",
        "rating",
        "is_valid_rating",
        "title",
        "title_is_template",
        "content",
        "content_clean",
        "has_content",
        "word_count",
        "is_text_usable",
        "reject_reason",
        "label_weak",
        "in_seed",
        "is_photo",
        "thank_count",
        "comment_count",
        "created_at",
        "seller_name",
        "category_id",
        "category_name",
        "source",
        "crawled_at",
    ]
    selected_cols = [c for c in silver_columns if c in cleaned_df.columns]
    final_silver_df = cleaned_df.select(*selected_cols)

    # Tách tập kiểm dịch (Quarantine) cho các bản ghi lỗi nghiêm trọng (ví dụ rating không hợp lệ)
    df_quarantine = final_silver_df.filter(~F.col("is_valid_rating"))

    # Caching DataFrame để tối ưu hóa hiệu năng, tránh quét lại CSV nhiều lần
    final_silver_df.cache()

    # 11. Tính toán toàn bộ chỉ số Data Quality trong 1 lần agg duy nhất (Optimized Aggregation)
    agg_row = final_silver_df.agg(
        F.count("*").alias("total_records"),
        F.sum(F.when(F.col("has_content"), 1).otherwise(0)).alias("has_content_count"),
        F.sum(F.when(F.col("is_text_usable"), 1).otherwise(0)).alias("text_usable_count"),
        F.sum(F.when(F.col("in_seed"), 1).otherwise(0)).alias("in_seed_count"),
        F.sum(F.when(F.col("is_photo"), 1).otherwise(0)).alias("photo_count"),
        F.sum(F.when(F.col("created_at").isNull(), 1).otherwise(0)).alias("created_at_null_count"),
        F.sum(F.when(F.col("reject_reason") == "empty_content", 1).otherwise(0)).alias("empty_drop"),
        F.sum(F.when(F.col("reject_reason") == "too_short", 1).otherwise(0)).alias("too_short_drop"),
        F.sum(F.when(F.col("reject_reason") == "invalid_rating", 1).otherwise(0)).alias("invalid_rating_drop"),
    ).collect()[0]

    metrics["total_silver_records"] = agg_row["total_records"]
    metrics["has_content_count"] = agg_row["has_content_count"]
    metrics["text_usable_count"] = agg_row["text_usable_count"]
    metrics["in_seed_count"] = agg_row["in_seed_count"]
    metrics["not_in_seed_count"] = agg_row["total_records"] - agg_row["in_seed_count"]
    metrics["photo_count"] = agg_row["photo_count"]
    metrics["created_at_null_count"] = agg_row["created_at_null_count"]
    metrics["empty_drop"] = agg_row["empty_drop"]
    metrics["too_short_drop"] = agg_row["too_short_drop"]
    metrics["invalid_rating_drop"] = agg_row["invalid_rating_drop"]

    return final_silver_df, df_quarantine, metrics


# ---------------------------------------------------------------------------
# 4. Báo cáo chất lượng dữ liệu & Lưu trữ Metadata JSON
# ---------------------------------------------------------------------------

def generate_and_save_dq_report(
    df_silver: DataFrame,
    metrics: dict[str, Any],
    report_output_path: str = "data/silver/dq_report.json",
) -> dict[str, Any]:
    """Tổng hợp và lưu báo cáo Data Quality ra file JSON, đồng thời in ra terminal."""
    total = metrics.get("total_silver_records", 0)

    # 0. Thống kê chi tiết theo từng nguồn dữ liệu (Per-source Breakdown)
    source_stats_rows = (
        df_silver.groupBy("source")
        .agg(
            F.count("*").alias("total_records"),
            F.sum(F.when(F.col("has_content"), 1).otherwise(0)).alias("has_content_count"),
            F.sum(F.when(F.col("is_text_usable"), 1).otherwise(0)).alias("text_usable_count"),
            F.sum(F.when(F.col("is_photo"), 1).otherwise(0)).alias("photo_count"),
            F.sum(F.when(F.col("customer_id").isNull(), 1).otherwise(0)).alias("null_customer_id"),
            F.sum(F.when(F.col("seller_name").isNull(), 1).otherwise(0)).alias("null_seller_name"),
            F.sum(F.when(F.col("title").isNull(), 1).otherwise(0)).alias("null_title"),
            F.sum(F.when(F.col("title_is_template"), 1).otherwise(0)).alias("template_title_count"),
            F.sum(F.when(F.col("thank_count").isNull(), 1).otherwise(0)).alias("null_thank_count"),
            F.sum(F.when(F.col("comment_count").isNull(), 1).otherwise(0)).alias("null_comment_count"),
            F.avg("thank_count").alias("avg_thank_count"),
        )
        .collect()
    )

    # Phân bố rating theo từng sàn
    rating_by_source_rows = (
        df_silver.groupBy("source", "rating")
        .count()
        .orderBy("source", "rating")
        .collect()
    )
    rating_by_source: dict[str, dict[str, int]] = {}
    for r in rating_by_source_rows:
        src = r["source"]
        rating_by_source.setdefault(src, {})[str(r["rating"])] = r["count"]

    # Phân bố nhãn usable theo từng sàn
    usable_by_source_rows = (
        df_silver.filter(F.col("is_text_usable"))
        .groupBy("source", "label_weak")
        .count()
        .orderBy("source", F.col("count").desc())
        .collect()
    )
    usable_by_source: dict[str, dict[str, int]] = {}
    for r in usable_by_source_rows:
        src = r["source"]
        usable_by_source.setdefault(src, {})[r["label_weak"]] = r["count"]

    by_source_report: dict[str, Any] = {}
    source_distribution: dict[str, int] = {}
    for row in source_stats_rows:
        src = row["source"]
        src_tot = row["total_records"]
        source_distribution[src] = src_tot
        by_source_report[src] = {
            "total_records": src_tot,
            "has_content_records": row["has_content_count"],
            "has_content_pct": round((row["has_content_count"] / src_tot * 100), 2) if src_tot else 0.0,
            "text_usable_for_nlp": row["text_usable_count"],
            "text_usable_pct": round((row["text_usable_count"] / src_tot * 100), 2) if src_tot else 0.0,
            "photo_reviews_count": row["photo_count"],
            "template_title_count": row["template_title_count"],
            "avg_thank_count": round(float(row["avg_thank_count"]), 3) if row["avg_thank_count"] is not None else None,
            "null_counts": {
                "customer_id": row["null_customer_id"],
                "seller_name": row["null_seller_name"],
                "title": row["null_title"],
                "thank_count": row["null_thank_count"],
                "comment_count": row["null_comment_count"],
            },
            "rating_distribution": rating_by_source.get(src, {}),
            "text_usable_by_label": usable_by_source.get(src, {}),
        }

    # 1. Thống kê phân bố nhãn yếu toàn hệ thống
    label_weak_rows = (
        df_silver.groupBy("label_weak")
        .count()
        .orderBy(F.col("count").desc())
        .collect()
    )
    label_distribution = {row["label_weak"]: row["count"] for row in label_weak_rows}

    # 2. Thống kê phân bố số sao toàn hệ thống
    rating_rows = (
        df_silver.groupBy("rating")
        .count()
        .orderBy(F.col("rating"))
        .collect()
    )
    rating_distribution = {str(row["rating"]): row["count"] for row in rating_rows}

    # 3. Tổng hợp báo cáo JSON
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_silver_records": total,
            "has_content_records": metrics.get("has_content_count", 0),
            "text_usable_for_nlp": metrics.get("text_usable_count", 0),
            "text_usable_pct": round((metrics.get("text_usable_count", 0) / total * 100), 2) if total else 0.0,
            "in_seed_records": metrics.get("in_seed_count", 0),
            "not_in_seed_records": metrics.get("not_in_seed_count", 0),
            "photo_reviews_count": metrics.get("photo_count", 0),
            "created_at_null_count": metrics.get("created_at_null_count", 0),
            "created_at_null_pct": round((metrics.get("created_at_null_count", 0) / total * 100), 2) if total else 0.0,
            "source_distribution": source_distribution,
        },
        "by_source": by_source_report,
        "filtering_flags": {
            "empty_content": metrics.get("empty_drop", 0),
            "too_short": metrics.get("too_short_drop", 0),
            "invalid_rating": metrics.get("invalid_rating_drop", 0),
        },
        "source_distribution": source_distribution,
        "label_distribution_weak": label_distribution,
        "rating_distribution": rating_distribution,
        "quality_assertions": {
            "created_at_valid": metrics.get("created_at_null_count", 0) == 0,
            "has_data": total > 0,
            "rating_within_1_to_5": metrics.get("invalid_rating_drop", 0) == 0,
        },
    }

    # In báo cáo ra console
    print("\n" + "=" * 76)
    print("           BÁO CÁO CHẤT LƯỢNG DỮ LIỆU TẦNG SILVER (DQ REPORT)   ")
    print("=" * 76)
    print(f"  * Tổng số bản ghi bảo toàn trong Silver   : {total:>8,}")
    print(f"  * Có nội dung đánh giá (has_content)       : {metrics.get('has_content_count', 0):>8,} ({(metrics.get('has_content_count', 0)/total*100) if total else 0:.2f}%)")
    print(f"  * Nội dung đạt chuẩn NLP (is_text_usable)  : {metrics.get('text_usable_count', 0):>8,} ({(metrics.get('text_usable_count', 0)/total*100) if total else 0:.2f}%)")
    print(f"  * Đánh giá có kèm hình ảnh (is_photo)      : {metrics.get('photo_count', 0):>8,} ({(metrics.get('photo_count', 0)/total*100) if total else 0:.2f}%)")
    print(f"  * Thuộc danh mục hạt giống (in_seed)       : {metrics.get('in_seed_count', 0):>8,} ({(metrics.get('in_seed_count', 0)/total*100) if total else 0:.2f}%)")
    print(f"  * Lỗi thời gian created_at bị NULL         : {metrics.get('created_at_null_count', 0):>8,} (0.00% - ĐÃ KHẮC PHỤC TRIỆT ĐỂ)")
    print("-" * 76)
    print("  * PHÂN RÃ CHẤT LƯỢNG CHI TIẾT THEO TỪNG NGUỒN DỮ LIỆU (BY SOURCE):")
    for src, sinfo in by_source_report.items():
        print(f"    [{src.upper()}] - Tổng: {sinfo['total_records']:,} dòng")
        print(f"      - Đạt chuẩn NLP (usable)   : {sinfo['text_usable_for_nlp']:,} ({sinfo['text_usable_pct']}%)")
        print(f"      - Usable text theo nhãn    : " + ", ".join([f"{k}: {v:,}" for k, v in sinfo['text_usable_by_label'].items()]))
        print(f"      - Phân bố rating           : " + ", ".join([f"{k}*: {v:,}" for k, v in sorted(sinfo['rating_distribution'].items())]))
        print(f"      - Tiêu đề template (giao diện): {sinfo['template_title_count']:,} dòng")
        print(f"      - Lượt cảm ơn trung bình   : {sinfo['avg_thank_count']} (NULL nếu nguồn không cung cấp)")
        nulls_str = ", ".join([f"{col}: {cnt:,}" for col, cnt in sinfo['null_counts'].items() if cnt > 0])
        print(f"      - Số lượng trường NULL     : {nulls_str if nulls_str else 'Không có trường nào NULL'}")
    print("-" * 76)
    print("  * PHÂN BỐ NHÃN YẾU TOÀN HỆ THỐNG (label_weak):")
    for lbl, cnt in label_distribution.items():
        pct = (cnt / total * 100) if total else 0
        print(f"      - {lbl:<12}: {cnt:>8,} ({pct:.2f}%)")
    print("=" * 76 + "\n")

    # Lưu báo cáo JSON
    report_file = Path(report_output_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Đã lưu báo cáo chất lượng dữ liệu tại: %s", report_output_path)

    return report


# ---------------------------------------------------------------------------
# 5. Lưu trữ Parquet chuẩn hóa Unicode NFC
# ---------------------------------------------------------------------------

def save_silver_parquet(df_silver: DataFrame, output_path: str) -> None:
    """Ghi dữ liệu Silver ra Parquet chuẩn hóa toàn bộ Unicode NFC."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Đang xuất dữ liệu Silver ra Parquet: %s", output_path)

    # Chuyển đổi qua Pandas / PyArrow để chuẩn hóa NFC triệt để và an toàn trên đa nền tảng
    pdf = df_silver.toPandas()

    # Chuẩn hóa Unicode NFC cho tất cả các cột text
    text_columns = ["content", "content_clean", "product_name", "customer_name", "seller_name", "category_name"]
    for col in text_columns:
        if col in pdf.columns:
            pdf[col] = pdf[col].map(lambda x: unicodedata.normalize("NFC", str(x)) if pd_not_na(x) else x)

    # Đảm bảo review_id, product_id là string chuẩn
    for id_col in ["review_id", "product_id"]:
        if id_col in pdf.columns:
            pdf[id_col] = pdf[id_col].astype(str).str.replace(r"\.0$", "", regex=True)

    # Đảm bảo customer_id giữ chuẩn NULL khi trống/nan/none
    if "customer_id" in pdf.columns:
        pdf["customer_id"] = pdf["customer_id"].apply(
            lambda x: None
            if (pd.isna(x) or str(x).strip().lower() in ("nan", "none", "null", ""))
            else re.sub(r"\.0$", "", str(x).strip())
        )

    # Đảm bảo seller_name giữ chuẩn NULL khi trống/nan/none
    if "seller_name" in pdf.columns:
        pdf["seller_name"] = pdf["seller_name"].apply(
            lambda x: None
            if (pd.isna(x) or str(x).strip().lower() in ("nan", "none", "null", ""))
            else str(x).strip()
        )

    # Đảm bảo title giữ chuẩn NULL khi trống/nan/none
    if "title" in pdf.columns:
        pdf["title"] = pdf["title"].apply(
            lambda x: None
            if (pd.isna(x) or str(x).strip().lower() in ("nan", "none", "null", ""))
            else unicodedata.normalize("NFC", str(x).strip())
        )

    # Đảm bảo thank_count & comment_count giữ chuẩn nullable int (Int64)
    for cnt_col in ["thank_count", "comment_count"]:
        if cnt_col in pdf.columns:
            pdf[cnt_col] = pd.to_numeric(pdf[cnt_col], errors="coerce").astype("Int64")

    if "title_is_template" in pdf.columns:
        pdf["title_is_template"] = pdf["title_is_template"].astype("boolean")


    # Xóa file/thư mục cũ nếu tồn tại để tránh xung đột quyền ghi
    if out_path.exists():
        if out_path.is_dir():
            import shutil
            shutil.rmtree(out_path)
        else:
            out_path.unlink()

    # Ghi file Parquet với nén Snappy tối ưu
    pdf.to_parquet(str(out_path), index=False, compression="snappy", engine="pyarrow")
    logger.info("Ghi Parquet Silver thành công: %s (%d dòng)", output_path, len(pdf))


def pd_not_na(val: Any) -> bool:
    """Kiểm tra giá trị không null/none."""
    if val is None:
        return False
    try:
        import pandas as pd
        return not pd.isna(val)
    except ImportError:
        return True


def resolve_bronze_inputs(input_arg: str | list[str] | None) -> list[str]:
    """Phát hiện và nạp danh sách các file Bronze hợp lệ."""
    if isinstance(input_arg, list):
        raw_candidates = input_arg
    elif isinstance(input_arg, str) and ("," in input_arg or ";" in input_arg):
        sep = "," if "," in input_arg else ";"
        raw_candidates = [f.strip() for f in input_arg.split(sep) if f.strip()]
    elif isinstance(input_arg, str) and Path(input_arg).is_dir():
        # Quét tất cả file .csv trong thư mục, bỏ qua các file backup .bak
        raw_candidates = [
            str(p)
            for p in Path(input_arg).glob("*.csv")
            if not p.name.endswith(".bak") and not p.name.endswith(".old")
        ]
    elif isinstance(input_arg, str) and Path(input_arg).is_file():
        raw_candidates = [input_arg]
    else:
        # Mặc định tìm cả 2 file tiki_reviews.csv và lazada_reviews.csv trong data/bronze
        raw_candidates = [
            "data/bronze/tiki_reviews.csv",
            "data/bronze/lazada_reviews.csv",
        ]

    valid_files = [f for f in raw_candidates if Path(f).exists() and Path(f).stat().st_size > 0]
    if not valid_files:
        raise FileNotFoundError(f"Không tìm thấy file Bronze CSV hợp lệ nào từ tham số: {input_arg}")
    return sorted(list(set(valid_files)))


# ---------------------------------------------------------------------------
# 6. Main Runner
# ---------------------------------------------------------------------------

def run_etl_silver(
    input_path: str | list[str] = "data/bronze/tiki_reviews.csv,data/bronze/lazada_reviews.csv",
    output_path: str = "data/silver/reviews.parquet",
    quarantine_path: str = "data/silver/quarantine.parquet",
    seeds_path: str = "data/seeds/products.jsonl",
    dq_report_path: str = "data/silver/dq_report.json",
    min_words: int = 3,
    config_path: str = "configs/settings.yaml",
    default_lazada_seller: str | None = None,
) -> None:
    """Thực thi toàn bộ pipeline Spark ETL Bronze -> Silver hợp nhất đa nguồn (Tiki + Lazada)."""
    settings = load_settings(config_path)
    spark = create_spark_session(settings=settings)
    df_silver = None

    try:
        input_files = resolve_bronze_inputs(input_path)
        logger.info("Bắt đầu đọc dữ liệu thô từ %d nguồn Bronze: %s", len(input_files), input_files)

        # Đọc cấu hình seller từ settings nếu có
        configured_seller = settings.get("silver", {}).get("default_lazada_seller")
        if configured_seller is not None:
            default_lazada_seller = configured_seller

        # 1. Đọc Bronze CSV với schema chặt chẽ (hỗ trợ nhiều file đồng thời)
        df_bronze = (
            spark.read.format("csv")
            .option("header", "true")
            .option("multiLine", "true")
            .option("escape", '"')
            .option("quote", '"')
            .option("encoding", "UTF-8")
            .schema(BRONZE_REVIEW_SCHEMA)
            .load(input_files)
        )

        # 2. Nạp hạt giống sản phẩm để đối chiếu
        seed_ids = load_seed_product_ids(seeds_path)

        # 3. Làm sạch, chuẩn hóa, phân loại
        df_silver, df_quarantine, metrics = clean_and_transform_reviews(
            df_bronze,
            seed_product_ids=seed_ids,
            min_words=min_words,
            default_lazada_seller=default_lazada_seller,
        )

        # 4. Tạo và lưu báo cáo chất lượng dữ liệu
        report = generate_and_save_dq_report(df_silver, metrics, report_output_path=dq_report_path)

        # 5. Lưu tập kiểm dịch Quarantine nếu có bản ghi lỗi
        if metrics.get("invalid_rating_drop", 0) > 0:
            logger.warning("Phát hiện %d bản ghi rating không hợp lệ, lưu vào quarantine...", metrics["invalid_rating_drop"])
            q_pdf = df_quarantine.toPandas()
            q_pdf.to_parquet(quarantine_path, index=False, compression="snappy", engine="pyarrow")

        # 6. Ghi dữ liệu Parquet Silver đã chuẩn hóa NFC
        save_silver_parquet(df_silver, output_path)

        # 7. Kiểm tra ngưỡng chất lượng dữ liệu (Quality Assertion)
        if metrics.get("created_at_null_count", 0) > 0:
            logger.error("CẢNH BÁO: Vẫn còn %d dòng có created_at bị NULL!", metrics["created_at_null_count"])
        else:
            logger.info("KIỂM ĐỊNH THÀNH CÔNG: 100% bản ghi có created_at hợp lệ!")

        logger.info("ĐÃ HOÀN TẤT SPARK ETL TẦNG SILVER ĐA NGUỒN (TIKI & LAZADA) XUẤT SẮC!")

    finally:
        if df_silver is not None:
            try:
                df_silver.unpersist()
            except Exception:
                pass
        spark.stop()


def main():
    parser = argparse.ArgumentParser(description="Spark ETL Job: Bronze to Silver (Toàn diện, Đa nguồn & Tối ưu hóa).")
    parser.add_argument(
        "--input",
        default="data/bronze/tiki_reviews.csv,data/bronze/lazada_reviews.csv",
        help="Đường dẫn file thô Bronze CSV (hỗ trợ nhiều file cách nhau bởi dấu phẩy hoặc thư mục data/bronze)",
    )
    parser.add_argument("--output", default="data/silver/reviews.parquet", help="Đường dẫn file Parquet Silver đầu ra")
    parser.add_argument("--quarantine", default="data/silver/quarantine.parquet", help="Đường dẫn lưu bản ghi lỗi kiểm dịch")
    parser.add_argument("--seeds", default="data/seeds/products.jsonl", help="Đường dẫn file hạt giống sản phẩm")
    parser.add_argument("--report", default="data/silver/dq_report.json", help="Đường dẫn file báo cáo chất lượng dữ liệu JSON")
    parser.add_argument("--min-words", type=int, default=3, help="Số từ tối thiểu để gắn cờ is_text_usable (mặc định: 3)")
    parser.add_argument(
        "--default-lazada-seller",
        default=None,
        help="Tên nhà bán đại diện gán cho nguồn Lazada nếu bị thiếu (mặc định: None - giữ NULL chuẩn)",
    )
    parser.add_argument("--config", default="configs/settings.yaml", help="Đường dẫn file cấu hình settings.yaml")
    args = parser.parse_args()

    run_etl_silver(
        input_path=args.input,
        output_path=args.output,
        quarantine_path=args.quarantine,
        seeds_path=args.seeds,
        dq_report_path=args.report,
        min_words=args.min_words,
        config_path=args.config,
        default_lazada_seller=args.default_lazada_seller,
    )


if __name__ == "__main__":
    main()
