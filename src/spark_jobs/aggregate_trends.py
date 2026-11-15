"""
Spark Aggregation Job: Tổng hợp Xu hướng Cảm xúc Đa chiều (Tầng Gold -> Marts & MongoDB)
=======================================================================================
Thực hiện các phép tổng hợp phân tán bằng Apache Spark `groupBy().agg()` theo đúng thiết kế
mục 4.5 và Tuần 4 trong IMPLEMENTATION_PLAN.md:

1. Theo sản phẩm (`trends_by_product`):
   - Tỷ lệ % tích cực / trung lập / tiêu cực mỗi sản phẩm (trên toàn bộ review & trên review có chữ).
   - Tỷ lệ mâu thuẫn sao vs văn bản (`conflict_pct`, `high_star_negative_count`).
   - Phát hiện sản phẩm bị đánh giá tiêu cực bất thường (`is_anomaly_negative`).
2. Theo thời gian (`trends_by_time`):
   - Tổng hợp theo tháng (`month`: yyyy-MM) và theo tuần (`week`: yyyy-Www), tách theo từng sàn và toàn hệ thống.
   - Phát hiện các thời điểm bất thường có tỷ lệ tiêu cực tăng đột biến (`is_anomaly_spike`).
3. Theo nhóm khách hàng (`trends_by_customer`):
   - Nhóm theo hành vi đăng ảnh (`is_photo`), tần suất mua/đánh giá lặp lại (`customer_id`), và nhà bán (`seller_name`).
4. Theo phân khúc giá (`trends_by_price_tier`):
   - Join `gold_reviews` với bảng chiều sản phẩm `seeds/products.jsonl` theo `(source, product_id)`.
   - Phân nhóm giá: Bình dân (< 200k), Tầm trung (200k – 1.5M), Cao cấp (> 1.5M) để kiểm chứng
     giả thuyết khách hàng ở phân khúc giá cao khắt khe hơn.
5. Theo ngành hàng & mâu thuẫn sao vs chữ (`trends_by_category`):
   - Chuẩn hóa nhóm ngành hàng liên sàn (`category_group`) giữa Tiki và Lazada.
   - Thống kê tỷ lệ đánh giá sao cao (4–5★) nhưng nội dung tiêu cực theo từng ngành hàng.

Lưu trữ kết quả:
- Xuất các bảng Parquet tại `data/gold/trends_by_*.parquet` và báo cáo `data/gold/trends_summary_report.json`.
- Đồng bộ vào các collection riêng biệt trong MongoDB (`ecommerce_sentiment`) phục vụ Dashboard truy vấn tức thì.
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
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

# Tự động cấu hình JAVA_HOME trỏ tới Java 11 (tương thích PySpark 3.5 trên Windows)
JAVA_11_CANDIDATES = [
    r"C:\Program Files\Eclipse Adoptium\jdk-11.0.32.101-hotspot",
    r"C:\Program Files\Eclipse Adoptium\jdk-11.0.32.1-hotspot",
    r"C:\Program Files\Java\jdk-11",
]
for _java_candidate in JAVA_11_CANDIDATES:
    if os.path.exists(_java_candidate):
        os.environ["JAVA_HOME"] = _java_candidate
        break

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

HADOOP_DIR = ROOT_DIR / "hadoop"
if (HADOOP_DIR / "bin" / "winutils.exe").exists():
    os.environ["HADOOP_HOME"] = str(HADOOP_DIR)
    hadoop_bin_str = str(HADOOP_DIR / "bin")
    if hadoop_bin_str not in os.environ.get("PATH", ""):
        os.environ["PATH"] = hadoop_bin_str + ";" + os.environ.get("PATH", "")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AggregateTrends")


def load_settings(config_path: str | Path = "configs/settings.yaml") -> Dict[str, Any]:
    cfg_file = Path(config_path)
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_spark_session(settings: Dict[str, Any] | None = None) -> SparkSession:
    """Khởi tạo SparkSession cho tác vụ Aggregation tầng Gold."""
    spark_cfg = (settings or {}).get("spark", {})
    driver_mem = spark_cfg.get("driver_memory", "4g")
    executor_mem = spark_cfg.get("executor_memory", "4g")

    spark_tmp = ROOT_DIR / "tmp" / "spark"
    spark_tmp.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName("SentimentAnalysis_AggregateTrends")
        .master(spark_cfg.get("master", "local[*]"))
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", executor_mem)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.local.dir", str(spark_tmp))
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_enriched_gold_dataframe(
    spark: SparkSession,
    gold_path: str | Path = "data/gold/gold_reviews.parquet",
    seeds_path: str | Path = "data/seeds/products.jsonl",
) -> DataFrame:
    """Nạp bảng Fact `gold_reviews.parquet`, join với bảng Dimension `products.jsonl`
    và chuẩn hóa các trường `price_tier`, `category_group`, `discount_rate`.
    """
    g_path = Path(gold_path)
    s_path = Path(seeds_path)
    if not g_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu tầng Gold tại: {g_path.resolve()}")
    if not s_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu hạt giống sản phẩm tại: {s_path.resolve()}")

    logger.info("Đang nạp dữ liệu Fact từ: %s", g_path)
    df_gold = spark.read.parquet(str(g_path))

    logger.info("Đang nạp dữ liệu Dimension sản phẩm từ: %s", s_path)
    df_seeds_raw = spark.read.json(str(s_path))
    df_seeds = (
        df_seeds_raw.select(
            F.lower(F.trim(F.col("source").cast("string"))).alias("source"),
            F.trim(F.col("product_id").cast("string")).alias("product_id"),
            F.coalesce(F.col("price").cast("double"), F.lit(0.0)).alias("price"),
            F.coalesce(F.col("original_price").cast("double"), F.col("price").cast("double"), F.lit(0.0)).alias(
                "original_price"
            ),
            F.coalesce(F.col("rating_average").cast("double"), F.lit(0.0)).alias("seed_rating_avg"),
            F.coalesce(F.col("quantity_sold").cast("int"), F.lit(0)).alias("quantity_sold"),
        )
        .dropDuplicates(["source", "product_id"])
    )

    # Join Fact (Reviews) với Dimension (Products)
    df = df_gold.join(F.broadcast(df_seeds), on=["source", "product_id"], how="left")

    # Phân nhóm phân khúc giá (Price Tier) theo đề cương:
    # - Bình dân (< 200k): price > 0 và price < 200,000
    # - Tầm trung (200k - 1.5M): 200,000 <= price <= 1,500,000
    # - Cao cấp (> 1.5M): price > 1,500,000
    df = df.withColumn(
        "price_tier",
        F.when((F.col("price").isNull()) | (F.col("price") <= 0), F.lit("Chưa xác định"))
        .when(F.col("price") < 200000, F.lit("Bình dân (< 200k)"))
        .when(F.col("price") <= 1500000, F.lit("Tầm trung (200k - 1.5M)"))
        .otherwise(F.lit("Cao cấp (> 1.5M)")),
    ).withColumn(
        "price_tier_order",
        F.when(F.col("price_tier") == "Bình dân (< 200k)", F.lit(1))
        .when(F.col("price_tier") == "Tầm trung (200k - 1.5M)", F.lit(2))
        .when(F.col("price_tier") == "Cao cấp (> 1.5M)", F.lit(3))
        .otherwise(F.lit(4)),
    )

    # Tính tỷ lệ giảm giá (%)
    df = df.withColumn(
        "discount_pct",
        F.when(
            (F.col("original_price").isNotNull())
            & (F.col("original_price") > F.col("price"))
            & (F.col("original_price") > 0),
            F.round((F.col("original_price") - F.col("price")) / F.col("original_price") * 100.0, 1),
        ).otherwise(F.lit(0.0)),
    )

    # Chuẩn hóa nhóm ngành hàng liên sàn (category_group) giữa Tiki và Lazada
    df = df.withColumn(
        "category_group",
        F.when(
            F.col("category_name").isin("Điện Gia Dụng", "Đồ Gia Dụng"),
            F.lit("Điện & Đồ Gia Dụng"),
        )
        .when(
            F.col("category_name").isin("Nhà Sách Tiki", "Dụng Cụ Học Tập & Văn Phòng Phẩm"),
            F.lit("Sách & Văn Phòng Phẩm"),
        )
        .otherwise(F.coalesce(F.col("category_name"), F.lit("Khác"))),
    )

    # Chuẩn bị các cột cờ số học phục vụ Spark groupBy().agg() nhanh gọn
    df = (
        df.withColumn("_is_pos", F.when(F.col("label_final") == "positive", 1).otherwise(0))
        .withColumn("_is_neu", F.when(F.col("label_final") == "neutral", 1).otherwise(0))
        .withColumn("_is_neg", F.when(F.col("label_final") == "negative", 1).otherwise(0))
        .withColumn("_is_usable", F.when(F.col("is_text_usable") == True, 1).otherwise(0))
        .withColumn(
            "_is_text_pos",
            F.when((F.col("is_text_usable") == True) & (F.col("label_final") == "positive"), 1).otherwise(0),
        )
        .withColumn(
            "_is_text_neu",
            F.when((F.col("is_text_usable") == True) & (F.col("label_final") == "neutral"), 1).otherwise(0),
        )
        .withColumn(
            "_is_text_neg",
            F.when((F.col("is_text_usable") == True) & (F.col("label_final") == "negative"), 1).otherwise(0),
        )
        .withColumn("_is_conflict", F.when(F.col("is_conflict") == True, 1).otherwise(0))
        .withColumn(
            "_is_high_star_neg",
            F.when((F.col("rating") >= 4) & (F.col("label_final") == "negative"), 1).otherwise(0),
        )
        .withColumn("_is_photo", F.when(F.col("is_photo") == True, 1).otherwise(0))
    )

    return df


def _standard_sentiment_aggs() -> list[Any]:
    """Tập hợp các biểu thức tổng hợp Spark chuẩn dùng chung cho các chiều phân tích."""
    return [
        F.count("*").alias("total_reviews"),
        F.sum("_is_usable").alias("usable_text_reviews"),
        F.round(F.avg("rating"), 2).alias("avg_rating"),
        F.sum("_is_pos").alias("positive_count"),
        F.sum("_is_neu").alias("neutral_count"),
        F.sum("_is_neg").alias("negative_count"),
        F.round(F.sum("_is_pos") * 100.0 / F.count("*"), 2).alias("positive_pct"),
        F.round(F.sum("_is_neu") * 100.0 / F.count("*"), 2).alias("neutral_pct"),
        F.round(F.sum("_is_neg") * 100.0 / F.count("*"), 2).alias("negative_pct"),
        F.round(
            F.when(F.sum("_is_usable") > 0, F.sum("_is_text_pos") * 100.0 / F.sum("_is_usable")).otherwise(0.0),
            2,
        ).alias("text_positive_pct"),
        F.round(
            F.when(F.sum("_is_usable") > 0, F.sum("_is_text_neu") * 100.0 / F.sum("_is_usable")).otherwise(0.0),
            2,
        ).alias("text_neutral_pct"),
        F.round(
            F.when(F.sum("_is_usable") > 0, F.sum("_is_text_neg") * 100.0 / F.sum("_is_usable")).otherwise(0.0),
            2,
        ).alias("text_negative_pct"),
        F.sum("_is_conflict").alias("conflict_count"),
        F.round(
            F.when(F.sum("_is_usable") > 0, F.sum("_is_conflict") * 100.0 / F.sum("_is_usable")).otherwise(0.0),
            2,
        ).alias("conflict_pct"),
        F.sum("_is_high_star_neg").alias("high_star_negative_count"),
        F.round(
            F.when(F.sum("_is_usable") > 0, F.sum("_is_high_star_neg") * 100.0 / F.sum("_is_usable")).otherwise(0.0),
            2,
        ).alias("high_star_negative_pct"),
        F.sum("_is_photo").alias("photo_review_count"),
        F.round(F.sum("_is_photo") * 100.0 / F.count("*"), 2).alias("photo_review_pct"),
    ]


def aggregate_by_product(df: DataFrame) -> pd.DataFrame:
    """1. Tổng hợp xu hướng cảm xúc theo từng sản phẩm bằng Spark groupBy().agg()."""
    logger.info("1. Đang tổng hợp xu hướng cảm xúc theo Sản phẩm (trends_by_product)...")
    agg_df = (
        df.groupBy(
            "source",
            "product_id",
            "category_id",
            "category_name",
            "category_group",
            "price",
            "original_price",
            "discount_pct",
            "price_tier",
            "quantity_sold",
        )
        .agg(
            F.first("product_name", ignorenulls=True).alias("product_name"),
            *_standard_sentiment_aggs(),
        )
        .withColumn(
            "is_anomaly_negative",
            (F.col("total_reviews") >= 20)
            & ((F.col("negative_pct") >= 10.0) | (F.col("text_negative_pct") >= 20.0)),
        )
        .orderBy(F.col("total_reviews").desc(), F.col("negative_count").desc())
    )
    pdf = agg_df.toPandas()
    # Sắp xếp lại cột cho đẹp
    front_cols = [
        "source",
        "product_id",
        "product_name",
        "category_name",
        "category_group",
        "price",
        "original_price",
        "discount_pct",
        "price_tier",
        "quantity_sold",
    ]
    other_cols = [c for c in pdf.columns if c not in front_cols]
    return pdf[front_cols + other_cols]


def aggregate_by_time(df: DataFrame) -> pd.DataFrame:
    """2. Tổng hợp xu hướng cảm xúc theo Thời gian (Tháng & Tuần, theo sàn & toàn bộ)."""
    logger.info("2. Đang tổng hợp xu hướng cảm xúc theo Thời gian (trends_by_time)...")
    df_time = (
        df.filter(F.col("created_at").isNotNull())
        .withColumn("period_month", F.date_format(F.col("created_at"), "yyyy-MM"))
        .withColumn(
            "period_week",
            F.concat_ws(
                "-W",
                F.year(F.date_trunc("week", F.col("created_at"))).cast("string"),
                F.lpad(F.weekofyear(F.col("created_at")).cast("string"), 2, "0"),
            ),
        )
    )

    # 2.1 Theo tháng - Từng sàn
    m_by_source = (
        df_time.groupBy(
            F.lit("month").alias("granularity"),
            F.col("period_month").alias("period"),
            F.col("source"),
        ).agg(*_standard_sentiment_aggs())
    )
    # 2.2 Theo tháng - Toàn hệ thống ('all')
    m_all = (
        df_time.groupBy(
            F.lit("month").alias("granularity"),
            F.col("period_month").alias("period"),
            F.lit("all").alias("source"),
        ).agg(*_standard_sentiment_aggs())
    )
    # 2.3 Theo tuần - Từng sàn
    w_by_source = (
        df_time.groupBy(
            F.lit("week").alias("granularity"),
            F.col("period_week").alias("period"),
            F.col("source"),
        ).agg(*_standard_sentiment_aggs())
    )
    # 2.4 Theo tuần - Toàn hệ thống ('all')
    w_all = (
        df_time.groupBy(
            F.lit("week").alias("granularity"),
            F.col("period_week").alias("period"),
            F.lit("all").alias("source"),
        ).agg(*_standard_sentiment_aggs())
    )

    unioned = m_by_source.unionByName(m_all).unionByName(w_by_source).unionByName(w_all)
    pdf = unioned.orderBy("granularity", "source", "period").toPandas()

    # Phát hiện điểm bất thường (Anomaly Spike) theo từng nhóm (granularity, source)
    pdf["is_anomaly_spike"] = False
    for (gran, src), grp in pdf.groupby(["granularity", "source"]):
        valid_grp = grp[grp["total_reviews"] >= 50]
        if len(valid_grp) >= 5:
            mean_neg = valid_grp["negative_pct"].mean()
            std_neg = valid_grp["negative_pct"].std()
            threshold = max(mean_neg + 1.5 * std_neg, 6.0)
            spike_idx = valid_grp[valid_grp["negative_pct"] >= threshold].index
            pdf.loc[spike_idx, "is_anomaly_spike"] = True

    return pdf


def aggregate_by_customer(df: DataFrame) -> pd.DataFrame:
    """3. Tổng hợp xu hướng cảm xúc theo Nhóm khách hàng (is_photo, tần suất mua lặp lại, top seller)."""
    logger.info("3. Đang tổng hợp xu hướng cảm xúc theo Nhóm khách hàng (trends_by_customer)...")

    # 3.1 Nhóm theo hành vi đăng ảnh (photo_behavior)
    df_photo = df.withColumn(
        "segment_value",
        F.when(F.col("is_photo") == True, F.lit("Có kèm hình ảnh (Photo Review)")).otherwise(
            F.lit("Chỉ văn bản / sao (No Photo)")
        ),
    )
    photo_by_src = df_photo.groupBy(
        F.lit("photo_behavior").alias("segment_type"),
        F.col("segment_value"),
        F.col("source"),
    ).agg(*_standard_sentiment_aggs())
    photo_all = df_photo.groupBy(
        F.lit("photo_behavior").alias("segment_type"),
        F.col("segment_value"),
        F.lit("all").alias("source"),
    ).agg(*_standard_sentiment_aggs())

    # 3.2 Nhóm theo tần suất đánh giá / mua lặp lại (purchase_frequency)
    cust_counts = (
        df.filter(F.col("customer_id").isNotNull() & (F.trim(F.col("customer_id")) != ""))
        .groupBy("source", "customer_id")
        .agg(F.count("*").alias("cust_review_count"))
    )
    df_with_freq = df.join(cust_counts, on=["source", "customer_id"], how="left").withColumn(
        "segment_value",
        F.when(F.col("cust_review_count").isNull(), F.lit("Ẩn danh / Khách vãng lai"))
        .when(F.col("cust_review_count") == 1, F.lit("Khách mua 1 lần (1 đánh giá)"))
        .when(F.col("cust_review_count").between(2, 4), F.lit("Khách mua lặp lại (2-4 đánh giá)"))
        .otherwise(F.lit("Khách hàng thân thiết (>= 5 đánh giá)")),
    )
    freq_by_src = df_with_freq.groupBy(
        F.lit("purchase_frequency").alias("segment_type"),
        F.col("segment_value"),
        F.col("source"),
    ).agg(*_standard_sentiment_aggs())
    freq_all = df_with_freq.groupBy(
        F.lit("purchase_frequency").alias("segment_type"),
        F.col("segment_value"),
        F.lit("all").alias("source"),
    ).agg(*_standard_sentiment_aggs())

    # 3.3 Nhóm theo Top Nhà bán hàng (seller_name có >= 100 đánh giá)
    seller_agg = (
        df.filter(F.col("seller_name").isNotNull() & (F.trim(F.col("seller_name")) != ""))
        .groupBy(
            F.lit("seller").alias("segment_type"),
            F.col("seller_name").alias("segment_value"),
            F.col("source"),
        )
        .agg(*_standard_sentiment_aggs())
        .filter(F.col("total_reviews") >= 100)
    )

    unioned = (
        photo_by_src.unionByName(photo_all)
        .unionByName(freq_by_src)
        .unionByName(freq_all)
        .unionByName(seller_agg)
    )
    return unioned.orderBy("segment_type", "source", F.col("total_reviews").desc()).toPandas()


def aggregate_by_price_tier(df: DataFrame) -> pd.DataFrame:
    """4. Tổng hợp xu hướng cảm xúc theo Phân khúc giá (Price Tier)."""
    logger.info("4. Đang tổng hợp xu hướng cảm xúc theo Phân khúc giá (trends_by_price_tier)...")

    # 4.1 Theo Price Tier + Sàn (category_group = 'ALL')
    tier_by_src = df.groupBy(
        F.col("price_tier"),
        F.col("price_tier_order"),
        F.col("source"),
        F.lit("ALL").alias("category_group"),
    ).agg(
        F.countDistinct("product_id").alias("total_products"),
        F.round(F.avg("price"), 0).alias("avg_price"),
        *_standard_sentiment_aggs(),
    )

    # 4.2 Theo Price Tier trên Toàn hệ thống (source = 'all', category_group = 'ALL')
    tier_all = df.groupBy(
        F.col("price_tier"),
        F.col("price_tier_order"),
        F.lit("all").alias("source"),
        F.lit("ALL").alias("category_group"),
    ).agg(
        F.countDistinct("product_id").alias("total_products"),
        F.round(F.avg("price"), 0).alias("avg_price"),
        *_standard_sentiment_aggs(),
    )

    # 4.3 Theo Price Tier x Nhóm ngành hàng (source = 'all')
    tier_by_cat = df.groupBy(
        F.col("price_tier"),
        F.col("price_tier_order"),
        F.lit("all").alias("source"),
        F.col("category_group"),
    ).agg(
        F.countDistinct("product_id").alias("total_products"),
        F.round(F.avg("price"), 0).alias("avg_price"),
        *_standard_sentiment_aggs(),
    )

    unioned = tier_all.unionByName(tier_by_src).unionByName(tier_by_cat)
    return unioned.orderBy("source", "category_group", "price_tier_order").toPandas()


def aggregate_by_category(df: DataFrame) -> pd.DataFrame:
    """5. Tổng hợp xu hướng cảm xúc & tỷ lệ mâu thuẫn sao vs chữ theo Ngành hàng."""
    logger.info("5. Đang tổng hợp xu hướng cảm xúc theo Ngành hàng (trends_by_category)...")

    # 5.1 Theo từng sàn và danh mục gốc
    cat_by_src = df.groupBy(
        F.col("source"),
        F.col("category_id"),
        F.col("category_name"),
        F.col("category_group"),
    ).agg(
        F.countDistinct("product_id").alias("total_products"),
        F.round(F.avg("price"), 0).alias("avg_price"),
        *_standard_sentiment_aggs(),
    )

    # 5.2 Gộp liên sàn (source = 'all') theo category_group
    cat_all = df.groupBy(
        F.lit("all").alias("source"),
        F.lit("ALL").alias("category_id"),
        F.col("category_group").alias("category_name"),
        F.col("category_group"),
    ).agg(
        F.countDistinct("product_id").alias("total_products"),
        F.round(F.avg("price"), 0).alias("avg_price"),
        *_standard_sentiment_aggs(),
    )

    unioned = cat_all.unionByName(cat_by_src)
    return unioned.orderBy("source", F.col("total_reviews").desc()).toPandas()


def sync_tables_to_mongodb(
    tables: Dict[str, pd.DataFrame],
    settings: Dict[str, Any],
    mongo_uri: str | None = None,
) -> bool:
    """Đồng bộ các bảng tổng hợp vào MongoDB (collection riêng cho từng chiều phân tích)."""
    try:
        from pymongo import MongoClient
    except ImportError:
        logger.warning("Thư viện pymongo chưa được cài đặt, bỏ qua bước đồng bộ MongoDB.")
        return False

    mongo_cfg = settings.get("mongodb", {})
    uri = mongo_uri or mongo_cfg.get("uri", "mongodb://localhost:27017")
    db_name = mongo_cfg.get("database", "ecommerce_sentiment")
    collections_cfg = mongo_cfg.get("collections", {})

    collection_mapping = {
        "trends_by_product": collections_cfg.get("trends_product", "trends_by_product"),
        "trends_by_time": collections_cfg.get("trends_time", "trends_by_time"),
        "trends_by_customer": collections_cfg.get("trends_customer", "trends_by_customer"),
        "trends_by_price_tier": collections_cfg.get("trends_price_tier", "trends_by_price_tier"),
        "trends_by_category": collections_cfg.get("trends_category", "trends_by_category"),
    }

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=2500)
        client.server_info()
        db = client[db_name]

        for key, pdf in tables.items():
            coll_name = collection_mapping.get(key, key)
            coll = db[coll_name]
            # Chuyển NaN thành None để tương thích chuẩn BSON của MongoDB
            clean_df = pdf.replace({np.nan: None})
            records = clean_df.to_dict(orient="records")
            coll.delete_many({})
            if records:
                coll.insert_many(records, ordered=False)
            logger.info("Đã đồng bộ %d bản ghi vào MongoDB [%s.%s]", len(records), db_name, coll_name)

        client.close()
        logger.info("✅ Đồng bộ toàn bộ 5 bảng tổng hợp vào MongoDB thành công!")
        return True
    except Exception as e:
        logger.info(
            "MongoDB tại %s hiện chưa bật (%s). Đã lưu đầy đủ các bảng tổng hợp ở dạng Parquet & JSON phục vụ Dashboard.",
            uri,
            e.__class__.__name__,
        )
        return False


def run_aggregate_trends(
    gold_path: str | Path = "data/gold/gold_reviews.parquet",
    seeds_path: str | Path = "data/seeds/products.jsonl",
    output_dir: str | Path = "data/gold",
    config_path: str | Path = "configs/settings.yaml",
    sync_mongo: bool = True,
    mongo_uri: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """Chạy toàn bộ pipeline tổng hợp xu hướng cảm xúc bằng Spark."""
    t0 = time.time()
    settings = load_settings(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spark = get_spark_session(settings)
    df_enriched = None

    try:
        df_enriched = load_enriched_gold_dataframe(spark, gold_path=gold_path, seeds_path=seeds_path)
        df_enriched.cache()
        total_rows = df_enriched.count()
        logger.info("Đã nạp và enrich thành công %d bản ghi đánh giá tầng Gold.", total_rows)

        pdf_product = aggregate_by_product(df_enriched)
        pdf_time = aggregate_by_time(df_enriched)
        pdf_customer = aggregate_by_customer(df_enriched)
        pdf_price_tier = aggregate_by_price_tier(df_enriched)
        pdf_category = aggregate_by_category(df_enriched)

        tables = {
            "trends_by_product": pdf_product,
            "trends_by_time": pdf_time,
            "trends_by_customer": pdf_customer,
            "trends_by_price_tier": pdf_price_tier,
            "trends_by_category": pdf_category,
        }

        # Lưu từng bảng ra Parquet tại data/gold/
        for name, pdf in tables.items():
            out_file = out_dir / f"{name}.parquet"
            pdf.to_parquet(out_file, index=False, compression="snappy")
            logger.info("Đã lưu bảng tổng hợp: %s (%d dòng, %d cột)", out_file, len(pdf), len(pdf.columns))

        # Đồng bộ sang MongoDB (nếu MongoDB đang chạy)
        mongo_synced = False
        if sync_mongo:
            mongo_synced = sync_tables_to_mongodb(tables, settings, mongo_uri=mongo_uri)

        # Trích xuất các Insight nổi bật ra file JSON phục vụ kiểm tra nhanh & báo cáo
        tier_overall = pdf_price_tier[
            (pdf_price_tier["source"] == "all") & (pdf_price_tier["category_group"] == "ALL")
        ].to_dict(orient="records")
        cat_overall = pdf_category[pdf_category["source"] == "all"].to_dict(orient="records")
        photo_overall = pdf_customer[
            (pdf_customer["segment_type"] == "photo_behavior") & (pdf_customer["source"] == "all")
        ].to_dict(orient="records")
        freq_overall = pdf_customer[
            (pdf_customer["segment_type"] == "purchase_frequency") & (pdf_customer["source"] == "all")
        ].to_dict(orient="records")
        anomaly_products = pdf_product[pdf_product["is_anomaly_negative"] == True]
        top_negative_products = (
            pdf_product[pdf_product["total_reviews"] >= 30]
            .sort_values(by="text_negative_pct", ascending=False)
            .head(10)[
                [
                    "source",
                    "product_id",
                    "product_name",
                    "category_group",
                    "price_tier",
                    "total_reviews",
                    "avg_rating",
                    "negative_pct",
                    "text_negative_pct",
                    "conflict_pct",
                ]
            ]
            .to_dict(orient="records")
        )

        summary_report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "execution_time_seconds": round(time.time() - t0, 2),
            "total_reviews_aggregated": int(total_rows),
            "total_products_aggregated": int(len(pdf_product)),
            "anomaly_negative_products_count": int(len(anomaly_products)),
            "mongodb_synced": mongo_synced,
            "price_tier_insights": tier_overall,
            "category_insights": cat_overall,
            "photo_behavior_insights": photo_overall,
            "purchase_frequency_insights": freq_overall,
            "top_10_negative_products": top_negative_products,
        }

        summary_path = out_dir / "trends_summary_report.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2)
        logger.info("Đã lưu báo cáo tổng hợp xu hướng tại: %s", summary_path)

        # In báo cáo tóm tắt trực quan ra màn hình
        print("\n" + "=" * 95)
        print(" BÁO CÁO KẾT QUẢ TỔNG HỢP XU HƯỚNG CẢM XÚC ĐA CHIỀU (SPARK GOLD AGGREGATION)")
        print("=" * 95)
        print(f"- Tổng số đánh giá đã phân tích: {total_rows:,} bản ghi | Số sản phẩm: {len(pdf_product):,}")
        print(f"- Số sản phẩm phát hiện tỷ lệ tiêu cực bất thường (Anomaly): {len(anomaly_products):,} sản phẩm")
        print("-" * 95)
        print("1. XU HƯỚNG THEO PHÂN KHÚC GIÁ (PRICE TIER):")
        print(
            f"  {'Phân khúc giá':<26} | {'Số SP':<6} | {'Tổng Review':<11} | {'Sao TB':<6} | {'%Neg(All)':<9} | {'%Neg(Text)':<10} | {'%Mâu thuẫn':<10}"
        )
        for r in tier_overall:
            print(
                f"  {r['price_tier']:<26} | {r['total_products']:<6} | {r['total_reviews']:<11,} | "
                f"{r['avg_rating']:<6.2f} | {r['negative_pct']:<8.2f}% | {r['text_negative_pct']:<9.2f}% | {r['conflict_pct']:<9.2f}%"
            )

        print("-" * 95)
        print("2. XU HƯỚNG THEO NHÓM NGÀNH HÀNG LIÊN SÀN (CATEGORY GROUP):")
        print(
            f"  {'Ngành hàng':<26} | {'Số SP':<6} | {'Tổng Review':<11} | {'Sao TB':<6} | {'%Pos(Text)':<10} | {'%Neg(Text)':<10} | {'%Mâu thuẫn':<10}"
        )
        for r in cat_overall:
            print(
                f"  {str(r['category_group'])[:26]:<26} | {r['total_products']:<6} | {r['total_reviews']:<11,} | "
                f"{r['avg_rating']:<6.2f} | {r['text_positive_pct']:<9.2f}% | {r['text_negative_pct']:<9.2f}% | {r['conflict_pct']:<9.2f}%"
            )

        print("-" * 95)
        print("3. XU HƯỚNG THEO HÀNH VI KHÁCH HÀNG (ĐĂNG ẢNH & TẦN SUẤT MUA):")
        for r in photo_overall + freq_overall:
            print(
                f"  {r['segment_value']:<38} | Reviews: {r['total_reviews']:<9,} | "
                f"Sao TB: {r['avg_rating']:<4.2f} | %Pos(Text): {r['text_positive_pct']:<6.2f}% | %Neg(Text): {r['text_negative_pct']:<5.2f}%"
            )
        print("=" * 95 + "\n")

        return tables

    finally:
        if df_enriched is not None:
            try:
                df_enriched.unpersist()
            except Exception:
                pass
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Spark Job: Tổng hợp xu hướng cảm xúc đa chiều tầng Gold.")
    parser.add_argument("--gold-path", default="data/gold/gold_reviews.parquet", help="Đường dẫn file Gold parquet")
    parser.add_argument("--seeds-path", default="data/seeds/products.jsonl", help="Đường dẫn file Seeds jsonl")
    parser.add_argument("--output-dir", default="data/gold", help="Thư mục lưu các bảng Parquet tổng hợp")
    parser.add_argument("--config", default="configs/settings.yaml", help="Đường dẫn file cấu hình")
    parser.add_argument("--no-mongo", action="store_true", help="Bỏ qua bước đồng bộ vào MongoDB")
    parser.add_argument("--mongo-uri", default=None, help="Ghi đè MongoDB URI nếu cần")
    args = parser.parse_args()

    run_aggregate_trends(
        gold_path=args.gold_path,
        seeds_path=args.seeds_path,
        output_dir=args.output_dir,
        config_path=args.config,
        sync_mongo=not args.no_mongo,
        mongo_uri=args.mongo_uri,
    )


if __name__ == "__main__":
    main()
