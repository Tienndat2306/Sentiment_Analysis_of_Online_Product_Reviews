"""
Airflow DAG 1: Data Engineering Pipeline (dag_de.py)
====================================================
Lịch chạy: 01:00 sáng hàng ngày (`0 1 * * *`) — chạy trước DAG Data Science 1 tiếng.
Nhiệm vụ:
1. `discover_seed_products`: Quét danh mục Tiki & Lazada (`configs/categories.yaml`) -> `data/seeds/products.jsonl`.
2. `crawl_bronze_reviews`: Thu thập đánh giá mới từ danh sách hạt giống -> `data/bronze/*.csv`.
3. `spark_etl_silver`: Chạy PySpark làm sạch văn bản, khử trùng lặp, gán nhãn yếu (`label_weak`)
   và kiểm định chất lượng dữ liệu tự động -> `data/silver/reviews.parquet` & `data/silver/dq_report.json`.
4. `verify_silver_dq_gate`: Kiểm tra ngưỡng chất lượng dữ liệu (Data Quality Gate) trước khi kết thúc.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DQ_REPORT_PATH = PROJECT_ROOT / "data" / "silver" / "dq_report.json"

default_args = {
    "owner": "bigdata_de_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def validate_silver_data_quality() -> None:
    """Data Quality Gate: Đảm bảo tầng Silver đạt chuẩn trước khi cho phép tầng DS chạy."""
    if not DQ_REPORT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy báo cáo chất lượng dữ liệu tại: {DQ_REPORT_PATH}")

    with open(DQ_REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    summary = report.get("summary", {})
    total_silver = int(summary.get("total_silver_records", 0))
    usable_text = int(summary.get("text_usable_for_nlp", 0))
    not_in_seed = int(summary.get("not_in_seed_records", 0))

    if total_silver < 1000:
        raise ValueError(f"Data Quality Gate FAILED: Số bản ghi Silver quá thấp ({total_silver} < 1000).")
    if usable_text < 500:
        raise ValueError(f"Data Quality Gate FAILED: Số bản ghi có văn bản hợp lệ quá thấp ({usable_text} < 500).")
    if not_in_seed > 0:
        raise ValueError(f"Data Quality Gate FAILED: Phát hiện {not_in_seed} bản ghi mồ côi không khớp products.jsonl.")

    print(
        f"[DQ GATE PASSED] Tổng Silver: {total_silver:,} | Có văn bản NLP: {usable_text:,} | "
        f"Khớp Seed: 100% (not_in_seed={not_in_seed})"
    )


with DAG(
    dag_id="ecommerce_sentiment_dag_de",
    default_args=default_args,
    description="Pipeline Data Engineering: Crawl Tiki/Lazada (Bronze) -> PySpark ETL & Data Quality (Silver)",
    schedule_interval="0 1 * * *",  # 01:00 AM mỗi ngày
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["bigdata", "data_engineering", "bronze", "silver", "tiki", "lazada"],
) as dag:

    discover_seed_products = BashOperator(
        task_id="discover_seed_products",
        bash_command=(
            f'cd "{PROJECT_ROOT.as_posix()}" && '
            "python src/crawler/product_discovery.py --source tiki"
        ),
    )

    crawl_bronze_reviews = BashOperator(
        task_id="crawl_bronze_reviews",
        bash_command=(
            f'cd "{PROJECT_ROOT.as_posix()}" && '
            "python src/crawler/crawler.py --from-seeds data/seeds/products.jsonl --max-pages 5"
        ),
    )

    spark_etl_silver = BashOperator(
        task_id="spark_etl_silver",
        bash_command=(
            f'cd "{PROJECT_ROOT.as_posix()}" && '
            "python src/spark_jobs/etl_silver.py"
        ),
    )

    verify_silver_dq_gate = PythonOperator(
        task_id="verify_silver_dq_gate",
        python_callable=validate_silver_data_quality,
    )

    discover_seed_products >> crawl_bronze_reviews >> spark_etl_silver >> verify_silver_dq_gate
