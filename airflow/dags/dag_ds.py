"""
Airflow DAG 2: Data Science & Gold BI Pipeline (dag_ds.py)
==========================================================
Lịch chạy: 02:00 sáng hàng ngày (`0 2 * * *`) — chạy lệch giờ (1 tiếng sau `dag_de.py`).
Nhiệm vụ:
1. `feature_engineering_split`: Trích xuất TF-IDF & tạo tập Train/Test cố định chống rò rỉ dữ liệu
   (`data/gold/train_test_split.parquet`).
2. `train_sklearn_models`: Huấn luyện 4 mô hình Scikit-learn (MultinomialNB, ComplementNB,
   LogisticRegression Balanced, LinearSVC Calibrated) -> lưu `models/best_sentiment_model.joblib`.
3. `train_spark_mllib_models`: Huấn luyện phân tán trên Spark MLlib (NaiveBayes & LogisticRegression Balanced)
   -> lưu `models/spark_mllib_evaluation.json`.
4. `evaluate_best_model`: Đánh giá chuyên sâu F1-Macro, Confusion Matrix & Error Analysis
   -> lưu `models/evaluation_report.json`.
5. `run_gold_inference`: Dự đoán cảm xúc trên toàn bộ 225,659 đánh giá, bảo toàn nhãn Teacher
   (PhoBERT + Gemini) và phát hiện mâu thuẫn Sao vs Chữ -> `data/gold/gold_reviews.parquet`.
6. `aggregate_gold_trends`: Tổng hợp 5 bảng Data Mart xu hướng (Sản phẩm, Thời gian, Hành vi KH,
   Phân khúc giá Price Tier, Ngành hàng) và đồng bộ sang MongoDB phục vụ Dashboard.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = Path(__file__).resolve().parents[2]

default_args = {
    "owner": "bigdata_ds_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ecommerce_sentiment_dag_ds",
    default_args=default_args,
    description="Pipeline Data Science & BI: TF-IDF -> Train/Eval ML -> Gold Inference -> Aggregate Trends & MongoDB",
    schedule_interval="0 2 * * *",  # 02:00 AM mỗi ngày (lệch 1 giờ sau DAG DE)
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["bigdata", "data_science", "gold", "nlp", "sentiment", "dashboard"],
) as dag:

    feature_engineering_split = BashOperator(
        task_id="feature_engineering_split",
        bash_command=(
            f'cd "{PROJECT_ROOT.as_posix()}" && '
            "python src/spark_jobs/feature_engineering.py"
        ),
    )

    train_sklearn_models = BashOperator(
        task_id="train_sklearn_models",
        bash_command=(
            f'cd "{PROJECT_ROOT.as_posix()}" && '
            "python src/spark_jobs/train_model.py"
        ),
    )

    train_spark_mllib_models = BashOperator(
        task_id="train_spark_mllib_models",
        bash_command=(
            f'cd "{PROJECT_ROOT.as_posix()}" && '
            "python src/spark_jobs/train_spark_mllib.py"
        ),
    )

    evaluate_best_model = BashOperator(
        task_id="evaluate_best_model",
        bash_command=(
            f'cd "{PROJECT_ROOT.as_posix()}" && '
            "python src/spark_jobs/evaluate_model.py"
        ),
    )

    run_gold_inference = BashOperator(
        task_id="run_gold_inference",
        bash_command=(
            f'cd "{PROJECT_ROOT.as_posix()}" && '
            "python src/spark_jobs/inference.py"
        ),
    )

    aggregate_gold_trends = BashOperator(
        task_id="aggregate_gold_trends",
        bash_command=(
            f'cd "{PROJECT_ROOT.as_posix()}" && '
            "python src/spark_jobs/aggregate_trends.py"
        ),
    )

    # Luồng phụ thuộc:
    # Feature Engineering -> [Train Sklearn, Train Spark MLlib] -> Evaluate -> Inference -> Aggregate Trends
    feature_engineering_split >> [train_sklearn_models, train_spark_mllib_models] >> evaluate_best_model
    evaluate_best_model >> run_gold_inference >> aggregate_gold_trends
