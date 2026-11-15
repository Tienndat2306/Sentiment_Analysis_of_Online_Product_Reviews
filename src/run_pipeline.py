"""
Bộ Điều Phối Pipeline Toàn Trình (End-to-End Local Orchestrator — src/run_pipeline.py)
======================================================================================
Mô phỏng chính xác 2 DAG của Apache Airflow (`dag_de.py` lúc 01:00 & `dag_ds.py` lúc 02:00)
để chạy trực tiếp trên môi trường Windows / Local mà không bắt buộc phải khởi chạy Docker Airflow.

Các chế độ thực thi:
1. Chạy kiểm thử toàn trình (Bỏ qua bước cào lại từ đầu để bảo toàn 225,659 đánh giá Bronze hiện có):
   python src/run_pipeline.py
2. Chỉ chạy luồng Data Engineering (Bronze -> Silver ETL & Kiểm định chất lượng):
   python src/run_pipeline.py --stage de
3. Chỉ chạy luồng Data Science & Gold BI (Train/Evaluate -> Inference -> Aggregate Trends):
   python src/run_pipeline.py --stage ds
4. Chạy nhanh cập nhật tầng Gold & Dashboard (bỏ qua bước huấn luyện lại mô hình):
   python src/run_pipeline.py --stage ds --skip-train
5. Chạy xong tự động mở Dashboard BI:
   python src/run_pipeline.py --launch-dashboard
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT_DIR / "data" / "gold" / "pipeline_execution_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PipelineOrchestrator")


def run_step(step_id: str, description: str, cmd: List[str]) -> Dict[str, Any]:
    """Thực thi 1 bước trong pipeline dưới dạng tiến trình con và ghi nhận thời gian/trạng thái."""
    logger.info("=" * 78)
    logger.info("▶ BẮT ĐẦU TASK [%s]: %s", step_id, description)
    logger.info("  Lệnh thực thi: %s", " ".join(cmd))
    t0 = time.perf_counter()

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = round(time.perf_counter() - t0, 2)

    if proc.returncode != 0:
        logger.error("✖ TASK [%s] THẤT BẠI (exit_code=%d) sau %.2fs", step_id, proc.returncode, elapsed)
        raise RuntimeError(f"Task '{step_id}' thất bại với mã lỗi {proc.returncode}")

    logger.info("✔ HOÀN TẤT TASK [%s] trong %.2fs", step_id, elapsed)
    return {
        "task_id": step_id,
        "description": description,
        "command": " ".join(cmd),
        "status": "SUCCESS",
        "duration_seconds": elapsed,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }


def verify_silver_dq_gate() -> Dict[str, Any]:
    """Kiểm tra cổng chất lượng dữ liệu tầng Silver (tương đương PythonOperator trong dag_de.py)."""
    t0 = time.perf_counter()
    dq_path = ROOT_DIR / "data" / "silver" / "dq_report.json"
    if not dq_path.exists():
        raise FileNotFoundError(f"Không tìm thấy báo cáo chất lượng dữ liệu Silver: {dq_path}")

    with open(dq_path, "r", encoding="utf-8") as f:
        dq = json.load(f)

    s = dq.get("summary", {})
    total_silver = int(s.get("total_silver_records", 0))
    usable_text = int(s.get("text_usable_for_nlp", 0))
    not_in_seed = int(s.get("not_in_seed_records", 0))

    if total_silver < 1000 or usable_text < 500 or not_in_seed > 0:
        raise ValueError(
            f"Data Quality Gate FAILED: total={total_silver}, usable={usable_text}, not_in_seed={not_in_seed}"
        )

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "✔ [DQ GATE PASSED] Silver hợp lệ: %d bản ghi | Có văn bản NLP: %d | Khớp Seed: 100%%",
        total_silver,
        usable_text,
    )
    return {
        "task_id": "verify_silver_dq_gate",
        "description": "Kiểm tra cổng chất lượng dữ liệu tầng Silver (Data Quality Gate)",
        "command": "internal:verify_silver_dq_gate()",
        "status": "SUCCESS",
        "duration_seconds": elapsed,
        "metrics": {
            "total_silver_records": total_silver,
            "usable_text_records": usable_text,
            "not_in_seed_records": not_in_seed,
        },
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Điều phối chạy toàn trình Pipeline DE + DS + Gold BI.")
    parser.add_argument(
        "--stage",
        choices=["all", "de", "ds"],
        default="all",
        help="Chọn phân hệ cần chạy: 'de' (Bronze->Silver), 'ds' (Silver->Gold & ML), hoặc 'all' (mặc định)",
    )
    parser.add_argument(
        "--with-crawler",
        action="store_true",
        help="Chạy thêm bước cào dữ liệu mới từ API Tiki (mặc định tắt để dùng dữ liệu Bronze sẵn có)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Bỏ qua bước huấn luyện lại mô hình, chỉ chạy Inference & Tổng hợp xu hướng Gold",
    )
    parser.add_argument(
        "--skip-spark-mllib",
        action="store_true",
        help="Bỏ qua bước huấn luyện đối chứng Spark MLlib để tiết kiệm thời gian",
    )
    parser.add_argument(
        "--launch-dashboard",
        action="store_true",
        help="Tự động khởi chạy máy chủ Dashboard BI (cổng 8501) ngay sau khi pipeline hoàn tất",
    )
    args = parser.parse_args()

    py = sys.executable
    pipeline_start = time.perf_counter()
    executed_tasks: List[Dict[str, Any]] = []

    logger.info("==============================================================================")
    logger.info("🚀 KHỞI CHẠY BỘ ĐIỀU PHỐI PIPELINE BIG DATA (STAGE: %s)", args.stage.upper())
    logger.info("==============================================================================")

    # ------------------------------------------------------------------
    # DAG 1: DATA ENGINEERING (ecommerce_sentiment_dag_de)
    # ------------------------------------------------------------------
    if args.stage in ("all", "de"):
        if args.with_crawler:
            executed_tasks.append(
                run_step(
                    "discover_seed_products",
                    "Quét danh mục sản phẩm hạt giống (Stage 1 Discovery)",
                    [py, "src/crawler/product_discovery.py", "--source", "tiki"],
                )
            )
            executed_tasks.append(
                run_step(
                    "crawl_bronze_reviews",
                    "Thu thập đánh giá sản phẩm về tầng Bronze (Stage 2 Review Crawler)",
                    [
                        py,
                        "src/crawler/crawler.py",
                        "--from-seeds",
                        "data/seeds/products.jsonl",
                        "--max-pages",
                        "2",
                    ],
                )
            )

        executed_tasks.append(
            run_step(
                "spark_etl_silver",
                "Chạy PySpark ETL làm sạch văn bản, khử trùng lặp & xuất Silver Parquet",
                [py, "src/spark_jobs/etl_silver.py"],
            )
        )
        executed_tasks.append(verify_silver_dq_gate())

    # ------------------------------------------------------------------
    # DAG 2: DATA SCIENCE & GOLD BI (ecommerce_sentiment_dag_ds)
    # ------------------------------------------------------------------
    if args.stage in ("all", "ds"):
        if not args.skip_train:
            executed_tasks.append(
                run_step(
                    "train_sklearn_models",
                    "Trích xuất TF-IDF (Leak-free split) & Huấn luyện 4 mô hình Scikit-learn",
                    [py, "src/spark_jobs/train_model.py"],
                )
            )
            if not args.skip_spark_mllib:
                executed_tasks.append(
                    run_step(
                        "train_spark_mllib_models",
                        "Huấn luyện phân tán trên Spark MLlib (NaiveBayes & LogisticRegression Balanced)",
                        [py, "src/spark_jobs/train_spark_mllib.py"],
                    )
                )
            executed_tasks.append(
                run_step(
                    "evaluate_best_model",
                    "Đánh giá mô hình tốt nhất (F1-Macro, Confusion Matrix, Error Analysis)",
                    [py, "src/spark_jobs/evaluate_model.py"],
                )
            )

        executed_tasks.append(
            run_step(
                "run_gold_inference",
                "Suy luận cảm xúc toàn bộ 225.6K đánh giá & Phát hiện mâu thuẫn Sao vs Chữ (Gold Layer)",
                [py, "src/spark_jobs/inference.py"],
            )
        )
        executed_tasks.append(
            run_step(
                "aggregate_gold_trends",
                "Tổng hợp 5 bảng Data Mart xu hướng (Sản phẩm, Thời gian, KH, Price Tier, Ngành hàng)",
                [py, "src/spark_jobs/aggregate_trends.py"],
            )
        )

    total_duration = round(time.perf_counter() - pipeline_start, 2)
    report = {
        "orchestrator": "src/run_pipeline.py (Airflow Local Equivalent)",
        "stage_executed": args.stage,
        "status": "SUCCESS",
        "total_duration_seconds": total_duration,
        "executed_at": datetime.now().isoformat(timespec="seconds"),
        "tasks": executed_tasks,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("==============================================================================")
    logger.info("🎉 TOÀN BỘ PIPELINE HOÀN TẤT THÀNH CÔNG TRONG %.2f GIÂY!", total_duration)
    logger.info("📄 Báo cáo điều phối đã lưu tại: %s", REPORT_PATH)
    logger.info("==============================================================================")

    if args.launch_dashboard:
        logger.info("Đang khởi chạy máy chủ Dashboard tại http://127.0.0.1:8501 ...")
        subprocess.run([py, "src/dashboard/app.py"], cwd=str(ROOT_DIR))


if __name__ == "__main__":
    main()
