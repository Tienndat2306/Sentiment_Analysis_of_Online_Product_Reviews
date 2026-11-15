"""
Anfra-Style BI & AI Sentiment Analytics Server (FastAPI + Governed Semantic Layer)
==================================================================================
Ứng dụng Dashboard phân tích cảm xúc đa chiều cho Đề tài 4 (Tiki & Lazada):
- Kiến trúc Governed Semantic Layer (lấy cảm hứng từ Anfra BI-as-Code):
  + Tự động kết nối MongoDB (`ecommerce_sentiment`) nếu dịch vụ đang chạy,
    và fallback mượt mà sang động cơ OLAP In-Memory đọc trực tiếp các bảng Parquet tầng Gold (`data/gold/`).
  + Cung cấp các endpoint ngữ nghĩa chuẩn hóa (Cross-filtering toàn cục theo Sàn, Ngành hàng,
    Phân khúc giá, Thời gian mà không cần tải lại trang).
  + Tích hợp cổng Live AI Sentiment Sandbox gọi trực tiếp `models/best_sentiment_model.joblib`
    kèm Ma trận quyết định Text-First để kiểm thử trực tiếp khi bảo vệ đồ án.

Cách khởi chạy:
    python src/dashboard/app.py
Hoặc:
    uvicorn src.dashboard.app:app --host 127.0.0.1 --port 8501 --reload
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import webbrowser
from pathlib import Path
from threading import Timer
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SentimentDashboard")

app = FastAPI(
    title="Tiki & Lazada Sentiment Analytics BI (Anfra-Style Semantic Engine)",
    version="1.0.0",
)

DASHBOARD_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = DASHBOARD_DIR / "templates" / "index.html"
STATIC_DIR = DASHBOARD_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class PredictRequest(BaseModel):
    text: str
    rating: int = 5


class SemanticDataStore:
    """Kho dữ liệu ngữ nghĩa (Governed Semantic Store) nạp sẵn các bảng Gold & Model vào RAM."""

    def __init__(self) -> None:
        self.gold_dir = ROOT_DIR / "data" / "gold"
        self.seeds_path = ROOT_DIR / "data" / "seeds" / "products.jsonl"
        self.models_dir = ROOT_DIR / "models"
        self.config_path = ROOT_DIR / "configs" / "settings.yaml"

        self.storage_mode = "Parquet OLAP (data/gold/*.parquet)"
        self.mongo_connected = False

        self.df_reviews: pd.DataFrame = pd.DataFrame()
        self.df_product: pd.DataFrame = pd.DataFrame()
        self.df_time: pd.DataFrame = pd.DataFrame()
        self.df_customer: pd.DataFrame = pd.DataFrame()
        self.df_price_tier: pd.DataFrame = pd.DataFrame()
        self.df_category: pd.DataFrame = pd.DataFrame()

        self.summary_report: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Any] = {}
        self.eval_report: Dict[str, Any] = {}
        self.spark_eval_report: Dict[str, Any] = {}
        self.dq_report: Dict[str, Any] = {}
        self.pipeline: Any = None

    def load_all(self) -> None:
        logger.info("Đang khởi tạo Governed Semantic Store cho Dashboard...")

        # 1. Kiểm tra kết nối MongoDB nếu có
        settings = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f) or {}

        try:
            from pymongo import MongoClient

            uri = settings.get("mongodb", {}).get("uri", "mongodb://localhost:27017")
            db_name = settings.get("mongodb", {}).get("database", "ecommerce_sentiment")
            client = MongoClient(uri, serverSelectionTimeoutMS=1200)
            client.server_info()
            db = client[db_name]
            if db["trends_by_product"].count_documents({}, limit=1) > 0:
                self.mongo_connected = True
                self.storage_mode = f"MongoDB ({db_name}) + Parquet OLAP"
                logger.info("Đã kết nối thành công với MongoDB [%s]", db_name)
            client.close()
        except Exception:
            self.mongo_connected = False
            self.storage_mode = "Parquet OLAP Engine (data/gold/)"

        # 2. Nạp 5 bảng Marts từ data/gold/
        self.df_product = pd.read_parquet(self.gold_dir / "trends_by_product.parquet")
        self.df_time = pd.read_parquet(self.gold_dir / "trends_by_time.parquet")
        self.df_customer = pd.read_parquet(self.gold_dir / "trends_by_customer.parquet")
        self.df_price_tier = pd.read_parquet(self.gold_dir / "trends_by_price_tier.parquet")
        self.df_category = pd.read_parquet(self.gold_dir / "trends_by_category.parquet")

        # 3. Nạp bảng Fact gold_reviews.parquet và enrich nhanh với price_tier & category_group
        cols_to_load = [
            "review_id",
            "product_id",
            "product_name",
            "rating",
            "content_clean",
            "is_text_usable",
            "label_weak",
            "label_text",
            "confidence_score",
            "label_final",
            "is_conflict",
            "is_noisy",
            "is_photo",
            "created_at",
            "seller_name",
            "category_name",
            "source",
            "prediction_source",
        ]
        self.df_reviews = pd.read_parquet(self.gold_dir / "gold_reviews.parquet", columns=cols_to_load)

        # Map price & price_tier từ df_product sang df_reviews để hỗ trợ lọc chéo (Cross-filter) tức thì
        self.df_product["product_id"] = self.df_product["product_id"].astype(str)
        prod_meta = self.df_product[
            ["source", "product_id", "price", "price_tier", "category_group"]
        ].drop_duplicates(subset=["source", "product_id"])
        self.df_reviews["product_id"] = self.df_reviews["product_id"].astype(str)
        self.df_reviews = self.df_reviews.merge(prod_meta, on=["source", "product_id"], how="left")
        self.df_reviews["created_at"] = pd.to_datetime(self.df_reviews["created_at"], errors="coerce")
        self.df_reviews["year_month"] = self.df_reviews["created_at"].dt.strftime("%Y-%m")
        self.df_reviews["year"] = self.df_reviews["created_at"].dt.year.fillna(2026).astype(int)

        # 4. Nạp các file JSON Metadata & Evaluation
        self._load_json_into("trends_summary_report.json", self.gold_dir, "summary_report")
        self._load_json_into("metadata.json", self.models_dir, "model_metadata")
        self._load_json_into("evaluation_report.json", self.models_dir, "eval_report")
        self._load_json_into("spark_mllib_evaluation.json", self.models_dir, "spark_eval_report")
        self._load_json_into("dq_report.json", ROOT_DIR / "data" / "silver", "dq_report")

        # 5. Nạp model phục vụ Live AI Sentiment Sandbox
        model_file = self.models_dir / "best_sentiment_model.joblib"
        if model_file.exists():
            self.pipeline = joblib.load(model_file)
            logger.info("Đã nạp thành công mô hình AI: %s", model_file.name)

        logger.info(
            "✅ Semantic Store sẵn sàng: %d đánh giá, %d sản phẩm.",
            len(self.df_reviews),
            len(self.df_product),
        )

    def _load_json_into(self, filename: str, folder: Path, attr: str) -> None:
        p = folder / filename
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                setattr(self, attr, json.load(f))

    def filter_reviews(
        self,
        source: str = "all",
        category_group: str = "ALL",
        price_tier: str = "ALL",
        start_year: int = 2020,
    ) -> pd.DataFrame:
        df = self.df_reviews
        mask = pd.Series(True, index=df.index)
        if source and source.lower() != "all":
            mask &= df["source"] == source.lower()
        if category_group and category_group != "ALL":
            mask &= df["category_group"] == category_group
        if price_tier and price_tier != "ALL":
            mask &= df["price_tier"] == price_tier
        if start_year and start_year > 2014:
            mask &= df["year"] >= int(start_year)
        return df.loc[mask]

    def filter_products(
        self,
        source: str = "all",
        category_group: str = "ALL",
        price_tier: str = "ALL",
    ) -> pd.DataFrame:
        df = self.df_product
        mask = pd.Series(True, index=df.index)
        if source and source.lower() != "all":
            mask &= df["source"] == source.lower()
        if category_group and category_group != "ALL":
            mask &= df["category_group"] == category_group
        if price_tier and price_tier != "ALL":
            mask &= df["price_tier"] == price_tier
        return df.loc[mask]


STORE = SemanticDataStore()


@app.on_event("startup")
def on_startup() -> None:
    if STORE.df_reviews.empty:
        STORE.load_all()


@app.get("/", response_class=HTMLResponse)
def serve_dashboard() -> HTMLResponse:
    if not TEMPLATE_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Không tìm thấy giao diện tại {TEMPLATE_PATH}")
    return HTMLResponse(content=TEMPLATE_PATH.read_text(encoding="utf-8"))


@app.get("/api/dashboard-data")
def get_dashboard_data(
    source: str = Query("all", description="tiki | lazada | all"),
    category_group: str = Query("ALL", description="Nhóm ngành hàng liên sàn"),
    price_tier: str = Query("ALL", description="Phân khúc giá"),
    start_year: int = Query(2021, description="Năm bắt đầu trên biểu đồ thời gian"),
) -> Dict[str, Any]:
    """Endpoint ngữ nghĩa trung tâm: Trả về toàn bộ số liệu KPI và các chiều phân tích đã lọc chéo (Cross-filtered)."""
    df_rev = STORE.filter_reviews(
        source=source,
        category_group=category_group,
        price_tier=price_tier,
        start_year=2014,  # KPI tổng tính trên toàn bộ lịch sử của bộ lọc
    )
    df_prod = STORE.filter_products(
        source=source,
        category_group=category_group,
        price_tier=price_tier,
    )

    total_reviews = len(df_rev)
    usable_df = df_rev[df_rev["is_text_usable"] == True]
    usable_count = len(usable_df)

    avg_rating = round(float(df_rev["rating"].mean()), 2) if total_reviews > 0 else 0.0

    # Phân bố 3 lớp cảm xúc (Toàn bộ & Chỉ tính review có văn bản)
    all_counts = df_rev["label_final"].value_counts().to_dict()
    pos_all = int(all_counts.get("positive", 0))
    neu_all = int(all_counts.get("neutral", 0))
    neg_all = int(all_counts.get("negative", 0))

    txt_counts = usable_df["label_final"].value_counts().to_dict()
    pos_txt = int(txt_counts.get("positive", 0))
    neu_txt = int(txt_counts.get("neutral", 0))
    neg_txt = int(txt_counts.get("negative", 0))

    conflict_count = int(usable_df["is_conflict"].sum()) if usable_count > 0 else 0
    conflict_pct = round(conflict_count * 100.0 / max(usable_count, 1), 2)
    high_star_neg = int(((usable_df["rating"] >= 4) & (usable_df["label_final"] == "negative")).sum())
    anomaly_prod_count = int(df_prod["is_anomaly_negative"].sum()) if not df_prod.empty else 0

    # Phân bố nguồn nhãn (Data Lineage)
    src_counts = df_rev["prediction_source"].value_counts().to_dict()

    # 1. Chuỗi thời gian theo tháng (lọc theo start_year)
    df_time_filtered = df_rev[df_rev["year"] >= int(start_year)]
    if not df_time_filtered.empty:
        ts_grp = (
            df_time_filtered.groupby("year_month", as_index=False)
            .agg(
                total_reviews=("review_id", "count"),
                avg_rating=("rating", "mean"),
                positive_count=("label_final", lambda s: int((s == "positive").sum())),
                neutral_count=("label_final", lambda s: int((s == "neutral").sum())),
                negative_count=("label_final", lambda s: int((s == "negative").sum())),
                conflict_count=("is_conflict", lambda s: int(s.sum())),
            )
            .sort_values("year_month")
        )
        ts_grp["positive_pct"] = (ts_grp["positive_count"] * 100.0 / ts_grp["total_reviews"]).round(2)
        ts_grp["neutral_pct"] = (ts_grp["neutral_count"] * 100.0 / ts_grp["total_reviews"]).round(2)
        ts_grp["negative_pct"] = (ts_grp["negative_count"] * 100.0 / ts_grp["total_reviews"]).round(2)
        ts_grp["avg_rating"] = ts_grp["avg_rating"].round(2)
        time_series = ts_grp.to_dict(orient="records")
    else:
        time_series = []

    # 2. Xu hướng theo Phân khúc giá (Price Tier)
    tier_order = {"Bình dân (< 200k)": 1, "Tầm trung (200k - 1.5M)": 2, "Cao cấp (> 1.5M)": 3}
    df_for_tier = STORE.filter_reviews(source=source, category_group=category_group, price_tier="ALL", start_year=2014)
    price_tier_rows = []
    if not df_for_tier.empty:
        for tier_name, grp in df_for_tier.groupby("price_tier"):
            if tier_name == "Chưa xác định":
                continue
            u_grp = grp[grp["is_text_usable"] == True]
            n_tot = len(grp)
            n_u = len(u_grp)
            price_tier_rows.append(
                {
                    "price_tier": tier_name,
                    "order": tier_order.get(str(tier_name), 9),
                    "total_products": int(grp["product_id"].nunique()),
                    "total_reviews": n_tot,
                    "usable_text_reviews": n_u,
                    "avg_rating": round(float(grp["rating"].mean()), 2),
                    "negative_pct": round(float((grp["label_final"] == "negative").sum() * 100.0 / max(n_tot, 1)), 2),
                    "text_positive_pct": round(
                        float((u_grp["label_final"] == "positive").sum() * 100.0 / max(n_u, 1)), 2
                    ),
                    "text_neutral_pct": round(
                        float((u_grp["label_final"] == "neutral").sum() * 100.0 / max(n_u, 1)), 2
                    ),
                    "text_negative_pct": round(
                        float((u_grp["label_final"] == "negative").sum() * 100.0 / max(n_u, 1)), 2
                    ),
                    "conflict_pct": round(float(u_grp["is_conflict"].sum() * 100.0 / max(n_u, 1)), 2),
                }
            )
        price_tier_rows.sort(key=lambda x: x["order"])

    # 3. Xu hướng theo Ngành hàng (Category Group)
    df_for_cat = STORE.filter_reviews(source=source, category_group="ALL", price_tier=price_tier, start_year=2014)
    category_rows = []
    if not df_for_cat.empty:
        for cat_name, grp in df_for_cat.groupby("category_group"):
            u_grp = grp[grp["is_text_usable"] == True]
            n_tot = len(grp)
            n_u = len(u_grp)
            category_rows.append(
                {
                    "category_group": str(cat_name),
                    "total_products": int(grp["product_id"].nunique()),
                    "total_reviews": n_tot,
                    "usable_text_reviews": n_u,
                    "avg_rating": round(float(grp["rating"].mean()), 2),
                    "text_positive_pct": round(
                        float((u_grp["label_final"] == "positive").sum() * 100.0 / max(n_u, 1)), 2
                    ),
                    "text_neutral_pct": round(
                        float((u_grp["label_final"] == "neutral").sum() * 100.0 / max(n_u, 1)), 2
                    ),
                    "text_negative_pct": round(
                        float((u_grp["label_final"] == "negative").sum() * 100.0 / max(n_u, 1)), 2
                    ),
                    "conflict_pct": round(float(u_grp["is_conflict"].sum() * 100.0 / max(n_u, 1)), 2),
                    "high_star_negative_count": int(
                        ((u_grp["rating"] >= 4) & (u_grp["label_final"] == "negative")).sum()
                    ),
                }
            )
        category_rows.sort(key=lambda x: x["total_reviews"], reverse=True)

    # 4. Ma trận Heatmap Ngành hàng x Phân khúc giá (% Tiêu cực trên văn bản)
    heatmap_matrix = []
    tier_labels = ["Bình dân (< 200k)", "Tầm trung (200k - 1.5M)", "Cao cấp (> 1.5M)"]
    cat_labels = [c["category_group"] for c in category_rows]
    if not df_for_cat.empty:
        u_all = df_for_cat[df_for_cat["is_text_usable"] == True]
        for c_idx, c_name in enumerate(cat_labels):
            for t_idx, t_name in enumerate(tier_labels):
                cell = u_all[(u_all["category_group"] == c_name) & (u_all["price_tier"] == t_name)]
                if len(cell) >= 5:
                    neg_rate = round(float((cell["label_final"] == "negative").sum() * 100.0 / len(cell)), 1)
                else:
                    neg_rate = 0.0
                heatmap_matrix.append([t_idx, c_idx, neg_rate, int(len(cell))])

    # 5. Nhóm khách hàng (is_photo, purchase_frequency, top sellers)
    src_filter = source.lower() if source.lower() in ("tiki", "lazada") else "all"
    cust_df = STORE.df_customer[STORE.df_customer["source"] == src_filter]
    photo_segments = cust_df[cust_df["segment_type"] == "photo_behavior"].to_dict(orient="records")
    freq_segments = cust_df[cust_df["segment_type"] == "purchase_frequency"].to_dict(orient="records")
    seller_df = STORE.df_customer[STORE.df_customer["segment_type"] == "seller"]
    if src_filter != "all":
        seller_df = seller_df[seller_df["source"] == src_filter]
    top_sellers = seller_df.sort_values("total_reviews", ascending=False).head(12).to_dict(orient="records")

    # 6. Mẫu các đánh giá Mâu thuẫn điển hình (4-5 sao nhưng văn bản tiêu cực / trung lập)
    conflict_samples_df = usable_df[
        (usable_df["is_conflict"] == True) & (usable_df["rating"] >= 4) & (usable_df["label_final"] == "negative")
    ].sort_values("confidence_score", ascending=False)
    conflict_samples = []
    for _, r in conflict_samples_df.head(20).iterrows():
        conflict_samples.append(
            {
                "review_id": str(r["review_id"]),
                "source": str(r["source"]),
                "product_name": str(r["product_name"]),
                "category_group": str(r["category_group"]),
                "price_tier": str(r["price_tier"]),
                "rating": int(r["rating"]),
                "label_weak": str(r["label_weak"]),
                "label_final": str(r["label_final"]),
                "confidence_score": 0.0 if pd.isna(r["confidence_score"]) else round(float(r["confidence_score"]), 3),
                "prediction_source": str(r["prediction_source"]),
                "content_clean": str(r["content_clean"]),
            }
        )

    return {
        "storage_mode": STORE.storage_mode,
        "kpis": {
            "total_reviews": total_reviews,
            "total_products": int(len(df_prod)),
            "usable_text_reviews": usable_count,
            "usable_text_pct": round(usable_count * 100.0 / max(total_reviews, 1), 1),
            "avg_rating": avg_rating,
            "positive_all": pos_all,
            "neutral_all": neu_all,
            "negative_all": neg_all,
            "positive_all_pct": round(pos_all * 100.0 / max(total_reviews, 1), 2),
            "neutral_all_pct": round(neu_all * 100.0 / max(total_reviews, 1), 2),
            "negative_all_pct": round(neg_all * 100.0 / max(total_reviews, 1), 2),
            "positive_text": pos_txt,
            "neutral_text": neu_txt,
            "negative_text": neg_txt,
            "positive_text_pct": round(pos_txt * 100.0 / max(usable_count, 1), 2),
            "neutral_text_pct": round(neu_txt * 100.0 / max(usable_count, 1), 2),
            "negative_text_pct": round(neg_txt * 100.0 / max(usable_count, 1), 2),
            "conflict_count": conflict_count,
            "conflict_pct": conflict_pct,
            "high_star_neg_count": high_star_neg,
            "anomaly_product_count": anomaly_prod_count,
            "prediction_sources": {
                "phobert_teacher": int(src_counts.get("phobert_teacher", 0)),
                "gemini_verified": int(src_counts.get("gemini_verified", 0)),
                "phobert_weak_agree": int(src_counts.get("phobert_weak_agree", 0)),
                "rule_weak_label": int(src_counts.get("rule_weak_label", 0)),
            },
        },
        "time_series": time_series,
        "price_tiers": price_tier_rows,
        "categories": category_rows,
        "heatmap": {
            "tiers": tier_labels,
            "categories": cat_labels,
            "data": heatmap_matrix,
        },
        "customer_segments": {
            "photo": photo_segments,
            "frequency": freq_segments,
            "sellers": top_sellers,
        },
        "conflict_samples": conflict_samples,
    }


@app.get("/api/products")
def get_products_table(
    source: str = Query("all"),
    category_group: str = Query("ALL"),
    price_tier: str = Query("ALL"),
    search: str = Query(""),
    anomaly_only: bool = Query(False),
    sort_by: str = Query("total_reviews"),
    limit: int = Query(100, ge=10, le=700),
) -> Dict[str, Any]:
    """Danh sách sản phẩm tương tác kèm bộ lọc tìm kiếm và cảnh báo bất thường."""
    df = STORE.filter_products(source=source, category_group=category_group, price_tier=price_tier).copy()
    if anomaly_only:
        df = df[df["is_anomaly_negative"] == True]
    if search.strip():
        q = search.strip().lower()
        df = df[
            df["product_name"].fillna("").str.lower().str.contains(q, regex=False)
            | df["product_id"].astype(str).str.contains(q, regex=False)
        ]

    valid_sort_cols = {
        "total_reviews": False,
        "text_negative_pct": False,
        "negative_pct": False,
        "conflict_pct": False,
        "avg_rating": True,
        "price": False,
    }
    asc = valid_sort_cols.get(sort_by, False)
    if sort_by in df.columns:
        df = df.sort_values(by=[sort_by, "total_reviews"], ascending=[asc, False])

    total_matched = len(df)
    records = df.head(limit).replace({np.nan: None}).to_dict(orient="records")
    return {
        "total_matched": total_matched,
        "products": records,
    }


@app.get("/api/product/{product_id}/reviews")
def get_product_reviews_drilldown(
    product_id: str,
    sentiment: str = Query("all"),
    conflict_only: bool = Query(False),
    limit: int = Query(40, ge=5, le=200),
) -> Dict[str, Any]:
    """Drill-down: Xem chi tiết các bình luận thực tế của 1 sản phẩm cụ thể."""
    df = STORE.df_reviews[
        (STORE.df_reviews["product_id"] == str(product_id)) & (STORE.df_reviews["is_text_usable"] == True)
    ].copy()
    if sentiment in ("positive", "neutral", "negative"):
        df = df[df["label_final"] == sentiment]
    if conflict_only:
        df = df[df["is_conflict"] == True]

    df = df.sort_values(by=["is_conflict", "created_at"], ascending=[False, False])
    items = []
    for _, r in df.head(limit).iterrows():
        items.append(
            {
                "review_id": str(r["review_id"]),
                "rating": int(r["rating"]),
                "content_clean": str(r["content_clean"]),
                "label_weak": str(r["label_weak"]),
                "label_final": str(r["label_final"]),
                "confidence_score": 0.0 if pd.isna(r["confidence_score"]) else round(float(r["confidence_score"]), 3),
                "is_conflict": bool(r["is_conflict"]),
                "prediction_source": str(r["prediction_source"]),
                "created_at": r["created_at"].strftime("%Y-%m-%d") if pd.notna(r["created_at"]) else "",
            }
        )
    return {
        "product_id": product_id,
        "total_usable_reviews": len(df),
        "reviews": items,
    }


@app.get("/api/models")
def get_models_evaluation() -> Dict[str, Any]:
    """Trả về thông tin Model Registry, kết quả so sánh Scikit-learn vs Spark MLlib và Confusion Matrix."""
    active_info = {}
    if STORE.model_metadata.get("models"):
        active_info = STORE.model_metadata["models"][-1]
    return {
        "active_model": active_info,
        "evaluation_report": STORE.eval_report,
        "spark_mllib_report": STORE.spark_eval_report,
        "dq_summary": STORE.dq_report.get("summary", {}),
    }


@app.post("/api/predict")
def predict_live_sentiment(req: PredictRequest) -> Dict[str, Any]:
    """Live AI Sentiment Sandbox: Dự đoán cảm xúc trực tiếp cho văn bản bất kỳ + giải thích Ma trận quyết định."""
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung đánh giá.")
    if STORE.pipeline is None:
        raise HTTPException(status_code=500, detail="Mô hình chưa được nạp.")

    rating = max(1, min(5, int(req.rating)))
    label_weak = "positive" if rating >= 4 else ("neutral" if rating == 3 else "negative")

    pred_label = str(STORE.pipeline.predict([text])[0])
    probs = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    confidence = 1.0

    if hasattr(STORE.pipeline, "predict_proba"):
        proba_arr = STORE.pipeline.predict_proba([text])[0]
        classes = list(STORE.pipeline.classes_)
        for cls_name, p_val in zip(classes, proba_arr):
            probs[str(cls_name)] = round(float(p_val) * 100.0, 2)
        confidence = round(float(np.max(proba_arr)), 4)

    words = text.split()
    if len(words) < 3:
        label_final = label_weak
        decision_rule = "Văn bản quá ngắn (< 3 từ) -> Ưu tiên nhãn số sao (label_weak)"
        is_noisy = False
    elif confidence >= 0.60:
        label_final = pred_label
        if label_weak != pred_label:
            decision_rule = f"Phát hiện Mâu thuẫn Sao vs Chữ (Confidence {confidence*100:.1f}% >= 60%) -> Sửa nhãn theo Ngữ nghĩa Văn bản (Text-First)"
        else:
            decision_rule = f"Đồng thuận Sao & Văn bản với độ tự tin cao ({confidence*100:.1f}%)"
        is_noisy = False
    elif label_weak == pred_label:
        label_final = label_weak
        decision_rule = f"Độ tự tin trung bình ({confidence*100:.1f}% < 60%) nhưng Đồng thuận Sao & Chữ -> Giữ nhãn an toàn"
        is_noisy = False
    else:
        label_final = label_weak
        decision_rule = f"Xung đột Sao vs Chữ nhưng Độ tự tin thấp ({confidence*100:.1f}% < 60%) -> Gắn cờ is_noisy=True chờ Gemini thẩm định"
        is_noisy = True

    return {
        "input_text": text,
        "word_count": len(words),
        "rating": rating,
        "label_weak": label_weak,
        "label_text": pred_label,
        "confidence_score": confidence,
        "probabilities": probs,
        "label_final": label_final,
        "is_conflict": label_weak != label_final,
        "is_noisy": is_noisy,
        "decision_rule": decision_rule,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Khởi chạy Dashboard Phân tích Cảm xúc (Anfra-Style BI).")
    parser.add_argument("--host", default="127.0.0.1", help="Host lắng nghe (mặc định: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8501, help="Cổng lắng nghe (mặc định: 8501)")
    parser.add_argument("--no-browser", action="store_true", help="Không tự động mở trình duyệt")
    args = parser.parse_args()

    STORE.load_all()
    url = f"http://{args.host}:{args.port}"
    logger.info("🚀 Dashboard đang chạy tại: %s", url)

    if not args.no_browser:
        Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
