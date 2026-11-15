# Kế hoạch & Hướng dẫn triển khai — Đề tài 4: Phân tích cảm xúc từ đánh giá sản phẩm trực tuyến

> Môn: Nhập môn Big Data
> Nguồn dữ liệu: **Tiki + Lazada** | Kiến trúc: **Data Engineering + Data Science**
> Hình thức: **1 người thực hiện**, thời gian: **5 tuần**

---

## 1. Giới thiệu đề tài

Xây dựng hệ thống thu thập, xử lý và phân tích cảm xúc (sentiment analysis) từ nội dung đánh giá sản phẩm trên Tiki và Lazada, nhằm phát hiện xu hướng hài lòng/không hài lòng của khách hàng theo **sản phẩm**, **thời gian** và **nhóm khách hàng**.

Đồ án kết hợp hai mảng:

- **Data Engineering**: pipeline thu thập → lưu trữ → xử lý dữ liệu, có khả năng chạy lặp lại (incremental), không chỉ xử lý một lần.
- **Data Science**: trích đặc trưng, huấn luyện và đánh giá mô hình phân loại cảm xúc.

## 2. Mục tiêu

- [x] Pipeline thu thập dữ liệu chạy lặp lại được (không hard-code chạy 1 lần).
- [x] Lưu trữ theo kiến trúc phân tầng Bronze – Silver – Gold trên HDFS và MongoDB.
- [x] Mô hình phân loại cảm xúc 3 lớp: tích cực – trung lập – tiêu cực.
- [x] Bảng tổng hợp xu hướng cảm xúc theo sản phẩm / thời gian / nhóm khách hàng.
- [x] Điều phối bằng Airflow (1 instance, 2 DAG) + giám sát chất lượng dữ liệu.

---

## 3. Phạm vi & nguồn dữ liệu

| Sàn             | Độ khó cào                           | Ghi chú kỹ thuật                                                      | Vai trò                                  |
| ---------------- | ---------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------- |
| **Tiki**   | Dễ — ít chặn, không cần proxy      | API JSON công khai cho sản phẩm & review                              | Nguồn chính                             |
| **Lazada** | Trung bình — thỉnh thoảng cần proxy | Có API JSON catalog &`getReviewList`; có cơ chế chống bot "baxia" | Nguồn bổ sung, đa dạng hoá dữ liệu |

> Không thêm Shopee — chi phí hạ tầng để vượt CAPTCHA/chống bot không tương xứng với giá trị học thuật tăng thêm.

### Quy mô dữ liệu mục tiêu

| Hạng mục                                        | Mục tiêu        | Ghi chú                                                                                  |
| ------------------------------------------------- | ----------------- | ----------------------------------------------------------------------------------------- |
| Tổng số review thô (2 sàn)                    | 80.000 – 150.000 | Không cần vượt quá 200.000                                                           |
| Số sản phẩm mỗi sàn                          | 150 – 300        | Trải nhiều danh mục khác nhau                                                         |
| Review có nội dung hữu ích (sau lọc)         | ~40.000 – 70.000 | Ước tính dựa trên tỷ lệ thiếu content thực tế đã khảo sát trên Tiki (~41%) |
| Tối thiểu mỗi lớp nhãn tiêu cực/trung lập | ~5.000 – 8.000   | Cào có chủ đích bằng filter rating tại API, không cào ngẫu nhiên               |

---

## 4. Kiến trúc hệ thống

### 4.1. Track Data Engineering (Bronze → Silver → Gold)

1. **Thu thập (Crawler)**: lấy dữ liệu thô từ Tiki và Lazada, hỗ trợ incremental (chỉ lấy review mới theo ngày/`review_id`).
2. **Vùng thô (Bronze — HDFS)**: lưu nguyên văn dữ liệu cào được, partition theo `nguồn/năm/tháng/ngày`, không chỉnh sửa.
   - **Tách biệt 2 tập dữ liệu (Fact & Dimension)**:
     - `bronze/reviews/`: Chứa các sự kiện đánh giá (review_id, product_id, rating, content, created_at...).
     - `bronze/products/` (hoặc `seeds/products.jsonl`): Chứa metadata sản phẩm (product_id, product_name, price, original_price, category...).
   - **Quyết định thiết kế dữ liệu (Data Modeling Decision)**: **KHÔNG đưa cột `price` vào bảng `reviews`** nhằm:
     - Tránh dư thừa dữ liệu (1 sản phẩm có hàng trăm review).
     - Tránh sai lệch ngữ nghĩa (API không trả về giá giao dịch lúc mua; giá cào được chỉ là snapshot giá niêm yết hiện tại, không đại diện cho giá mua của các review trong quá khứ).
3. **Xử lý (Silver — Spark ETL & Annotation)**: chuẩn hoá schema giữa 2 sàn, làm sạch text (loại content rỗng/ngắn, khử trùng lặp), gán nhãn yếu `label_weak` và kết hợp `label_text` để tạo nhãn chính `label_final`, kèm bước **kiểm tra chất lượng dữ liệu tự động** (tỷ lệ null, tỷ lệ trùng, phân bố nhãn — log cảnh báo nếu vượt ngưỡng).
4. **Phục vụ (Gold — MongoDB)**: dữ liệu đã làm sạch và chuẩn hoá nhãn, sẵn sàng cho cả mô hình ML và dashboard.

### 4.2. Track Data Science

1. **Trích đặc trưng**: TF-IDF hoặc Word2Vec trên văn bản đã làm sạch (Spark MLlib).
2. **Huấn luyện**: Logistic Regression / Naive Bayes / Linear SVM trên tập dữ liệu đã lọc sạch nhãn (`label_final`), có xử lý mất cân bằng nhãn (class weight hoặc oversampling).
3. **Đánh giá**: ưu tiên **F1-macro** và **Recall theo từng lớp** trên tập kiểm thử chuẩn (`label_text` / Golden Dataset) nhằm phản ánh đúng năng lực nhận diện ngữ nghĩa thực tế.
4. **Model registry**: lưu model kèm phiên bản dữ liệu huấn luyện + chỉ số đánh giá; chỉ thay model mới nếu vượt ngưỡng chấp nhận.
5. **Suy luận (Inference)**: dự đoán nhãn cho review mới, ghi kết quả ngược lại vào MongoDB Gold.

### 4.3. Cơ chế gán nhãn kết hợp & Lọc nhiễu dữ liệu (Weak Supervision & NLP Denoising)

Nhằm giải quyết hiện tượng **mâu thuẫn giữa số sao và nội dung** (ví dụ: khách chấm 5 sao nhưng bình luận chê thậm tệ, hoặc mỉa mai/sarcasm), hệ thống áp dụng cơ chế kết hợp nhãn đa nguồn và lọc nhiễu:

1. **Nguồn nhãn yếu (`label_weak`)**: Ánh xạ tự động từ số sao (`rating`):
   - 4 – 5 sao $\rightarrow$ `positive`
   - 3 sao $\rightarrow$ `neutral`
   - 1 – 2 sao $\rightarrow$ `negative`
2. **Nguồn nhãn văn bản (`label_text`)**: Sử dụng mô hình NLP tiếng Việt (`phobert-base-vietnamese-sentiment` hoặc Gemini API) phân tích ngữ nghĩa từ nội dung `content_clean` trên tập mẫu (~3.000 – 5.000 review) hoặc các mẫu có độ nghi ngờ cao. Mô hình trả về nhãn kèm độ tự tin `confidence_score`.
3. **Kiến trúc Teacher – Student**:
   - **Teacher (PhoBERT / LLM)**: Đóng vai trò chuyên gia gán nhãn ngữ nghĩa chất lượng cao cho tập mẫu benchmark và thẩm định các trường hợp mâu thuẫn.
   - **Student (Spark MLlib)**: Mô hình học máy phân tán phục vụ huấn luyện trên quy mô lớn và suy luận hàng loạt hàng ngày trong pipeline dữ liệu với chi phí tính toán thấp.
4. **Ma trận quyết định nhãn chính (`label_final`)**:
   - **Đồng thuận (`label_weak == label_text`)**: Độ tin cậy cao nhất $\rightarrow$ Chọn làm nhãn huấn luyện chính.
   - **Văn bản quá ngắn (< 4 từ / spam / icon)**: Không đủ ngữ cảnh phân tích text $\rightarrow$ Ưu tiên `label_weak`.
   - **Xung đột có độ tự tin cao (`label_weak != label_text` & `confidence >= 0.85`)**: Khách hàng viết rõ ràng cảm xúc nhưng bấm sai sao hoặc mỉa mai $\rightarrow$ Ưu tiên `label_text` (vì mô hình Spark MLlib học từ từ ngữ, nhãn huấn luyện phải phản ánh đúng ngữ nghĩa câu).
   - **Xung đột cực đoan & độ tự tin thấp**: Đánh dấu nhiễu và **loại bỏ (Drop)** khỏi tập Train để tránh làm hỏng mô hình.
   - **Trường hợp 3 sao**: Nếu text thể hiện rõ tích cực/tiêu cực $\rightarrow$ Phân loại lại theo `label_text`.

### 4.4. Điều phối (Airflow — 1 instance, 2 DAG)

| DAG        | Lịch chạy                                        | Nội dung                                                                                                                         |
| ---------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `dag_de` | 01:00 hằng ngày                                  | Crawler → HDFS Bronze → Spark Silver (Clean & Labeling) → Mongo Gold                                                           |
| `dag_ds` | 02:00 (train: hàng tuần, suy luận: hàng ngày) | Feature engineering → Train (Spark MLlib) → Evaluate → Registry → Inference (đọc dữ liệu từ Mongo Gold do`dag_de` ghi) |

> Không cần 2 Airflow riêng — 1 instance, 2 DAG độc lập chạy lệch giờ, liên kết qua phụ thuộc dữ liệu (không trigger trực tiếp).

### 4.5. Phân tích & tổng hợp xu hướng cảm xúc (Gold layer, tính sẵn bằng Spark `groupBy().agg()`)

- **Theo sản phẩm**: tỷ lệ % tích cực/trung lập/tiêu cực mỗi sản phẩm → phát hiện sản phẩm bị đánh giá kém bất thường.
- **Theo thời gian**: tổng hợp theo tuần/tháng → đường xu hướng 3 lớp theo thời gian, phát hiện thời điểm bất thường.
- **Theo nhóm khách hàng**: nhóm theo `is_photo`, tần suất mua lặp lại, hoặc `seller_name` → so sánh xu hướng cảm xúc giữa các nhóm.
- **Theo phân khúc giá (Price Tier / Bracket)**: Join `reviews` với `products` theo `product_id` và phân nhóm giá (Bình dân: < 200k, Tầm trung: 200k – 1.5M, Cao cấp: > 1.5M) → phân tích xem phân khúc giá cao có khiến khách hàng khắt khe hơn không.
- **Tỷ lệ mâu thuẫn (Rating vs Text Discrepancy)**: tỷ lệ % đánh giá có sao cao nhưng nội dung tiêu cực theo ngành hàng/sản phẩm.

---

## 5. Cấu trúc thư mục dự án (gợi ý cho VS Code)

```
sentiment-analysis-project/
├── crawler/
│   ├── tiki_crawler.py
│   ├── lazada_crawler.py
│   └── schema_mapping.py          # chuẩn hoá schema chung Tiki + Lazada
├── annotation/
│   ├── labeling.py                # sinh label_weak và ma trận quyết định hợp nhất label_final
│   ├── phobert_annotator.py       # sinh label_text từ PhoBERT cho tập mẫu/benchmark
│   └── prompt.py / gemini_client.py # module LLM kiểm định các trường hợp mâu thuẫn
├── spark_jobs/
│   ├── etl_silver.py              # làm sạch, kiểm tra chất lượng dữ liệu, lọc nhãn nhiễu
│   ├── feature_engineering.py     # TF-IDF / Word2Vec
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── inference.py
│   └── aggregate_trends.py        # bảng tổng hợp theo sản phẩm/thời gian/nhóm KH
├── airflow/
│   └── dags/
│       ├── dag_de.py
│       └── dag_ds.py
├── model_registry/
│   └── (model đã lưu + metadata.json)
├── dashboard/
│   └── app.py                     # Streamlit/Dash kết nối MongoDB
├── notebooks/
│   └── eda.ipynb                  # khảo sát dữ liệu ban đầu
├── config/
│   └── settings.yaml
├── requirements.txt
└── README.md
```

---

## 6. Kế hoạch triển khai — 5 tuần, 1 người thực hiện

> Nội dung công việc giữ nguyên như kế hoạch gốc (10 tuần/2 người) — chỉ nén lại thời gian, không cắt bớt đầu việc. Mỗi tuần dưới đây cần làm việc với khối lượng gấp đôi so với phân bổ theo team ban đầu.

### Tuần 1 — Khởi tạo & nền tảng dữ liệu

- [x] Chốt phạm vi cụ thể (ví dụ: nhóm ngành hàng sẽ crawl).
- [x] Khảo sát API Tiki (product + review endpoint).
- [x] Khảo sát API Lazada (`catalog/?ajax=true`, `/pdp/review/getReviewList`), test khả năng bị chặn (baxia).
- [x] Thiết kế schema chung giữa Tiki và Lazada (product_id, review_id, content, rating, created_at, customer_id, is_photo, source...).
- [x] Khởi tạo repo, cấu trúc thư mục, `requirements.txt`, `.gitignore`.
- [x] Cài đặt môi trường local: HDFS (Docker single-node) hoặc thay thế bằng local filesystem mô phỏng, MongoDB local/Atlas, Spark local mode.
- [x] Viết `tiki_crawler.py` — crawl thử 1 sản phẩm, verify dữ liệu lấy được đúng schema.

### Tuần 2 — Hoàn thiện thu thập + Vùng thô + Vùng Silver & Gán nhãn

- [x] Viết `lazada_crawler.py`, xử lý retry/backoff khi gặp chặn.
- [x] Hoàn thiện crawler cho 150–300 sản phẩm/sàn, ưu tiên lọc thêm review rating thấp (1–3 sao) để tăng mẫu thiểu số.
- [x] Lưu dữ liệu thô vào HDFS (Bronze), partition theo nguồn/ngày.
- [x] Viết `schema_mapping.py` chuẩn hoá schema Tiki + Lazada về 1 schema chung.
- [x] Viết `etl_silver.py`: loại content rỗng/quá ngắn, khử trùng lặp, chuẩn hoá text tiếng Việt.
- [x] Xây dựng module sinh `label_weak` từ số sao rating.
- [x] Viết `phobert_annotator.py` sinh `label_text` cho tập mẫu benchmark kèm `confidence_score`.
- [x] Tích hợp thuật toán ma trận quyết định sinh `label_final` và gắn cờ `is_conflict`, áp dụng chiến lược Text-First sửa sai cho số sao.
- [x] Thêm bước kiểm tra chất lượng dữ liệu tự động trong `etl_silver.py` (tỷ lệ null, tỷ lệ trùng, phân bố nhãn, tỷ lệ mâu thuẫn).
- [x] Ghi dữ liệu Silver & Gold — schema: `product_id, content_clean, rating, label_weak, label_text, label_final, is_conflict, created_at, source...`.

### Tuần 3 — Mô hình phân loại cảm xúc + Airflow

- [x] Viết `feature_engineering.py`: TF-IDF Vectorizer (n-gram 1-2, sublinear TF) trích xuất đặc trưng trên 69k review hợp lệ.
- [x] Viết `train_model.py`: huấn luyện và so sánh đa mô hình (MultinomialNB baseline, ComplementNB, LogisticRegression Balanced, LinearSVC Calibrated).
- [x] Áp dụng xử lý mất cân bằng nhãn (class_weight='balanced' cho LogisticRegression/SVM và ComplementNB).
- [x] Viết `evaluate_model.py`: tính F1-macro, Accuracy, Confusion matrix, per-class Precision/Recall và phân tích mẫu nhầm lẫn (Error Analysis).
- [x] Thiết lập model registry đơn giản (lưu model `best_sentiment_model.joblib` + `metadata.json` gồm ngày train, chỉ số đánh giá, phiên bản dữ liệu).
- [x] Viết `inference.py`: dự đoán nhãn cho toàn bộ 225k review, phát hiện mâu thuẫn sao vs văn bản (`is_sentiment_conflict`), xuất tầng Gold `data/gold/gold_reviews.parquet`.
- [x] Viết `airflow/dags/dag_de.py` và `dag_ds.py`, cấu hình lịch chạy lệch giờ (01:00 / 02:00) kèm bộ điều phối chạy trực tiếp `src/run_pipeline.py` có log tiến độ và báo cáo JSON từng bước.

### Tuần 4 — Phân tích xu hướng + Dashboard

- [x] Viết `aggregate_trends.py`:
  - [x] Tổng hợp theo sản phẩm (tỷ lệ % mỗi lớp cảm xúc).
  - [x] Tổng hợp theo thời gian (tuần/tháng).
  - [x] Tổng hợp theo nhóm khách hàng (`is_photo`, tần suất mua lặp lại...).
  - [x] Tổng hợp theo phân khúc giá (Price Tier: join với metadata sản phẩm để phân tích tương quan giữa phân khúc giá và tỷ lệ hài lòng).
- [x] Lưu các bảng tổng hợp vào MongoDB (collection riêng cho dashboard, không query raw data mỗi lần) và xuất các bảng Parquet tại `data/gold/trends_by_*.parquet`.
- [x] Xây `dashboard/app.py` (FastAPI + Anfra-style Governed Semantic Layer + Apache ECharts 5 + Live AI Sentiment Sandbox) kết nối MongoDB / Gold Parquet, hiển thị các chiều phân tích và kèm tài liệu tích hợp Anfra (`docs/anfra_dashboard_guide.md`).
- [x] Chạy thử toàn bộ pipeline end-to-end 1 lần, ghi nhận lỗi phát sinh.

### Tuần 5 — Hoàn thiện, kiểm thử, báo cáo & demo

- [x] Kiểm thử lại toàn bộ pipeline (crawler → Bronze → Silver → Gold → model → dashboard) chạy qua Airflow/script điều phối (`src/run_pipeline.py`).
- [x] Rà soát lại chỉ tiêu dữ liệu (tổng số review, số lượng mỗi lớp nhãn) — ghi rõ phân bố thực tế và cơ chế lọc nhiễu trong báo cáo.
- [x] Viết `README.md` hướng dẫn chạy lại toàn bộ hệ thống.
- [ ] Hoàn thiện báo cáo: kiến trúc, kết quả mô hình, insight từ phân tích xu hướng; trình bày rõ luận điểm xử lý giới hạn dữ liệu (nhận diện việc thiếu lịch sử giá giao dịch và giải pháp chuẩn hóa thành Phân khúc giá sản phẩm - Price Tier).
- [ ] Chuẩn bị slide + demo dashboard trực tiếp.

---

## 7. Công nghệ sử dụng

| Thành phần             | Công nghệ đề xuất                                                                       |
| ------------------------ | -------------------------------------------------------------------------------------------- |
| Thu thập dữ liệu      | Python (`requests`) gọi API JSON của Tiki và Lazada                                     |
| Gán nhãn & Lọc nhiễu | PhoBERT (`transformers`) / Gemini API kết hợp ma trận quyết định rule-based          |
| Lưu trữ thô           | HDFS (hoặc Docker Hadoop single-node)                                                       |
| Xử lý phân tán       | PySpark / Spark MLlib (local mode hoặc Databricks Community Edition)                        |
| Lưu trữ phục vụ      | MongoDB                                                                                      |
| Điều phối             | Apache Airflow (1 instance, 2 DAG) — hoặc script Python mô phỏng nếu không kịp deploy |
| Trực quan hoá          | Streamlit / Dash / Power BI kết nối MongoDB                                                |

---

## 8. Rủi ro & phương án dự phòng

| Rủi ro                                                                                             | Phương án dự phòng                                                                                                                                                      |
| --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Lazada chặn IP khi crawl (cơ chế "baxia")                                                        | Thêm retry/backoff, dùng proxy khi cần, hoặc tạm dừng Lazada và tập trung mở rộng Tiki                                                                             |
| Dữ liệu mất cân bằng nhãn nặng (lớp tiêu cực/trung lập quá ít)                         | Cào có chủ đích theo rating thấp; áp dụng class weight/oversampling khi huấn luyện                                                                                 |
| Mâu thuẫn giữa số sao và văn bản (khách vote 5 sao nhưng chê thậm tệ) gây nhiễu nhãn | Áp dụng cơ chế lọc nhiễu (Denoising): đối chiếu`label_weak` và `label_text`, ưu tiên `label_text` khi confidence cao hoặc loại bỏ mẫu khỏi tập Train |
| Không kịp deploy Airflow thật                                                                    | Mô phỏng điều phối bằng script Python gọi tuần tự các bước, có log, nêu rõ vai trò orchestration trong báo cáo                                             |
| Tỷ lệ review thiếu nội dung text cao (~41% ở Tiki)                                             | Lọc bỏ ở tầng Silver, nêu rõ đây là giới hạn dữ liệu thực tế trong báo cáo                                                                                  |
| Thiếu lịch sử giá mua thực tế tại thời điểm review (chỉ có giá snapshot hiện tại)                | **Điểm cộng học thuật & hiểu sâu nghiệp vụ**: Nhận diện rõ giới hạn dữ liệu của API TMĐT công khai; không gộp giá vào từng dòng review mà tách biệt bảng Sản phẩm (Dimension), sau đó chuẩn hoá thành biến định tính **Phân khúc giá (Price Tier)** khi phân tích xu hướng thay vì cố phân tích chuỗi thời gian giá sai lệch. |
| Một mình xử lý cả DE + DS trong 5 tuần — dễ quá tải                                       | Ưu tiên baseline chạy được toàn trình (end-to-end) sớm (cuối tuần 3), sau đó mới tối ưu từng phần ở tuần 4–5                                            |
