# BÁO CÁO ĐÁNH GIÁ THỰC TRẠNG VÀ HẠN CHẾ CỦA HỆ THỐNG CÀO DỮ LIỆU LAZADA
**Đề tài:** Phân tích cảm xúc đánh giá sản phẩm trực tuyến (Sentiment Analysis of Online Product Reviews)  
**Phạm vi nghiên cứu:** Nền tảng Thương mại điện tử Đa nguồn (Tiki & Lazada)  
**Tác giả:** Nhóm thực hiện Đồ án Big Data  
**Ngày lập báo cáo:** 18/10/2026  

---

## 1. TỔNG QUAN VÀ MỤC ĐÍCH

Trong quá trình xây dựng Lakehouse và luồng thu thập dữ liệu (Data Ingestion Pipeline) cho đề tài *Phân tích cảm xúc đánh giá sản phẩm trực tuyến*, nhóm nghiên cứu đã triển khai hệ thống cào dữ liệu tự động từ hai sàn thương mại điện tử lớn tại Việt Nam là **Tiki** và **Lazada**.

Tài liệu này được lập nhằm:
1. **Cung cấp bằng chứng thực nghiệm (Empirical Evidence)** về các hạn chế kỹ thuật và chất lượng dữ liệu của tập dữ liệu Lazada so với Tiki.
2. **Phân tích nguyên nhân kỹ thuật chuyên sâu (Root Cause Analysis)** xuất phát từ cơ chế bảo mật chống bot (WAF/Anti-scraping) và kiến trúc API của sàn Lazada.
3. **Phân tích bài toán đánh đổi chi phí - cơ hội (Cost-Benefit Analysis)** giữa việc "đập đi xây lại" crawler Lazada so với việc tập trung hoàn thiện các tầng cốt lõi của môn học Big Data (Spark ETL, MLlib, Airflow, Dashboard).
4. **Đưa ra định hướng xử lý kiến trúc (Architectural Strategy)**: Giữ lại tập dữ liệu hiện có để phục vụ mục tiêu xử lý đa nguồn (Multi-source Harmonization) và đưa các hạn chế này vào báo cáo chính thức như một minh chứng cho tư duy phản biện khoa học.

---

## 2. BẰNG CHỨNG THỰC NGHIỆM VÀ ĐỐI SOÁT DỮ LIỆU

Dựa trên kết quả thu thập thực tế tại tầng `Bronze` và quá trình tiền xử lý tại tầng `Silver` (được ghi nhận trong báo cáo chất lượng dữ liệu `data/silver/dq_report.json`), sự khác biệt giữa hai sàn được tổng hợp dưới bảng số liệu sau:

### Bảng 1: Bảng đối soát kỹ thuật giữa nguồn Tiki và Lazada

| Tiêu chí so sánh | Nguồn Dữ liệu Tiki | Nguồn Dữ liệu Lazada | Đánh giá & Tác động kỹ thuật |
| :--- | :--- | :--- | :--- |
| **Quy mô thu thập (Volume)** | **221.515 bản ghi** (~63.1 MB) | **4.145 bản ghi** (~1.6 MB) | Lazada chỉ chiếm **~1,84%** tổng dung lượng dữ liệu thu thập được. |
| **Giao thức thu thập (Protocol)** | HTTP REST API công khai (`/api/v2/reviews`) | Trình duyệt không đầu (Headless Browser Playwright) | Cào Lazada tốn tài nguyên CPU/RAM gấp hàng chục lần và tốc độ chậm hơn đáng kể. |
| **Tỷ lệ nhãn 5 sao (Positive)** | 190.495 (86,00%) | **4.078 (98,38%)** | Dữ liệu Lazada bị thiên lệch cực đoan về nhãn tích cực (Extreme Positivity Bias). |
| **Tỷ lệ nhãn 4 sao** | 21.895 (9,88%) | 45 (1,09%) | Phân bố bình thường ở Tiki, rất thấp ở Lazada. |
| **Tỷ lệ nhãn 3 sao (Neutral)** | 3.815 (1,72%) | 22 (0,53%) | Dữ liệu trung tính rất hạn chế trên Lazada. |
| **Tỷ lệ nhãn 1-2 sao (Negative)** | **5.310 (2,40%)** *(1 sao: 3.743; 2 sao: 1.567)* | **0 (0,00%)** *(Không có bất kỳ mẫu nào)* | **Hạn chế nghiêm trọng:** Không thể dùng tập Lazada đơn độc để huấn luyện mô hình phân loại cảm xúc. |
| **Trường `customer_id`** | Đầy đủ định danh số | **NULL 100% (4.145/4.145)** | Không thể phân tích hành vi người dùng lặp lại (Customer Profile / Retention). |
| **Trường `seller_name`** | Đầy đủ thông tin gian hàng | **NULL 100% (4.145/4.145)** | Không thể phân tích uy tín người bán trên Lazada qua review. |
| **Trường `title`** | Đầy đủ tiêu đề đánh giá | **NULL 100% (4.145/4.145)** | Lazada thiết kế giao diện không có tiêu đề bình luận riêng biệt. |
| **Trường `thank_count` & `comment_count`** | Có thông tin (trung bình 0,134 lượt cảm ơn) | **NULL 100% (4.145/4.145)** | Lazada không cung cấp thông tin tương tác này qua crawler. |

```
BIỂU ĐỒ SO SÁNH PHÂN BỐ XẾP HẠNG SAO (RATING DISTRIBUTION)

Tiki (221.515 reviews):
[5 sao] ████████████████████████████████████████ 86.00% (190.495)
[4 sao] ████ 9.88% (21.895)
[3 sao] █ 1.72% (3.815)
[1-2 sao] █ 2.40% (5.310) -> Đủ dữ liệu tiêu cực để học máy

Lazada (4.145 reviews):
[5 sao] ████████████████████████████████████████ 98.38% (4.078)
[4 sao] ▏ 1.09% (45)
[3 sao] ▏ 0.53% (22)
[1-2 sao] 0.00% (0) -> Mất cân bằng dữ liệu tuyệt đối
```

---

## 3. PHÂN TÍCH NGUYÊN NHÂN KỸ THUẬT CHUYÊN SÂU

### 3.1. Hệ thống tường lửa WAF và Cơ chế chống Bot (Alibaba Security / Akamai WAF)
* **Thử thách Captcha trượt (Slider Captcha):**  
  Lazada áp dụng giải pháp bảo mật của Tập đoàn Alibaba (Sec-Ch-Ua, thuật toán nhận diện chữ ký hành vi người dùng). Khi crawler gửi nhiều truy vấn liên tiếp hoặc mở trang đánh giá `/product-reviews/?item_id=...`, hệ thống lập tức chặn và điều hướng về trang kiểm tra bảo mật `https://www.lazada.vn/punish...`.
* **Cơ chế Cookie phiên `x5sec`:**  
  Để tiếp tục truy cập, phiên làm việc bắt buộc phải có cookie `x5sec` được cấp sau khi giải thành công captcha. Trong mã nguồn `src/crawler/solve_lazada_captcha.py`, nhóm đã phải phát triển script can thiệp để người vận hành kéo thanh trượt thủ công nhằm lấy cookie. Điều này ngăn cản hoàn toàn khả năng chạy tự động định kỳ (Fully Automated Ingestion) không người giám sát trên Apache Airflow.
* **Cơ chế khóa IP nghiêm ngặt (IP Rate-limiting):**  
  Nếu tăng tốc độ cào hoặc chạy đa luồng trên một địa chỉ IP công cộng (Home/Lab IP), Lazada sẽ áp dụng Soft-ban từ 30 phút đến vài giờ, khiến toàn bộ pipeline thu thập dữ liệu bị gián đoạn.

### 3.2. Mã hóa chữ ký API (Encrypted MTop Signature)
* Khác với Tiki cung cấp REST API mở trả về JSON rõ ràng, API nội bộ của Lazada (`mtop.lazada.review.item.getpcreviewlist`) yêu cầu kèm theo chuỗi `sign` (được băm MD5/HMAC từ token động, thời gian `t`, appKey và payload dữ liệu).
* Do việc đảo ngược mã hóa (reverse engineering) thuật toán sinh token của Alibaba nằm ngoài phạm vi đồ án, crawler buộc phải dùng Playwright để tải toàn bộ trang HTML/DOM. Phương pháp này khiến tốc độ xử lý chậm hơn từ 50 đến 100 lần so với việc gọi trực tiếp REST API.

### 3.3. Thiên lệch hiển thị mặc định của nền tảng (Platform Algorithmic Bias)
* **Sắp xếp mặc định ưu tiên 5 sao:** Thuật toán giao diện người dùng của Lazada mặc định sắp xếp các bình luận "Có ích nhất" hoặc "Có ảnh hưởng nhất" lên các trang đầu tiên, trong đó phần lớn là đánh giá 5 sao kèm hình ảnh.
* **Tương tác bộ lọc phức tạp:** Để lấy được các đánh giá 1 sao và 2 sao, crawler buộc phải mô phỏng thao tác click chuột vào nút lọc phân tầng rating trên giao diện. Tuy nhiên, các selector CSS của Lazada được tự động làm mờ (obfuscated class names) và thay đổi thường xuyên, dẫn đến tỷ lệ lỗi phân tích cú pháp (DOM Parser Error) rất cao.

### 3.4. Nhiễu văn bản từ mẫu gợi ý giao diện (UI Aspect Prompts & Escape Artifacts)
* **Ký tự escape xuống dòng dạng chuỗi thô (`\n`):** Do dữ liệu đánh giá của Lazada được lưu trữ và bóc tách từ chuỗi JSON/DOM, có **644 dòng** tồn tại chuỗi ký tự escape `\n` thay vì ký tự ngắt dòng ASCII thực tế. Nếu không tiền xử lý đặc biệt, chuỗi này sẽ làm dính từ và gây lỗi cho các bộ tách từ NLP (Tokenizer).
* **Nhiễu nhãn khía cạnh có sẵn của sàn (Aspect Template Prompts):** Khi đánh giá trên ứng dụng hoặc web Lazada, sàn cung cấp sẵn các khung tiêu chí để người dùng điền theo mẫu (ví dụ: *"Chất lượng:"*, *"Giá cả:"*, *"Hương vị:"*, *"Đáng tiền:"*, *"Thiết kế:"*, hoặc tiếng Anh *"Taste："*, *"Price："*, kèm dấu hai chấm toàn góc `：`).
* **Hậu quả tăng giả độ dài từ (`word_count`):** Các nhãn giao diện này vô tình làm tăng giả số từ của bình luận. Thậm chí có những đánh giá người dùng chỉ chọn khung gợi ý (ví dụ: *"Đáng tiền："*) mà không viết bất kỳ nội dung nào. Sau khi áp dụng bộ lọc regex loại bỏ triệt để các nhãn khía cạnh này ở tầng Silver, số lượng review thực sự đủ tiêu chuẩn huấn luyện NLP (`is_text_usable`) của Lazada được chuẩn hóa chính xác từ **3.866 dòng** về **3.828 dòng**, loại bỏ hoàn toàn các mẫu rác.

---

## 4. PHÂN TÍCH ĐÁNH ĐỔI VÀ CHI PHÍ CƠ HỘI (COST-BENEFIT ANALYSIS)

Nếu nhóm quyết định dừng lại để nâng cấp hệ thống cào Lazada đạt đến độ hoàn hảo tương đương Tiki:

| Hạng mục đầu tư | Chi phí & Khối lượng công việc cần bỏ ra | Giá trị mang lại cho đồ án Big Data |
| :--- | :--- | :--- |
| **Proxy xoay vòng** | Phải đăng ký dịch vụ Residential Rotating Proxy trả phí để tránh WAF ban IP. | Không đóng góp vào mục tiêu kiến trúc Big Data (chỉ là thủ thuật mạng). |
| **Dịch vụ Captcha** | Tích hợp dịch vụ giải mã Captcha bên thứ ba (như 2Captcha, Anti-Captcha qua API). | Tăng chi phí tài chính và độ phức tạp mã nguồn không cần thiết. |
| **Thời gian kỹ thuật** | Dự kiến mất từ **2 đến 3 tuần** liên tục gỡ lỗi DOM và duy trì phiên đăng nhập. | **Đánh mất thời gian làm các hạng mục cốt lõi**: Spark Streaming/Batch, Data Quality Test, Machine Learning Modeling, Airflow Scheduling, Dashboard. |

> **KẾT LUẬN VỀ CHI PHÍ CƠ HỘI:**  
> Đề tài này thuộc môn học **Kỹ thuật Dữ liệu Lớn (Big Data Engineering)**, không phải môn "Kỹ thuật Lách Tường lửa & Khai thác Web". Do đó, việc tiêu tốn thời gian vào cuộc chiến chống bot với một sàn thương mại điện tử cấp tập đoàn như Lazada là **sự đánh đổi không đáng giá và đi lệch khỏi mục tiêu đào tạo của đồ án**.

---

## 5. ĐỊNH HƯỚNG KIẾN TRÚC & GIẢI PHÁP ĐỀ XUẤT

Nhóm thống nhất phương án xử lý kỹ thuật tối ưu như sau:

### 5.1. Vẫn duy trì tập dữ liệu Lazada (4.145 bản ghi) trong Lakehouse
* **Chứng minh năng lực Schema Harmonization:** Tại tầng `Silver` (trong job Spark `etl_silver.py`), hệ thống đọc đồng thời cả hai file `tiki_reviews.csv` và `lazada_reviews.csv`. Bằng cách ánh xạ các cột không đồng nhất, gán `NULL` chuẩn xác cho `customer_id`, `seller_name` và phân loại cờ `source = 'lazada'`, pipeline thể hiện trọn vẹn khả năng xử lý **Dữ liệu Đa nguồn không đồng nhất (Heterogeneous Data Ingestion)**.
* **Gia tăng tính đa dạng cho Business Analytics:** Trên giao diện Dashboard (Streamlit/Superset), dữ liệu Lazada vẫn mang lại giá trị so sánh kinh doanh trực quan (ví dụ: so sánh giá bán, tỷ lệ để lại hình ảnh giữa hai sàn).

### 5.2. Sử dụng Tiki làm tập dữ liệu chuẩn cho Machine Learning
* Do tập dữ liệu Tiki đạt quy mô lớn (>220.000 dòng) và có phân bố đầy đủ các lớp cảm xúc (5.310 mẫu tiêu cực), tập dữ liệu Tiki sẽ được dùng làm tập dữ liệu huấn luyện chính cho mô hình Phân loại Cảm xúc (Spark MLlib / Transformers).
* Tập dữ liệu Lazada sẽ được sử dụng làm **Tập thử nghiệm suy diễn (Out-of-domain Inference Test)** để kiểm tra xem mô hình học từ Tiki có thể áp dụng tốt sang văn phong đánh giá của khách hàng Lazada hay không.

### 5.3. Trình bày minh bạch trong Báo cáo Đồ án
* Nhóm sẽ đưa toàn bộ nội dung của tài liệu này vào phần **"Hạn chế của Đề tài (Limitations)"** và **"Định hướng Phát triển (Future Work)"** trong Báo cáo Tổng kết và Slide thuyết trình.
* Việc chỉ ra sự thiên lệch (Selection Bias), mất cân bằng nhãn (Label Imbalance) và rào cản WAF của Lazada sẽ chứng minh nhóm thực sự hiểu sâu sắc về vòng đời dữ liệu thực tế (Data Lifecycle) và có tư duy nghiên cứu trung thực, chuyên nghiệp.
