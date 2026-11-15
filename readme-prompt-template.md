# README Generator Prompt (dùng với AI CLI)

Hãy đọc toàn bộ codebase trong thư mục hiện tại (source code, package.json/requirements.txt,
notebook, config, thư mục dataset nếu có) và viết cho tôi một file README.md hoàn chỉnh,
chuyên nghiệp, chuẩn để đưa lên GitHub public repo.

## Yêu cầu bắt buộc — README phải có đủ các phần sau, theo đúng thứ tự:

1. **Tiêu đề + mô tả ngắn** — tên project và 1-2 câu nêu rõ project giải quyết vấn đề gì.
   Thêm badge tương ứng (build, license, ngôn ngữ chính, version) nếu phù hợp.

2. **Demo / Screenshot / GIF** — nếu project có output trực quan (UI, dashboard, biểu đồ,
   ảnh output của model), chèn placeholder ảnh dạng `![demo](assets/demo.png)` và ghi chú
   cho tôi biết tôi cần chụp/thêm ảnh gì vào đó.

3. **Tính năng chính (Features)** — liệt kê gạch đầu dòng, chỉ nêu chức năng nổi bật,
   không liệt kê chi tiết vụn vặt.

4. **Tech Stack** — ngôn ngữ, framework, thư viện chính đã dùng, tự phát hiện từ
   package.json/requirements.txt/import trong code. Có thể dùng shields.io badge.

5. **Cài đặt (Installation)** — các bước chạy được từ đầu: clone, cài dependencies,
   biến môi trường (.env nếu có), migration/setup database nếu có. Viết dạng code block
   copy-paste chạy được ngay, tự suy ra lệnh đúng từ file cấu hình thực tế của project
   (không bịa lệnh generic).

6. **Cách sử dụng (Usage)** — ví dụ lệnh chạy, API call mẫu, hoặc input/output mẫu,
   lấy đúng theo entry point thực tế của code (main.py, index.js, app.py...).

7. **Cấu trúc project (Project Structure)** — cây thư mục ngắn gọn (chỉ các file/folder
   quan trọng), tự sinh từ cấu trúc thư mục thật.

8. **Kết quả / Đánh giá (Results)** — nếu là project ML/Data: các chỉ số (accuracy,
   ROC-AUC, RMSE...), bảng so sánh model nếu có trong code/notebook/log. Nếu không tìm
   thấy số liệu, để placeholder và ghi chú cho tôi tự điền.

9. **Dataset** — nếu có, nêu nguồn, số lượng mẫu, link (nếu public). Tự phát hiện từ
   code load dữ liệu.

10. **License** — hỏi tôi nếu chưa có file LICENSE, hoặc mặc định gợi ý MIT.

11. **Liên hệ / Tác giả** — để placeholder tên, email, LinkedIn tôi tự điền.

## Nguyên tắc khi viết

- Đọc code thật để lấy thông tin chính xác (tên hàm, entry point, dependencies thật),
  không tự bịa tính năng hay công nghệ không có trong code.
- Nếu thiếu thông tin nào (ví dụ số liệu kết quả, ảnh demo, license), đừng bịa —
  để `<!-- TODO: điền ... -->` và liệt kê rõ ở cuối những gì tôi cần bổ sung thủ công.
- Văn phong ngắn gọn, chuyên nghiệp, ưu tiên gạch đầu dòng hơn đoạn văn dài.
- Xuất kết quả là 1 file README.md hoàn chỉnh, không giải thích dài dòng bên ngoài file.
