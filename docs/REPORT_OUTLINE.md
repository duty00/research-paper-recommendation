# Dàn ý báo cáo — nhóm tự hoàn thiện bằng kết quả thực nghiệm

## Thông tin nộp

Ghi tên nhóm Kaggle, thành viên/MSSV, đóng góp, GitHub URL, final commit hash, submission ID, tên và SHA-256 của CSV đã nộp. Giữ đúng bản final CSV.

## 1. Bài toán và người dùng (1.5 điểm)

Giải thích sở thích đọc cá nhân của nhà nghiên cứu; nhãn không phải phán xét chất lượng khoa học phổ quát. Trình bày input/output, lựa chọn có đọc hay không, thứ tự ưu tiên, Q/R, tác động của false positive/negative và giới hạn metric.

## 2. Phân tích dữ liệu (1.5 điểm)

Dẫn số liệu từ data_audit.json: nhãn, thiếu tác giả, phân bố venue/year, tiêu đề và DOI/URL trùng, nhãn xung đột, train/test shift, định dạng RTF. Giải thích từng quyết định làm sạch. Tệp gốc và hash được giữ; không đoán nhãn test.

## 3. Insight và giả thuyết có thể kiểm chứng (1.5 điểm)

Liên hệ quan sát với lựa chọn mô hình: thiếu tác giả cần fallback; title có thuật ngữ chuyên ngành; venue có thể chứa thông tin sở thích nhưng phân bố train/test lệch; bài trùng có thể gây leakage. Phân biệt dữ kiện với giả thuyết. Kiểm tra bằng ablation/validation công bằng khi nhóm bổ sung thí nghiệm, không viết kết luận khi chưa có số liệu.

## 4. Thiết kế và triển khai (2 điểm)

Giải thích TF-IDF từ/character, encoding venue, tác giả và feature số; hồi quy hoặc phân loại được dùng ra sao; cách biến prediction thành gain, thứ tự và priority 0. Nêu tham số và lựa chọn ensemble. Tất cả feature được fit trong fold; khai báo dùng test venue names để xác định phạm vi đánh giá.

## 5. Đánh giá và phân tích lỗi (2 điểm)

Mô tả development CV, holdout giữ riêng, nhóm bài trùng, seed và protocol dùng đồng nhất. Bảng baseline và model gồm Q, R, score, độ lệch giữa folds, độ dài danh sách. Phân tích điểm development được dùng tuning nên lạc quan; holdout chỉ một mẫu, chưa phải bảo đảm tổng quát.

Dùng predictions/holdout_*.csv để xem ví dụ false positive (y=1 nhưng gain cao), false negative hoặc bài bị xếp thấp. Gắn với metadata và chủ đề; không gán nhãn hoặc dựng answer key cho test. Phân tích theo venue/year khi đủ cỡ mẫu. Phân tích leakage, sampling bias, leaderboard overfitting và độ không chắc chắn. Ghi mọi thay đổi protocol nếu xem holdout nhiều lần.

## 6. Kết luận (0.5 điểm)

Trả lời giả thuyết nào được dữ liệu hỗ trợ, hạn chế và hướng tiếp theo. Không biến điểm Public thành bằng chứng duy nhất. Chỉ điền điểm Kaggle thật sau khi nhóm nộp và ghi đúng submission ID.

## 7. Tái lập và trình bày (1 điểm trình bày xuyên suốt)

README, requirements, Python version, seed, thời gian/phần cứng, lệnh run/predict/check, expected metrics, exact commit và CSV. Trích dẫn tài liệu/thư viện/dữ liệu. Dùng biểu đồ có giải thích; ảnh leaderboard hay danh sách công cụ không thay thế phân tích.

## Bổ sung cho kết quả v2

- Dẫn phân tích lỗi v1 từ development OOF, có thể tái tạo bằng analyze.py. Phân biệt chọn tập đề xuất và thứ tự đọc trong tập đã chọn.
- Dẫn 576 cấu hình thử, shortlist 3 cấu hình cố định và kiểm tra seed 1262 từ search_plan.json/phase1.json/selection.json. Không gọi hai seed là hai test set độc lập.
- Holdout đã được xem ở v1: ghi rõ v2 chỉ đánh giá chẩn đoán trên tập cũ, không dùng nó chọn ứng viên. Không gọi đây là kiểm chứng độc lập hoàn toàn mới.
- Báo cáo cả phương pháp thử không cải thiện, chi phí chạy và giới hạn của việc tìm kiếm nhiều cấu hình.
- Ghi hai điểm Public thật của v1 từ EXPERIMENT_LOG.md. Chỉ thêm điểm v2 sau khi người dùng nộp; không dự báo thành tích leaderboard từ CV.

## Bổ sung cho v3 semantic

Giải thích giới hạn của TF-IDF và giả thuyết biểu diễn ngữ nghĩa; khai báo E5/MiniLM với revision, license, quantization, prefix, pooling và truncate. Phân biệt pretrained encoder cố định với head classification thực sự học từ train. Báo cả pure semantic và hybrid, Q/R, hai seed, fold variability và ví dụ lỗi development. Không dùng lại holdout đã xem như bằng chứng độc lập mới. Dẫn hash CSV, model/assets/dependency manifest và lệnh chạy không Docker. Chỉ điền điểm Kaggle v3 sau khi người dùng nộp.
