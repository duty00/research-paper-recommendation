# Nguồn và giả định — kiểm tra ngày 19/10/2026

- Overview/Evaluation: https://www.kaggle.com/competitions/assignment-data-mining-261/overview
- Data: https://www.kaggle.com/competitions/assignment-data-mining-261/data
- Rules: https://www.kaggle.com/competitions/assignment-data-mining-261/rules
- Final selection: https://www.kaggle.com/competitions/assignment-data-mining-261/submissions

Evaluation chi tiết và Data mô tả trung bình điều hòa của relevance/recall. Abstract Overview mô tả 60% relevance chuẩn hóa, 40% coverage dựa trên title similarity. Bộ code v1 thực hiện Evaluation chi tiết, không khẳng định đã xem hoặc tái tạo mã chấm ẩn của host. Nhờ host xác nhận mâu thuẫn này; không dùng việc nộp thử để suy ra nhãn hoặc membership test ẩn.

Q không phải NDCG: nó là trung bình gain có trọng số vị trí, không chia cho ideal DCG. R không phải độ đa dạng tiêu đề trong công thức chi tiết. Public/Private mỗi bên 298 bài, chấm độc lập. Local validation phải báo cả Q và R. Độ dài list ảnh hưởng Q, nên điểm validation không thể coi là điểm Public dự kiến chính xác.

Kaggle hiển thị còn 3 lượt nộp trong ngày trên tài khoản chưa nộp; trước đó hiển thị tối đa 2 final submissions; kiểm tra ngày 22/10/2026 hiện hiển thị 0/3 final selections. Nếu chọn chưa đủ 2, giao diện nói Kaggle sẽ tự bổ sung từ các bài có điểm tốt. Người dùng chủ động chọn final sau khi kiểm tra, chương trình không tự nộp hoặc chọn bài.

Thời điểm Close trên giao diện: 15/12/2026 06:59:59 giờ Việt Nam. Hạn báo cáo/GitHub và giới hạn bổ sung còn phụ thuộc LMS. Không tạo lịch nhắc tự động.

Điểm môn A=0.60L+0.40R_report, làm tròn đến nấc 0.5. L dựa trên rank Private giữa các nhóm đủ điều kiện, không lấy raw Kaggle score nhân 10. Rules có thêm dòng Public thấp hơn baseline thì L=6.5; cần host giải thích cách áp dụng nếu có tình huống mâu thuẫn với bảng rank. Public benchmark của host được quan sát là 0.60953; code không dùng điểm này để huấn luyện.

Tối đa 5 thành viên, một nhóm/sinh viên. Tài khoản dùng email HCMUT và chia sẻ contact với host theo thông báo giảng viên. Không thay đổi các cài đặt tài khoản từ code.

Được dùng dữ liệu công khai, model pretrained hoặc mã bên thứ ba nếu khai báo và trích dẫn. V1 không dùng nguồn ngoài. Cấm lấy nhãn ẩn, hand-label test, suy ra Public/Private membership, dò nhãn qua feedback hoặc trao đổi lời giải riêng giữa các nhóm. Phải công khai mọi cách dùng test metadata không giám sát (v1 dùng danh sách venue để chọn phạm vi validation).

## Đọc lại ngày 22/10/2026

Evaluation chi tiết vẫn nêu Q là weighted mean gain, R là recall nhãn >1, score là 2QR/(Q+R); title similarity không nằm trong metric chi tiết. Abstract Overview vẫn nêu 60/40 và còn mâu thuẫn. metric.py không thay đổi; hidden scorer không được cung cấp.

Rules cho phép external public data/pretrained nếu khai báo và trích dẫn. V3 dùng frozen E5/MiniLM, supervised ordinal classification và các head được mô tả trong V3_SEMANTIC.md. Submissions hiện ghi Select up to 3 và 0/3; đây là số slot final, không phải xác nhận quota nộp còn lại trong ngày. Cần theo giao diện và LMS tại lúc chốt bài.
