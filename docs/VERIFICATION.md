# Kiểm tra bản v1 — 19/10/2026

> **Kiểm chứng hiện hành V6:** bộ mã hiện có 30 test. V6 candidate 2 đã khớp CSV khi dự đoán từ bundle và huấn luyện lại độc lập; lệnh `reproduce.py train --verify-reference` cũng đã tạo đúng SHA-256 từ dữ liệu gốc. Xem [README](../README.md#reproducibility) và [REFERENCE_V6.json](REFERENCE_V6.json). Phần dưới là hồ sơ kiểm chứng các phiên bản trước.

Đã cài môi trường riêng với Python 3.12.14 64 bit và sáu phiên bản trong requirements.txt. `pip check` không phát hiện dependency hỏng.

Đã chạy 12 test, đều đạt:

- Công thức Q/R/harmonic mean đối chiếu tính tay.
- Đồng điểm theo ID dạng văn bản ("10" trước "2").
- Priority 0 bị loại; mọi số dương được tính; danh sách rỗng và không có bài đáng đọc.
- Từ chối priority âm, NaN, vô cực và ID trùng.
- Tỷ lệ chọn và mã hóa thứ tự ổn định khi đảo thứ tự input.
- RTF có dòng CSV và Unicode được đọc đúng.
- Nhóm trùng là thành phần liên thông, không vượt biên validation.
- CSV đúng bộ ID và giá trị hợp lệ.
- Lệnh dự đoán không ghi đè các tệp dữ liệu nguồn.
- ID và DOI/URL không nằm trong predictors.
- Token chỉ xuất hiện ở validation không lọt vào vocabulary đã fit.
- Lưu/tải cả mô hình hồi quy và phân loại cho prediction giống hệt.

Đã huấn luyện đầy đủ hai lần, cùng dữ liệu, seed 261 và 5 folds. Hai lần đều chọn Logistic Regression C=1 với tỷ lệ 65%, rồi Ridge alpha=1 với tỷ lệ 75%. SHA-256 của CSV tương ứng trùng nhau giữa hai lần huấn luyện. Nạp hai model đã lưu rồi dự đoán lại cũng tạo CSV có cùng hash.

| Tệp | SHA-256 | Số dòng | Bài được đề xuất |
|---|---|---:|---:|
| candidate_1.csv | 1698e9f8827cd803450eda2ca1c9ca570df8c86ca1ead41c3c645d357c815525 | 596 | 388 |
| candidate_2.csv | 75dae7735398ac29ebab05da4d1b5b2cc80f4241b32571aeb25b294f9afcd283 | 596 | 447 |

Đã đối chiếu tất cả 2.494 ID train với splits.csv; nhóm trùng không vượt giữa development/holdout hoặc giữa fit/validation trong cả 5 fold. Tất cả SHA-256 của dữ liệu nguồn giữ nguyên. Hash code và output khớp manifest của lần chạy tham chiếu.

Lần chạy tham chiếu mất khoảng 68 giây trên máy hiện tại, giới hạn 1 luồng. Đây là thời gian thực đo của lần đó, không phải cam kết thời gian trên máy khác. Chưa đo peak RAM.

Đã kiểm tra cú pháp của setup.ps1 và run.ps1. Các lệnh Python đã được chạy thực tế; script setup trên một bản Python Windows cài mới thông thường chưa được chạy trong phiên này. Môi trường kiểm tra dùng Python đi kèm Codex, venv riêng và bootstrap pip vì bản runtime đó không hỗ trợ ensurepip mặc định. Người dùng/giảng viên dùng Python chính thức 3.11/3.12 theo README.

Khi thực hiện kiểm tra tham chiếu ở trên, chưa nộp Kaggle. Sau đó người dùng đã nộp v1 và cung cấp hai điểm Public: 0.61105 và 0.60591 (xem EXPERIMENT_LOG.md). Chưa kiểm chứng hidden scorer, điểm Private hoặc tính tái lập bit-for-bit trên CPU/OS khác. Hai CSV là ứng viên baseline v1, không phải lựa chọn final của nhóm.

## Kiểm tra bản v2 — 19/10/2026

Môi trường thực đo: Python 3.12.10 Windows x64, sáu thư viện cùng phiên bản requirements.txt, CPU giới hạn một luồng. Toàn bộ 17 test đạt, gồm 12 test gốc và 5 test bổ sung:

- Chọn theo xác suất đọc nhưng xếp các bài đã chọn theo expected gain.
- Đồng điểm ở bước chọn theo thứ tự ID dạng văn bản.
- Từ chối shape không hợp lệ, NaN và component ngoài [0, 1].
- Ordinal, NB-weighted ordinal, hurdle: không đưa token chỉ có trong validation vào từ vựng; lưu/tải model cho kết quả giống hệt.
- Đối chứng v1 trong engine v2 cho cùng expected gain với engine gốc.

Đã chạy đầy đủ 576 cấu hình trên development seed 262; kiểm tra 3 cấu hình cố định với seed 1262. Hai seed dùng cùng development, không phải hai tập test độc lập. Baseline v1 seed 262 tái lập chính xác 0.5994405134316391. Kiểm tra group và toàn bộ fold ID khớp lần v1 đã chạy trên máy.

Lưu selection.json trước khi đánh giá holdout cũ. Không điều chỉnh lựa chọn sau khi xem holdout. V2 #1 có holdout gần ngang, hơi thấp hơn v1; v2 #2 có holdout tốt hơn. Tại thời điểm kiểm tra local chưa có điểm Kaggle v2. Cập nhật 22/10/2026 từ ảnh: ứng viên 1 Public 0.62516, ứng viên 2 Public 0.61864; chưa có Private score. Xem EXPERIMENT_LOG.md.

| CSV | Số ID | Đề xuất | SHA-256 |
|---|---:|---:|---|
| v2_candidate_1.csv | 596 | 388 | 3fe3fa7f5d0ad805f13faadf791d903f80ec37f628e678acf1af05459be688fc |
| v2_candidate_2.csv | 596 | 418 | 3c339f52e0877d933674c1e25ed8622a9378bfb00153ab669b922df51bb75384 |

Đã dự đoán lại từ cả hai bundle model đã lưu và fit lại độc lập hai mô hình đã chọn trên toàn bộ train. Mỗi cách đều tạo CSV có SHA-256 giống bản v2 gốc. Lần kiểm tra lại không chạy lại toàn bộ không gian 576 cấu hình; nó kiểm chứng hai bước refit cuối và saved-model inference. Báo cáo máy đọc được: outputs/v2_verification/verification.json.

Đã đối chiếu toàn bộ code/output được ghi trong manifest v2; tất cả khớp. Hash ba tệp dữ liệu gốc và hai CSV v1 đã nộp giữ nguyên. Thời gian chạy đầy đủ v2: 298.8 giây; chưa đo peak RAM. Không cài thêm thư viện, tải model hoặc dùng Kaggle API.

setup.ps1 bổ sung kiểm tra native Windows x64 Python và ưu tiên .venv đang có. Hướng dẫn README ghi cách dùng Python Install Manager khi lệnh python trỏ đến MSYS2/MinGW. Không yêu cầu Docker. Thay đổi setup được kiểm tra cú pháp và lệnh kiểm tra interpreter; chưa cài lại môi trường từ đầu trên một máy Windows sạch.

## Kiểm tra v3 — 22/10/2026

20/20 test đạt và pip check không phát hiện dependency hỏng. Ba test mới: masked mean pooling không lấy padding và L2 normalization; centering/venue encoder chỉ fit training; các semantic head ordinal/multiclass/SVC/Ridge lưu-tải chính xác và dự đoán trong [0,1].

Đã chạy 594 cấu hình development seed262 và xác nhận ba cấu hình cố định với seed1262, rồi chốt hai ứng viên trước chẩn đoán holdout cũ. V2 baseline tái lập đúng CV 0.6090860506382574 và 0.6090518605640872.

Đã dùng một cache mới để chạy lại toàn bộ title test qua E5 ONNX. Inference từ hai bundle và các fit full-data độc lập đều cho CSV trùng SHA-256 với bản gốc. Không chạy lại toàn bộ search CV trong bước replay. Dữ liệu gốc, bốn CSV v1/v2 và mọi code/output được manifest v3 ghi đều khớp hash. Báo cáo: outputs/v3_verification/verification.json.

Encoder: MiniLM FP32, E5 INT8, chỉ chạy CPU; assets khoảng 125.8 MB trong D:\DM. Thời gian search sau cache 47.471 giây, không tính tải/encode đầu. Chưa đo peak RAM, chưa kiểm tra một máy sạch hoặc OS khác. Chưa nộp v3 lên Kaggle.
