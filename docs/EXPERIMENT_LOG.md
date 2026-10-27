# Lịch sử thực nghiệm và nộp Kaggle

## Tổng hợp đến 27/10/2026

Điểm Public dưới đây do nhóm cung cấp từ ảnh kết quả Kaggle; chưa có điểm Private. Ngày tổng hợp không phải ngày nộp của tất cả các bản.

| Phiên bản | Public score |
|---|---:|
| V3 candidate 1 | 0.63502 |
| V4 candidate 1 | 0.66016 |
| V5 candidate 1 | 0.66154 |
| V6 candidate 1 | 0.65835 |
| **V6 candidate 2** | **0.66504** |

V6 candidate 2: `bge_authors75_mix25`, fraction `0.6`, policy `utility_half`. CV seed361 = `0.6295670034`; seed1361 = `0.6282070982`; tăng trung bình `0.0019688418` so với V5 trên cùng giao thức 7-fold và thắng 9/14 fold. Hai seed dùng lại cùng dữ liệu; đây là đánh giá thích nghi, không phải holdout độc lập.

Candidate 2 được xuất riêng khi người dùng hỏi sau kết quả Public candidate 1. Tham số của candidate 2 đã cố định trong shortlist trước khi xem điểm Public candidate 1; không chạy thêm search khi xuất bản thứ hai. Nhánh encoder lớn của V6 không qua tiêu chí xác nhận và không tạo CSV.

Tái tạo cấu hình đã chọn từ dữ liệu gốc bằng [README](../README.md#setup); hash và cấu hình ở [REFERENCE_V6.json](REFERENCE_V6.json). Các mục phía dưới lưu lịch sử V1–V3 tại thời điểm tương ứng.

## Kết quả v1 đã nộp

Ghi nhận ngày **19/10/2026** từ ảnh kết quả do người dùng cung cấp. Đây là điểm **Public**; chưa có điểm Private. Người dùng tự upload; chương trình không gửi bài qua Kaggle API. Chưa ghi lại submission ID nên không tự suy đoán ID hoặc giờ upload.

| File | Mô tả khi nộp | Public score | SHA-256 CSV |
|---|---|---:|---|
| candidate_1.csv | v1_logistic_C1_top65_seed261 | **0.61105** | 1698e9f8827cd803450eda2ca1c9ca570df8c86ca1ead41c3c645d357c815525 |
| candidate_2.csv | v1_ridge_alpha1_top75_seed261 | 0.60591 | 75dae7735398ac29ebab05da4d1b5b2cc80f4241b32571aeb25b294f9afcd283 |

Nguồn CSV local: `outputs/my_run01/submissions/`. Không thay đổi các file đã nộp. Khi viết báo cáo cuối, bổ sung submission ID, thời gian nộp và commit mã nguồn tương ứng từ lịch sử Kaggle.

## V2 — thiết kế trước khi xem kết quả

V2 xuất phát từ phân tích lỗi **OOF development**, không suy luận nhãn test từ leaderboard:

- Phạm vi test venue trong development: 1.646 bài.
- V1 chọn nhầm 274 bài nhãn 1, bỏ sót 221 bài nhãn 2–5, trong đó 35 bài nhãn 4–5.
- KR: 121 bài nhãn 1 được chọn, 13 bài đáng đọc bị bỏ sót.
- LICS: 54 bài nhãn 1 được chọn, 126 bài đáng đọc bị bỏ sót.
- Đây là số liệu mô tả từ dự đoán ngoài fold; chưa chứng minh một cách sửa cụ thể sẽ cải thiện trên test.

### Những thay đổi được thử

1. Title TF-IDF chi tiết hơn; đối chiếu title-only và metadata với trọng số nhỏ hơn.
2. Phân loại đa lớp, bốn classifier ordinal, mô hình hai bước đọc/độ phù hợp, NB-weighted ordinal logistic và SVD + RBF SVC.
3. Ba ensemble có trọng số cố định.
4. Dùng expected gain, xác suất nhãn > 1 hoặc hai tổng có trọng số cố định để chọn tập đề xuất. Trong tập đã chọn, luôn xếp theo expected gain.

NB-weighted ordinal lấy ý tưởng log-count ratio từ [Wang & Manning, 2012](https://aclanthology.org/P12-2018/); dùng Logistic Regression và chuẩn hóa L2, không phải bản sao chính xác NB-SVM trong bài báo. Không có dữ liệu bên ngoài hoặc mô hình pretrained.

### Giao thức lựa chọn

- Giữ nguyên 2.018 bài development và 476 bài holdout của v1, xác minh bằng hash dữ liệu và file splits.
- Seed 262: 5-fold StratifiedGroupKFold, fit toàn bộ biến đổi trong từng fold. 13 model cơ sở và 3 blend, 4 policy, 9 tỷ lệ chọn: **576 cấu hình**.
- Giữ cấu hình tốt nhất của từng model; lấy 3 model đứng đầu, giữ nguyên policy và tỷ lệ, kiểm tra với 5 folds seed 1262.
- V1 Logistic C=1, policy gain, tỷ lệ .65 là baseline cố định ở cả hai seed.
- Tối đa 2 ứng viên, yêu cầu điểm trung bình hai seed tăng ít nhất .002 và seed xác nhận tốt hơn baseline cùng seed.
- Hai seed dùng cùng development: đây là kiểm tra độ ổn định theo cách chia, **không phải hai tập kiểm thử độc lập**. Điểm CV đã dùng để lựa chọn nên có thể lạc quan.
- Lưu `selection.json` trước khi đánh giá holdout. Holdout đã được xem ở v1; ở v2 chỉ là chẩn đoán, không còn là kiểm thử hoàn toàn mới. Không đổi lựa chọn theo kết quả này.
- Nếu không ứng viên nào qua ngưỡng, không xuất CSV v2. Không cần dùng hết quota Kaggle mỗi ngày.

Kết quả thực đo và cấu hình cuối ở `outputs/v2_run01/reports/summary.md`, `results.json`, `selection.json` và `search_plan.json`. Điểm Public của v1 được lưu làm lịch sử, không đi vào hàm chọn cấu hình v2.

## Kết quả v2 đã đo

| Mô hình | CV seed 262 | CV seed 1262 | CV trung bình | Holdout cũ (chẩn đoán) |
|---|---:|---:|---:|---:|
| V1 Logistic C=1, top 65% | 0.599441 | 0.597583 | 0.598512 | 0.588169 |
| V2 #1: ordinal_title_c1, top 65% | 0.609086 | 0.609052 | 0.609069 | 0.587912 |
| V2 #2: ordinal_rich_c2, top 70% | 0.609626 | 0.606670 | 0.608148 | 0.598096 |

- v2_candidate_1.csv: ordinal_title_c1, policy=gain, fraction=0.65; 388/596 bài đề xuất; SHA-256 `3fe3fa7f5d0ad805f13faadf791d903f80ec37f628e678acf1af05459be688fc`.
- v2_candidate_2.csv: ordinal_rich_c2, policy=gain, fraction=0.7; 418/596 bài đề xuất; SHA-256 `3c339f52e0877d933674c1e25ed8622a9378bfb00153ab669b922df51bb75384`.

Thời gian chạy 298.8 giây. Khi hoàn tất lần chạy ngày 19/10/2026, chưa có điểm Kaggle v2; kết quả được xác nhận sau đó ở mục dưới. Không đổi ứng viên hoặc tỷ lệ sau khi xem holdout. Title-only có holdout hơi thấp hơn v1 dù CV tốt hơn; bản metadata cải thiện trên tập chẩn đoán cũ. Giữ cả hai để kiểm tra khả năng tổng quát, không khẳng định v2 sẽ thắng leaderboard.

## Kết quả Kaggle v2 — xác nhận ngày 22/10/2026

Nguồn: ảnh trang Submissions người dùng cung cấp ngày 22/10/2026, hiển thị trực tiếp tên file, trạng thái Complete và Public score. Đây là ngày ghi nhận; không suy đoán ngày/giờ nộp từ mốc tương đối trong ảnh. Chưa có submission ID và chưa có điểm Private. Hash bên dưới là của file local đã kiểm chứng; ảnh không cung cấp hash phía Kaggle để đối chiếu byte-for-byte.

| File | Mô hình | Public score | Chênh lệch so với Logistic v1 | SHA-256 local |
|---|---|---:|---:|---|
| v2_candidate_1.csv | ordinal_title_c1, top 65% | **0.62516** | +0.01411 | 3fe3fa7f5d0ad805f13faadf791d903f80ec37f628e678acf1af05459be688fc |
| v2_candidate_2.csv | ordinal_rich_c2, top 70% | 0.61864 | +0.00759 | 3c339f52e0877d933674c1e25ed8622a9378bfb00153ab669b922df51bb75384 |

Ứng viên 1 cao hơn ứng viên 2 là 0.00652 trên Public. Ứng viên 1 cũng có CV trung bình hai seed cao hơn (0.609069 so với 0.608148); ứng viên 2 có điểm chẩn đoán holdout cũ cao hơn (0.598096 so với 0.587912). Các kết quả không xác định mô hình nào sẽ tốt hơn trên Private. Chưa chọn final submission.

Ảnh cũng hiển thị test_1.csv = 0.51211, test_3.csv = 0.39409 và test_part1.csv lỗi. Chưa kiểm tra nội dung/mã nguồn các file này; không suy đoán nguyên nhân lỗi hoặc phương pháp tạo file từ tên và điểm.

Ảnh tham chiếu: codex-clipboard-ecb3bd31-6874-4eb7-ba3c-0e07c2632828.png. Ảnh riêng của ứng viên 2: codex-clipboard-7029d0b5-fc51-48f1-8a7a-4efd598aad1c.png. Không đổi code, model, lựa chọn development hoặc CSV dựa trên ảnh này.

## V3 semantic — 22/10/2026, chưa nộp Kaggle

| Mô hình | CV seed 262 | CV seed 1262 | CV trung bình | Holdout cũ (chẩn đoán) |
|---|---:|---:|---:|---:|
| V2 title-only, top65 | 0.609086 | 0.609052 | 0.609069 | 0.587912 |
| V3 #1: blend_e5_ordinal_meta_c4_title | 0.624535 | 0.617704 | 0.621120 | 0.613725 |
| V3 #2: blend_e5_ordinal_c4_title | 0.624322 | 0.616851 | 0.620586 | 0.611346 |

Huấn luyện/search mất **47.5 giây sau khi đã có cache embeddings** trên Python 3.12.10 Windows x64; không tính tải model và lần tạo embeddings đầu. Chưa đo peak RAM. V3 chưa được nộp Kaggle nên chưa có Public/Private score.

- `v3_candidate_1.csv`: 596 ID, 388 bài đề xuất, fraction=0.65, policy=gain, SHA-256 `439e832e5b851e534bc063f7dd981f58d9bea9a94d825cc6b0b6e73bfeea5780`.
- `v3_candidate_2.csv`: 596 ID, 388 bài đề xuất, fraction=0.65, policy=gain, SHA-256 `27ffb1fdd40eb2a20f1aac0f9f6a11e98852e27c9f994b68abc2db4d49ac95d4`.

Ứng viên 1 thắng v2 ở 5/5 fold seed 262 và 4/5 fold seed 1262. Hai seed vẫn dùng chung development, holdout cũ đã được xem. Không diễn giải đây là bằng chứng độc lập hoặc dự báo chính xác điểm Private.

Thử 27 candidate definitions × 11 tỷ lệ × 2 policy = 594 cấu hình trên seed262. Xác nhận 3 cấu hình đóng băng với seed1262. Gate: cả hai seed phải hơn v2 title top65 và delta trung bình >= .003; giữ tối đa 2, tương quan OOF gain < .995. Selection đã lưu trước khi đánh giá holdout cũ.

Kết quả ablation: pure E5 ordinal + metadata tốt nhất ở seed262 trong nhánh ordinal này đạt 0.616118, blend với TF-IDF đạt 0.624535. Các thử nghiệm MiniLM, RBF SVC, Ridge và blend khác cũng được lưu đầy đủ trong phase1.json/search_results.csv; không chỉ báo mô hình thắng. Kết quả tìm kiếm không được coi là kiểm thử độc lập.

Đã chạy lại ONNX encoding test với cache trống, inference hai bundle và refit độc lập các model cuối; hai CSV khớp hash với bản gốc. Xem outputs/v3_verification/verification.json. Code chỉ tạo file local; người dùng tự nộp.

Đọc lại Evaluation ngày 22/10: harmonic mean Q/R chi tiết vẫn khớp metric.py; abstract 60/40 vẫn mâu thuẫn. Rules cho phép pretrained nếu khai báo/trích dẫn. Submissions hiện hiển thị 0/3 final selections (trước đó 0/2); chưa chọn final và không đồng nhất số slot final với quota nộp/ngày.
