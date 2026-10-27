# V3 — Semantic title embeddings và ordinal classification

> **Snapshot lịch sử V3:** tài liệu dưới đây giữ lại trạng thái lúc thử nghiệm V3. Điểm Public của V3 candidate 1 sau đó được nhóm ghi nhận là **0.63502**. Bản tham chiếu hiện tại là V6 candidate 2 (**0.66504**); lệnh cài đặt và tái tạo hiện hành nằm trong [README](../README.md).

## Kết quả và mục tiêu

V1/v2 là các mô hình dựa trên title TF-IDF và metadata. V3 thử biểu diễn ngữ nghĩa để kết nối các tiêu đề có ý gần nhau nhưng khác từ. Mô hình pretrained chỉ tạo đặc trưng; **Logistic Regression cho bốn ngưỡng nhãn** là phần học có giám sát từ dữ liệu cuộc thi và đáp ứng yêu cầu classification của học phần.

| Mô hình | CV seed 262 | CV seed 1262 | CV trung bình | Holdout cũ (chẩn đoán) |
|---|---:|---:|---:|---:|
| V2 title-only, top65 | 0.609086 | 0.609052 | 0.609069 | 0.587912 |
| V3 #1: blend_e5_ordinal_meta_c4_title | 0.624535 | 0.617704 | 0.621120 | 0.613725 |
| V3 #2: blend_e5_ordinal_c4_title | 0.624322 | 0.616851 | 0.620586 | 0.611346 |

Huấn luyện/search mất **47.5 giây sau khi đã có cache embeddings** trên Python 3.12.10 Windows x64; không tính tải model và lần tạo embeddings đầu. Chưa đo peak RAM. V3 chưa được nộp Kaggle nên chưa có Public/Private score.

- `v3_candidate_1.csv`: 596 ID, 388 bài đề xuất, fraction=0.65, policy=gain, SHA-256 `439e832e5b851e534bc063f7dd981f58d9bea9a94d825cc6b0b6e73bfeea5780`.
- `v3_candidate_2.csv`: 596 ID, 388 bài đề xuất, fraction=0.65, policy=gain, SHA-256 `27ffb1fdd40eb2a20f1aac0f9f6a11e98852e27c9f994b68abc2db4d49ac95d4`.

Ứng viên 1 thắng v2 ở 5/5 fold seed 262 và 4/5 fold seed 1262. Hai seed vẫn dùng chung development, holdout cũ đã được xem. Không diễn giải đây là bằng chứng độc lập hoặc dự báo chính xác điểm Private.


## Kiểm tra lại quy trình ngày 22/10/2026

- Đọc lại Evaluation và Rules trực tiếp trên trình duyệt Kaggle. Công thức ở Evaluation chi tiết vẫn là harmonic mean của relevance và worthwhile-paper recall; title similarity được ghi rõ không nằm trong metric chi tiết. `metric.py` phù hợp với mô tả này, đã có test tính tay và quy tắc đồng điểm theo ID dạng văn bản. Chưa được cung cấp mã chấm ẩn.
- Abstract Overview vẫn mô tả 60/40 relevance/title coverage, mâu thuẫn với Evaluation chi tiết. Không tự đổi metric để đuổi theo điểm Public.
- Không thấy lỗi fit từ vựng lên test hoặc nhóm trùng vượt biên validation trong code đã kiểm tra. Các phép centering và venue encoding của v3 cũng chỉ fit trên phần training của fold.
- Vấn đề của các phiên bản trước gồm biểu diễn chưa hiểu ngữ nghĩa, việc tìm kiếm nhiều cấu hình có thể lạc quan, và holdout đã được xem nên không còn là kiểm thử độc lập mới. Quy trình có kiểm soát không đảm bảo mô hình đạt thứ hạng cao.
- Rules hiện cho phép pretrained/public external resources nếu trích dẫn và giải thích. V3 khai báo encoder, revision, license, lượng tử hóa và cách dùng bên dưới.

## Encoder công khai cố định

| Encoder | Revision | File | Xử lý |
|---|---|---|---|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | `5641a7880f40ebf4035d05e60c5f9b7a9c272c84` | `onnx/model.onnx` | FP32, không prefix, mean pooling có attention mask, L2 normalize |
| [e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) | `d908e3456e5b47835e008ab1bd688bdd2fe5abe1` | `onnx/model_qint8_avx512_vnni.onnx` | Bản INT8 của publisher, prefix `query: `, masked mean pooling, L2 normalize |

MiniLM license Apache-2.0; E5 license MIT. Model card gốc được lưu cùng asset. Chạy bằng ONNX Runtime CPU, không tải hoặc chạy Python từ repository model. Không fine-tune encoder; không gửi title, nhãn hoặc metadata của cuộc thi cho API bên ngoài. Prefix E5 theo hướng dẫn của model card cho classification features. Giới hạn 128 token, batch size 16; các title dài hơn bị truncate. Lượng tử hóa và cách batch có thể ảnh hưởng kết quả nên được cố định.

Tổng asset đã tải của hai encoder khoảng **125,8 MB**; không tính thư viện trong `.venv`. Model và cache ở ổ D, không dùng cache Hugging Face mặc định trong user profile. SHA-256 model ONNX được kiểm tra với Git LFS pointer tại revision cố định; các file còn lại được ghi hash trong `asset_manifest.json` và bundle đã fit.

Nguồn encoder pretrained có thể chứa văn bản nghiên cứu công khai; không biết toàn bộ corpus upstream để loại trừ overlap. Đây là giới hạn phải khai báo trong báo cáo. Không sử dụng hidden label hoặc suy ra Public/Private membership.

## Thực nghiệm

1. Giữ nguyên development 2.018 bài và holdout 476 bài của v1, kiểm tra hash input và mọi group/fold ID. Phạm vi metric chính là các venue có trong test, như các phiên bản trước.
2. Frozen embeddings được tính riêng cho train và test. Không học tokenizer, vocabulary, mean embedding hoặc classifier từ test. Cache có khóa theo ID/title, revision, hash assets, phiên bản ONNX/tokenizer, độ dài và batch size.
3. Trong mỗi fold, lấy mean embedding trên phần training, trừ mean rồi L2 normalize. Head có metadata thêm venue one-hot trọng số .35 và metadata số trọng số .05, đều fit/biến đổi theo training.
4. Mỗi encoder thử ordinal C=4/16, ordinal + metadata C=4, multiclass C=4, RBF SVC C=4 và Ridge alpha=1. Có hai baseline ordinal TF-IDF v2 được fit lại trên cùng fold.
5. So sánh 27 candidate definitions, gồm model đơn, blend 50/50 từng semantic model với v2 title-only và blend 50/50 hai encoder ordinal. Mỗi candidate thử 11 tỷ lệ từ .50 tới 1.00 và hai policy `gain`, `utility_half`: **594 cấu hình**. Tập đã chọn luôn được xếp theo expected gain.
6. Chọn cấu hình tốt nhất của từng candidate trên 5 folds seed 262. Giữ ba candidate semantic đứng đầu, đóng băng policy/fraction, kiểm tra seed 1262. Chỉ xuất nếu cả hai seed tốt hơn v2 title-only top65 và cải thiện trung bình ít nhất .003. Tối đa hai CSV; candidate thứ hai phải có OOF gain correlation < .995 với candidate đầu.
7. Lưu `selection.json` trước khi đọc kết quả holdout cũ, rồi mới đánh giá chẩn đoán và refit toàn bộ 2.494 bài train. Không đổi lựa chọn dựa trên holdout hay điểm Public.

Hai seed vẫn dùng chung development nên **không phải hai test độc lập**. Nhiều vòng tìm kiếm có thể tạo selection bias. Điểm holdout cũ chỉ phục vụ chẩn đoán. Hai ứng viên cuối tương quan khá cao; không coi chúng là hai phương pháp độc lập.

## Chạy trên máy hiện tại

Môi trường và asset trong `D:\DM` đã sẵn sàng. Để chạy lại toàn bộ thí nghiệm, dùng thư mục output mới:

```powershell
cd D:\DM
.\.venv\Scripts\python.exe -X utf8 semantic.py run --data-dir .\assignment-data-mining-261 --reference .\outputs\my_run01 --assets .\assets\encoders --cache .\.cache\embeddings --out .\outputs\v3_reproduce01
```

## Tái lập trên máy giảng viên, không Docker

1. Dùng Python Windows x64 3.12, tạo `.venv` và đặt ba tệp dữ liệu cuộc thi đúng vị trí theo README. Môi trường kiểm chứng là Python 3.12.10; chưa kiểm tra một máy sạch hoặc OS khác.
2. Cài đủ dependency cố định, tải encoder cố định và tạo bản chia tập v1:

```powershell
cd D:\DM
python -m venv .venv
$env:TEMP = Join-Path (Get-Location) '.cache\tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements-semantic.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 semantic.py prepare --data-dir .\assignment-data-mining-261 --assets .\assets\encoders --cache .\.cache\embeddings
.\.venv\Scripts\python.exe main.py run --data-dir .\assignment-data-mining-261 --out .\outputs\my_run01 --seed 261 --folds 5
.\.venv\Scripts\python.exe -X utf8 semantic.py run --data-dir .\assignment-data-mining-261 --reference .\outputs\my_run01 --out .\outputs\v3_reproduce01
```

Kiểm tra `python` là bản Windows x64, không phải MSYS2/MinGW. Nếu cần, thay dòng tạo venv bằng `pymanager exec -V:3.12 -m venv .venv` theo README. Không tạo lại `.venv` đang hoạt động. Không cần chạy lại search v2 để chạy v3. Cài thư viện và `prepare` cần Internet để GET file công khai; sau đó huấn luyện và inference dùng asset local, chạy offline.

3. Đọc `reports/summary.md`, đối chiếu `search_plan.json`, `selection.json`, `results.json` và manifest. Hai CSV nằm trong `submissions/`.

## Chỉ tái tạo CSV từ bundle đã lưu

```powershell
.\.venv\Scripts\python.exe -X utf8 semantic.py predict --data-dir .\assignment-data-mining-261 --bundle .\outputs\v3_run01\models\v3_candidate_1.joblib --assets .\assets\encoders --cache .\.cache\embeddings --output .\outputs\v3_reproduced\v3_candidate_1.csv
.\.venv\Scripts\python.exe main.py check --data-dir .\assignment-data-mining-261 --submission .\outputs\v3_reproduced\v3_candidate_1.csv
Get-FileHash .\outputs\v3_run01\submissions\v3_candidate_1.csv, .\outputs\v3_reproduced\v3_candidate_1.csv -Algorithm SHA256
```

Đổi số 1 thành 2 cho ứng viên thứ hai. Giữ chính xác requirements, encoder assets và dữ liệu. Chỉ nạp bundle do nhóm tạo và tin cậy. Asset/model/cache không commit mặc định; dùng `prepare` để giảng viên tải đúng phiên bản. Không có Kaggle API key trong quy trình.

## Kiểm chứng đã thực hiện

- 20/20 test đạt: bao gồm metric tính tay, dữ liệu, group, từ vựng, policy, pooling loại padding, centering/venue fit training-only và model round-trip.
- Đã mã hóa lại toàn bộ title test bằng cache trống, dự đoán lại từ hai bundle và refit độc lập các head đã chọn. Cả hai đường tạo CSV có hash giống bản gốc. Không chỉ kiểm tra việc đọc lại cache cũ.
- Dữ liệu gốc, bốn CSV v1/v2 và các code/output ghi trong manifest v3 khớp hash.
- `pip check` không phát hiện dependency hỏng. Chưa đo peak RAM, chưa chứng minh bit-for-bit trên CPU/OS khác.
- Chi tiết máy đọc được: `outputs/v3_verification/verification.json`. Việc kiểm chứng replay các fit cuối và encoder test, không chạy lại toàn bộ search CV.

## Nguồn

- [E5 model card và hướng dẫn prefix/pooling](https://huggingface.co/intfloat/e5-small-v2).
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- [Sentence Transformers: ONNX cần pooling/normalization khi dùng trực tiếp](https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html).
- [ONNX Runtime 1.23.2](https://pypi.org/project/onnxruntime/1.23.2/) và [Tokenizers 0.22.1](https://pypi.org/project/tokenizers/0.22.1/).
- [Competition Evaluation](https://www.kaggle.com/competitions/assignment-data-mining-261/overview) và [Rules](https://www.kaggle.com/competitions/assignment-data-mining-261/rules), đọc lại ngày 22/10/2026.
