# Thư viện và tài liệu

Thư viện được khai báo trong requirements.txt và manifest của mỗi lần chạy. Nhóm cần trích dẫn các phương pháp/thư viện đã sử dụng trong báo cáo của mình.

- scikit-learn 1.7.2: https://scikit-learn.org/1.7/ — TF-IDF, ColumnTransformer, LogisticRegression, Ridge và StratifiedGroupKFold.
- NumPy 2.3.5: https://numpy.org/doc/2.3/ — tính toán mảng và metric.
- SciPy 1.16.3: https://docs.scipy.org/doc/scipy-1.16.3/ — ma trận sparse/phép toán phục vụ scikit-learn.
- joblib 1.5.2: https://joblib.readthedocs.io/ — lưu/tải model.
- threadpoolctl 3.6.0: https://github.com/joblib/threadpoolctl — giới hạn luồng tính toán.
- striprtf 0.0.29: https://pypi.org/project/striprtf/0.0.29/ — chuyển Test.csv từ RTF về text trước khi đọc CSV.

Dữ liệu: TU VU NGOC, Assignment Data Mining - 261, Kaggle, truy cập ngày 19/10/2026. https://www.kaggle.com/competitions/assignment-data-mining-261/data . Trang Data ghi CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/ . Ba tệp dữ liệu gốc và bản prepared không được đóng gói trong bản ZIP; kết quả chẩn đoán có thể chứa ID/nhãn train và không được commit công khai mặc định.

Không có dữ liệu bên ngoài, mô hình pretrained hoặc mã lời giải của nhóm khác trong v1. Không sử dụng API dịch vụ trả phí. Mã triển khai được tạo với sự hỗ trợ của Codex trong chat của nhóm; nhóm cần tự đọc, hiểu, kiểm tra và khai báo việc hỗ trợ này nếu môn học yêu cầu.

## Phương pháp v2

- Wang, Sida I. và Christopher D. Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. ACL short papers, 90–94. https://aclanthology.org/P12-2018/ . V2 dùng ý tưởng supervised NB log-count ratio cho bốn bài toán nhị phân ordinal, chuẩn hóa L2 và Logistic Regression. Đây là biến thể triển khai mới, không gọi là tái lập chính xác NB-SVM của tác giả.
- scikit-learn 1.7: TruncatedSVD, SVC, TF-IDF/CountVectorizer, LogisticRegression, Ridge và preprocessing. Không thêm thư viện so với v1.
- Cả v1 và v2 chỉ huấn luyện từ dữ liệu cuộc thi. Không tải pretrained model, không dùng API trả phí hoặc dữ liệu của nhóm khác.

## Nguồn bổ sung v3

- intfloat/e5-small-v2: https://huggingface.co/intfloat/e5-small-v2 — MIT, revision d908e3456e5b47835e008ab1bd688bdd2fe5abe1, bản ONNX INT8 từ publisher. Prefix query, masked mean pooling, L2 normalization theo model card. Không fine-tune encoder.
- sentence-transformers/all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 — Apache-2.0, revision 5641a7880f40ebf4035d05e60c5f9b7a9c272c84, ONNX FP32. Thử trong search; không nằm trong hai bundle được chọn.
- ONNX Runtime 1.23.2: https://pypi.org/project/onnxruntime/1.23.2/ ; Tokenizers 0.22.1: https://pypi.org/project/tokenizers/0.22.1/ . Chạy trực tiếp file ONNX/tokenizer local, không dùng Torch hoặc mã Python từ repo model.
- Hướng dẫn ONNX/pooling: https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html . Toàn bộ dependency phụ có pin trong requirements-semantic.txt.

Asset model card, URL và hash tải về lưu ở assets/encoders/*/asset_manifest.json; model weights/cache không commit mặc định. Upstream corpus của pretrained model có thể chứa văn bản nghiên cứu công khai; không biết đầy đủ để xác định overlap. V3 sử dụng tài nguyên ngoài này nên không được mô tả là chỉ dùng dữ liệu train theo nghĩa không có pretrained resources.

## Nguồn bổ sung V4–V6

- [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2): MIT, revision `6d89b151ea59fac713a86ac7d7e5b684c58543bf`; ONNX INT8, mean pooling và L2. Dùng trong bản V6 candidate 2.
- [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5): MIT, revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`; ONNX FP32, CLS pooling và L2. Dùng trong bản V6 candidate 2.
- [intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2), revision `226d08282be20fa85017f8ec740d3298b706fc8b`, và [Xenova/bge-large-en-v1.5](https://huggingface.co/Xenova/bge-large-en-v1.5), revision `dfeef6070b90658e1b391a6940efdb0925c1de6f`: hai encoder ONNX đã thử trong nhánh V6 large. BGE-large là bản chuyển đổi từ [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5). Nhánh này không qua tiêu chí xác nhận và không nằm trong bản tham chiếu Public 0.66504.

Ba encoder của bản tham chiếu là E5-small, E5-base và BGE-base; đều được giữ cố định. Nhánh tác giả dùng tên đã có trong dữ liệu cuộc thi, không thu thập thêm hồ sơ tác giả. Các head có giám sát được fit bằng nhãn train. `reproduce.py` tái tạo cấu hình V6 candidate 2 đã chọn, không chọn lại mô hình bằng Public score.

URL, revision, model card và hash được lưu trong `assets/encoders/*/asset_manifest.json`. Các assets được tải từ nguồn upstream và không được commit mặc định. License upstream của encoder cần được phân biệt với quyền sử dụng dữ liệu cuộc thi và license mã nguồn của nhóm.
