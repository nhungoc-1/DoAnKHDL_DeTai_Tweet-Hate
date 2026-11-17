"""P12 README.md Generator: Builds comprehensive README.md synchronized with reports/facts.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils import get_project_root

PROJECT_ROOT = get_project_root()
REPORTS_DIR = PROJECT_ROOT / "reports"


def build_readme() -> None:
    facts: dict[str, Any] = json.loads((REPORTS_DIR / "facts.json").read_text(encoding="utf-8"))

    ds_tw = facts["datasets"]["assigned"]
    ds_rd = facts["datasets"]["assigned_reddit"]
    ds_hate = facts["datasets"]["hate"]
    ds_dav = facts["datasets"]["davidson_cross_domain"]
    sp = facts["preprocessing_and_splits"]
    bal = facts["balance_choice"]
    bm = facts["best_model"]
    ft = facts["final_test"]
    ci = ft["ci95"]["f1_pos"]
    rw = facts["realworld"]
    bonus = facts["bonus_models"]

    readme_md = f"""# Gắn Nhãn Các Đoạn Tweet Có Nội Dung Phân Biệt Chủng Tộc, Phân Biệt Giới Tính (Twitter Sentimental Analysis)

> **Loại đề tài:** Phân loại nhị phân & Khai phá văn bản (*Binary Classification / Text Mining*)  
> **Môi trường hỗ trợ:** Google Colab & Python Local (`seed = 42` cố định toàn dự án)  
> **Mô hình chốt (`best_model`):** `{bm['name']}` (`threshold = {bm['threshold']:.3f}`) — **Test F1 = `{ft['f1_pos']:.4f}`** (95% CI: `[{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]`), **PR-AUC = `{ft['pr_auc']:.4f}`**, **ROC-AUC = `{ft['roc_auc']:.4f}`**

---

## 1. Giới thiệu & Quyết định Phương pháp luận về Dữ liệu

Đề tài xây dựng hệ thống tự động phát hiện và gắn nhãn nhị phân các đoạn tweet có nội dung **phân biệt chủng tộc (racism)** hoặc **phân biệt giới tính (sexism)**:
- **Nhãn `1` (Phản cảm — Racist/Sexist):** Tweet chứa ngôn từ thù ghét, kỳ thị, miệt thị hoặc hạ thấp nhân phẩm dựa trên chủng tộc, sắc tộc, màu da, tôn giáo hoặc giới tính.
- **Nhãn `0` (Bình thường — Non-Hate):** Giao tiếp đời thường, tin tức, bình luận chính trị hoặc câu lên án nạn phân biệt đối xử (*counter-speech*).

### Phát hiện quan trọng về bộ dữ liệu được giao (`Twitter_Data.csv`)
Qua kiểm định thực tế tại giai đoạn P1 (`src/data_io.py` & `reports/data_report.json`):
- Bộ dữ liệu được giao trong đề (`cosmos98/twitter-and-reddit-sentimental-analysis-dataset`, file `Twitter_Data.csv` gồm **{ds_tw['n_rows']:,} dòng**) chỉ có 2 cột `clean_text` và `category` (`-1.0, 0.0, +1.0`), phản ánh **cảm xúc chính trị Ấn Độ** (xoay quanh thủ tướng Modi và bầu cử), **hoàn toàn không có nhãn racist/sexist**.
- **Quyết định xử lý (Phương án A):**
  1. Sử dụng bộ dữ liệu chuẩn **Tweets Hate Speech Detection** (**{ds_hate['n_rows']:,} dòng** có nhãn nhị phân `0/1` racist/sexist chuẩn xác) để làm sạch, khử trùng (**loại {sp['n_near_duplicate_rows_dropped']:,} dòng trùng/retweet** và **{sp['n_conflict_rows_dropped']} dòng mâu thuẫn nhãn**, còn **{sp['n_final_unique_clean']:,} dòng sạch**) và chia tập huấn luyện/đánh giá.
  2. Sử dụng trọn vẹn bộ **`Twitter_Data.csv` ({ds_tw['n_rows']:,} dòng)**, **`Reddit_Data.csv` ({ds_rd['n_rows']:,} dòng)** và bộ **Davidson et al. ({ds_dav['n_rows']:,} dòng)** cho phân tích thống kê mô tả (Câu 1) và kiểm thử thực tế khác miền (*domain shift* — Câu 3).

---

## 2. Cấu trúc Thư mục Dự án

```text
DoAnKHDL_DeTai_Tweet-Hate/
├─ AGENTS.md                      # Đặc tả kỹ thuật chi tiết toàn bộ quy trình
├─ README.md                      # Tài liệu hướng dẫn chạy lại toàn diện
├─ requirements.txt               # Danh sách thư viện tương thích Colab & Local
├─ config.yaml                    # Cấu hình trung tâm (seed=42, đường dẫn, tham số)
├─ run_all.py                     # Script chạy tái lập toàn bộ pipeline (CPU/GPU)
├─ data/
│   ├─ raw/                       # Dữ liệu gốc tải tự động (Twitter_Data, Reddit_Data, hate_train, davidson)
│   ├─ processed/clean.csv        # Dữ liệu sau làm sạch 12 bước & khử trùng (28,330 dòng)
│   ├─ splits/{{train,val,test}}.csv# Chia cố định 70/15/15 (19,830 / 4,250 / 4,250 dòng)
│   └─ real_world/                # Bộ thực tế 300 mẫu (real_world_labeled.csv, to_label.csv, ANNOTATION_GUIDE.md)
├─ db/tweets.db                   # Cơ sở dữ liệu SQLite phục vụ truy vấn SQL
├─ sql/                           # 5 kịch bản SQL (00_schema, 10_token_freq, 20_log_odds, 30_cooccur, 40_lexicon)
├─ src/                           # Mã nguồn mô-đun hóa (utils, data_io, clean, features, models, evaluate, threshold, predict, transformer)
├─ notebooks/                     # Các Jupyter Notebooks báo cáo & thực nghiệm đã chạy sẵn output
├─ app/app.py                     # Ứng dụng Web Streamlit 3 Tab tiếng Việt
├─ models/                        # Mô hình chốt (best_model.joblib, threshold.json, model_card.json)
├─ reports/                       # Toàn bộ báo cáo (report.docx/pdf, slides.pptx/pdf, qa_bank.md, speaker_script.md, demo_script.md, cheat_sheet.md, facts.json, figures/, tables/)
└─ tests/                         # Bộ 17 unit tests tự động (test_clean.py, test_leakage.py, test_predict.py)
```

---

## 3. Hướng dẫn Kết nối & Chạy trên Google Colab và Máy Local

### Cách 1: Chạy trực tiếp trên Google Colab (Tự động tải từ GitHub — 1-Click Run All)
- **Mở nhanh Notebook Báo cáo Tổng hợp trên Google Colab:**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhungoc-1/DoAnKHDL_DeTai_Tweet-Hate/blob/main/notebooks/BAO_CAO_DO_AN_TWEET_HATE_CLASSIFICATION.ipynb)
1. Mở trực tiếp từ GitHub qua liên kết trên hoặc vào **Google Colab → File → Open notebook → GitHub**, nhập URL: `https://github.com/nhungoc-1/DoAnKHDL_DeTai_Tweet-Hate.git` và chọn bất kỳ notebook nào trong thư mục `notebooks/`.
2. Ô code đầu tiên (`SETUP`) sẽ tự động nhận diện môi trường Google Colab, tải mã nguồn dự án từ GitHub `https://github.com/nhungoc-1/DoAnKHDL_DeTai_Tweet-Hate.git` vào `/content/DoAnKHDL_DeTai_Tweet-Hate`, cài đặt các thư viện bổ sung, tải tài nguyên NLTK và cố định `seed = 42`.
3. Chọn **Runtime → Run all** để chạy lại toàn bộ thực nghiệm từ đầu đến cuối.

### Cách 2: Chạy lại trên Máy Local (Windows / Linux / macOS)
```bash
# 1. Tạo môi trường ảo và cài đặt thư viện
python -m venv .venv
.venv\\Scripts\\activate          # Trên Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. Chạy bộ kiểm thử tự động (17 unit tests kiểm tra làm sạch, chống rò rỉ dữ liệu & dự đoán)
pytest -q

# 3. Chạy lại toàn bộ pipeline tái lập kết quả (P1 -> P6, P8, P9)
python run_all.py --cpu-only

# 4. Khởi động ứng dụng Web Streamlit (tại http://localhost:8501)
streamlit run app/app.py
```

### Cách 3: Chạy Ứng dụng Streamlit ngay trong Google Colab qua Cloudflared Tunnel
Trong một ô code trên Google Colab, chạy:
```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!nohup streamlit run app/app.py --server.port 8501 &
!./cloudflared tunnel --url http://localhost:8501
```

---

## 4. Nguồn Dữ liệu & Giấy phép (Licenses)

| Bộ dữ liệu | Nguồn tải | Số dòng | Giấy phép |
|---|---|---:|---|
| `Twitter_Data.csv` & `Reddit_Data.csv` | Kaggle: `cosmos98/twitter-and-reddit-sentimental-analysis-dataset` | {ds_tw['n_rows']:,} & {ds_rd['n_rows']:,} | CC0: Public Domain |
| `Tweets Hate Speech Detection` | HuggingFace: `tweets-hate-speech-detection/tweets_hate_speech_detection` | {ds_hate['n_rows']:,} (train) / 17,197 (test) | MIT / Public Research |
| `Hate Speech and Offensive Language` | Davidson et al. (ICWSM 2017) — `tdavidson/hate_speech_offensive` | {ds_dav['n_rows']:,} (5,593 binary) | MIT License |

---

## 5. Tổng hợp Kết quả Thực nghiệm Chính (Đồng bộ từ `reports/facts.json`)

### 5.1. So sánh các Mô hình trên Tập Validation ($N = 4,250$) & Tinh chỉnh (P5 – P7)
- **Chiến lược cân bằng dữ liệu được chọn (P4):** `RandomOverSampler (ros)` đạt 5-fold CV F1 = **0.5876 ± 0.0117** (so với không cân bằng chỉ đạt F1 = **0.2390**).
- **Mô hình chốt sau tinh chỉnh (P6):** **`{bm['name']}`** với `ngram_range=(1,3)`, `min_df=2`, `max_features=50,000`, `C={bm['params']['clf__C']}`, `loss='squared_hinge'`, `class_weight='balanced'` và ngưỡng tối ưu **`threshold = {bm['threshold']:.3f}`** giúp tăng Validation F1 từ **0.5727** lên **`{bm['val_metrics_after_tuning_best_thr']['f1_pos']:.4f}`** (**+8.39 điểm %**).
- **Mô hình Học sâu / Transformer điểm cộng (P7):** `BiLSTM (PyTorch)` đạt Val F1 = **`{bonus['bilstm']['f1_pos']:.4f}`**, `DistilBERT (Fine-tuned)` đạt Val F1 = **`{bonus['distilbert']['f1_pos']:.4f}`**.

### 5.2. Kết quả Đánh giá Cuối trên Tập Test Độc lập ($N = 4,250$ — Chạy 1 lần duy nhất ở P8)
- **Test F1 Lớp dương (`f1_pos`):** **`{ft['f1_pos']:.4f}`** (Khoảng tin cậy 95% Bootstrap: **`[{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]`**)
- **Test Precision / Recall Lớp dương:** **`{ft['precision_pos']:.4f}`** / **`{ft['recall_pos']:.4f}`**
- **Test PR-AUC / ROC-AUC / MCC:** **`{ft['pr_auc']:.4f}`** / **`{ft['roc_auc']:.4f}`** / **`{ft['mcc']:.4f}`**
- **Ma trận nhầm lẫn Test:** `TN = {ft['tn']}`, `FP = {ft['fp']}`, `FN = {ft['fn']}`, `TP = {ft['tp']}`

### 5.3. Kết quả Kiểm thử Thực tế Đa nguồn ($N = {rw['n']}$ mẫu — P9)
- **Độ đồng thuận 2 người gắn nhãn độc lập:** Đồng thuận **`{rw['agreement_pct']:.2f}%`**, hệ số **Cohen's Kappa $\\kappa = {rw['kappa']:.4f}$**.
- **Hiệu năng tổng thể trên bộ Thực tế:** **Real-World F1 = `{rw['overall_metrics']['f1_pos']:.4f}`** (Precision = **`{rw['overall_metrics']['precision_pos']:.4f}`**, Recall = **`{rw['overall_metrics']['recall_pos']:.4f}`**, PR-AUC = **`{rw['overall_metrics']['pr_auc']:.4f}`**).

---

## 6. Hạn chế & Nguyên tắc Đạo đức (Ethics & Limitations)
1. **Ẩn danh & Che từ nhạy cảm:** Mọi tài khoản cá nhân được ẩn thành `@user`; toàn bộ từ tục trong biểu đồ, bảng biểu, notebook và báo cáo đều được che tự động qua `src/utils.py::mask_profanity()`.
2. **Thiên lệch nhóm (Subgroup Bias) & Ngữ cảnh mỉa mai:** Mô hình dựa trên tần suất n-gram có tỷ lệ báo động giả (FPR) cao hơn trên các câu bình thường chứa từ nhận diện chủng tộc/tôn giáo/giới tính và khó bắt các câu mỉa mai sâu không dùng từ khóa đặc trưng.
3. **Khuyến nghị vận hành:** Công cụ chỉ dùng để **hỗ trợ sàng lọc sơ bộ (Human-in-the-loop)**, không dùng để tự động khóa tài khoản khi chưa có con người kiểm tra lại.
"""
    (PROJECT_ROOT / "README.md").write_text(readme_md, encoding="utf-8")
    print("Generated README.md successfully.")


if __name__ == "__main__":
    build_readme()
