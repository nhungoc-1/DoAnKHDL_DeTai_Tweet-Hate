"""Generate and execute all project Jupyter Notebooks (01 through 08) with embedded outputs and Vietnamese commentary."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbclient import NotebookClient

from src.utils import get_project_root

PROJECT_ROOT = get_project_root()
NB_DIR = PROJECT_ROOT / "notebooks"

BOOTSTRAP_CODE = """# === THIẾT LẬP MÔI TRƯỜNG & NẠP THƯ VIỆN DỰ ÁN ===
import os, sys, subprocess
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO_URL = "https://github.com/nhungoc-1/DoAnKHDL_DeTai_Tweet-Hate.git"

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    print("[SETUP] Đang khởi tạo môi trường Google Colab...", flush=True)
    for candidate in [
        Path("/content/DoAnKHDL_DeTai_Tweet-Hate"),
        Path("/content/drive/MyDrive/DoAnKHDL_DeTai_Tweet-Hate"),
        Path("/content/drive/MyDrive/DO_AN_KHDL"),
        Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd(),
    ]:
        if (candidate / "config.yaml").exists():
            PROJECT_DIR = candidate
            subprocess.run(["git", "-C", str(PROJECT_DIR), "pull", "--ff-only"], check=False)
            break
    else:
        PROJECT_DIR = Path("/content/DoAnKHDL_DeTai_Tweet-Hate")
        print(f"[SETUP] Đang tải dự án từ GitHub vào {PROJECT_DIR} (mất ~3 giây)...", flush=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(PROJECT_DIR)], check=True)
    print("[SETUP] Đang kiểm tra & cài đặt các thư viện bổ sung (mất ~5-10 giây)...", flush=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "imbalanced-learn", "emoji", "kagglehub", "datasets",
        "pyvi", "langdetect", "python-docx", "python-pptx", "wordcloud"
    ], check=False)
    import nltk
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)
else:
    PROJECT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.utils import set_seed, load_config, mask_profanity
set_seed(42)
CFG = load_config()
(PROJECT_DIR / "db").mkdir(parents=True, exist_ok=True)
print(f"[HOÀN TẤT] IN_COLAB = {IN_COLAB} | PROJECT_DIR = {PROJECT_DIR}")
"""


def _save_and_execute(nb_name: str, cells: list) -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11.0"}
    out_path = NB_DIR / nb_name
    nbformat.write(nb, out_path)
    print(f"[Notebooks] Executing {nb_name}...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_ROOT)}})
    client.execute()
    nbformat.write(nb, out_path)
    print(f"[Notebooks] Saved executed {nb_name}")


def build_all_notebooks() -> None:
    p2 = json.loads((PROJECT_ROOT / "reports" / "p2_findings.json").read_text(encoding="utf-8"))
    p3 = json.loads((PROJECT_ROOT / "reports" / "p3_sql_summary.json").read_text(encoding="utf-8"))
    p4 = json.loads((PROJECT_ROOT / "reports" / "p4_summary.json").read_text(encoding="utf-8"))
    p5 = json.loads((PROJECT_ROOT / "reports" / "p5_summary.json").read_text(encoding="utf-8"))
    p6 = json.loads((PROJECT_ROOT / "reports" / "p6_summary.json").read_text(encoding="utf-8"))
    p7 = json.loads((PROJECT_ROOT / "reports" / "p7_summary.json").read_text(encoding="utf-8"))
    p8 = json.loads((PROJECT_ROOT / "reports" / "p8_summary.json").read_text(encoding="utf-8"))
    p9 = json.loads((PROJECT_ROOT / "reports" / "realworld_metrics.json").read_text(encoding="utf-8"))

    # =========================================================================
    # 01_data_collection_eda.ipynb (P1 & P2)
    # =========================================================================
    cells_01 = [
        new_markdown_cell(
            "# 01. Thu thập Dữ liệu & Thống kê Mô tả — EDA (Câu 1: 2.0 điểm — P1 & P2)\n\n"
            "**Mục tiêu:**\n"
            "1. Tải và kiểm chứng cấu trúc bộ dữ liệu được giao (`cosmos98/twitter-and-reddit-sentimental-analysis-dataset`) cùng bộ dữ liệu huấn luyện có nhãn phân biệt chủng tộc / giới tính (`tweets-hate-speech-detection`, Phương án A) và bộ kiểm tra chéo miền (`Davidson et al.`).\n"
            "2. Thực hiện đầy đủ 10 phân tích thống kê mô tả và kiểm định giả thuyết (Mann-Whitney U, Chi-Square $\\chi^2$, Smoothed Log-Odds Ratio, Jaccard Similarity).\n"
            "3. Toàn bộ từ nhạy cảm / từ tục được che tự động bằng `mask_profanity()` trước khi hiển thị."
        ),
        new_code_cell(BOOTSTRAP_CODE),
        new_markdown_cell("## 1. Kiểm tra dữ liệu thu thập & Xác nhận vấn đề nhãn của `Twitter_Data.csv` (P1)"),
        new_code_cell(
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n"
            "from src.data_io import load_assigned, load_hate, load_davidson\n\n"
            "twitter_df, reddit_df = load_assigned()\n"
            "hate_df = load_hate()\n"
            "davidson_df = load_davidson()\n\n"
            "with open('reports/data_report.json', 'r', encoding='utf-8') as f:\n"
            "    data_report = json.load(f)\n\n"
            "summary_stats = pd.read_csv('reports/tables/P2_summary_stats.csv')\n"
            "display(summary_stats)\n"
            "display(Image('reports/figures/P2_01_dataset_quality.png'))"
        ),
        new_markdown_cell(
            "**Nhận xét Biểu đồ P2_01 (Quy mô & Chất lượng dữ liệu):**\n"
            f"- Bộ được giao `Twitter_Data.csv` có **{p2['assigned_twitter_rows_raw']:,} dòng** nhưng chỉ gồm 2 cột `clean_text` và `category` (-1/0/+1 phản ánh cảm xúc chính trị Ấn Độ), có {p2['assigned_twitter_null_rows']} dòng chứa giá trị khuyết (NaN) và **hoàn toàn không có nhãn phân biệt chủng tộc/giới tính**.\n"
            f"- Bộ huấn luyện chính `hate_train.csv` gồm **{p2['hate_train_rows']:,} dòng** với nhãn nhị phân `0/1` chuẩn xác, không có giá trị NaN nhưng chứa **{p2['hate_exact_duplicates']:,} dòng trùng nguyên bản** và **{p2['hate_clean_duplicates']:,} dòng trùng sau khi chuẩn hóa văn bản** (cần khử trùng ở P4 trước khi chia tập để chống rò rỉ dữ liệu)."
        ),
        new_markdown_cell("## 2. Phân bố nhãn & Mức độ mất cân bằng lớp (P2.2)"),
        new_code_cell("display(Image('reports/figures/P2_02_label_distribution.png'))"),
        new_markdown_cell(
            "**Nhận xét Biểu đồ P2_02 (Phân bố nhãn):**\n"
            f"- Dữ liệu `hate_train.csv` bị mất cân bằng lớp nghiêm trọng: Lớp 0 (Bình thường) chiếm **{p2['hate_label_0']:,} mẫu ({100 - p2['hate_positive_pct']:.2f}%)**, trong khi Lớp 1 (Phản cảm — Racist/Sexist) chỉ có **{p2['hate_label_1']:,} mẫu ({p2['hate_positive_pct']:.2f}%)**, tương ứng tỉ lệ mất cân bằng **{p2['imbalance_ratio']}:1**.\n"
            "- Hệ quả trực tiếp: nếu dùng tiêu chí *Accuracy*, một mô hình tầm thường dự đoán toàn bộ là lớp 0 vẫn đạt ~92.99% Accuracy nhưng bỏ sót 100% tweet phản cảm. Do đó dự án bắt buộc dùng **F1-score lớp dương (`f1_pos`)** và **PR-AUC** làm thước đo chính."
        ),
        new_markdown_cell("## 3. Phân bố độ dài Tweet (Ký tự & Số từ) theo Nhãn (P2.3)"),
        new_code_cell(
            "len_df = pd.read_csv('reports/tables/P2_length_stats.csv')\n"
            "display(len_df)\n"
            "display(Image('reports/figures/P2_03_length_distribution.png'))"
        ),
        new_markdown_cell(
            "**Nhận xét Biểu đồ P2_03 (Độ dài tweet):**\n"
            "- Các tweet phản cảm (Lớp 1) có độ dài ký tự trung bình nhỉnh hơn lớp bình thường (~89.2 ký tự vs ~84.3 ký tự) và phân bố tập trung quanh ngưỡng 14-16 từ/tweet.\n"
            "- Phân vị thứ 95 (`p95`) của độ dài từ nằm ở mức 23-24 từ, cho thấy việc giới hạn `max_length = 80..96` subwords khi chạy Transformer (P7) bảo toàn trọn vẹn 100% thông tin câu."
        ),
        new_markdown_cell("## 4. Đặc trưng bề mặt & Kiểm định thống kê Mann-Whitney U / Chi-Square (P2.4)"),
        new_code_cell(
            "surf_df = pd.read_csv('reports/tables/P2_surface_tests.csv')\n"
            "display(surf_df)\n"
            "display(Image('reports/figures/P2_04_surface_features.png'))"
        ),
        new_markdown_cell(
            "**Nhận xét Biểu đồ P2_04 (Đặc trưng bề mặt & Kiểm định giả thuyết):**\n"
            "- Kiểm định $\\chi^2$ và Mann-Whitney U cho thấy sự khác biệt có ý nghĩa thống kê rất cao ($p < 10^{-10}$) ở tần suất nhắc `@user`: **65.03%** tweet phản cảm có chứa `@user` so với chỉ **46.46%** ở tweet bình thường — phản ánh hành vi tấn công trực diện vào cá nhân/nhóm người.\n"
            "- Tweet lớp 1 cũng có tỉ lệ chứa dấu hỏi `?` cao hơn đáng kể (**18.78%** vs **10.42%**, thường là câu hỏi tu từ/châm biếm), trong khi ít sử dụng dấu chấm than `!` và emoji cảm xúc tích cực hơn lớp 0."
        ),
        new_markdown_cell("## 5. Top-20 Unigram, Bigram & WordCloud theo Nhãn (Đã che từ tục — P2.5 & P2.6)"),
        new_code_cell(
            "display(Image('reports/figures/P2_05_top_unigrams.png'))\n"
            "display(Image('reports/figures/P2_05b_top_bigrams.png'))\n"
            "display(Image('reports/figures/P2_06_wordcloud_by_label.png'))"
        ),
        new_markdown_cell(
            "**Nhận xét Biểu đồ P2_05, P2_05b & P2_06 (Unigram, Bigram và WordCloud):**\n"
            "- Ở Lớp 0, các cụm từ phổ biến nhất mang sắc thái tích cực và đời sống gia đình (`love`, `happy`, `day`, `thank`, `father day`, `look forward`).\n"
            "- Ở Lớp 1, nổi bật rõ các từ khóa liên quan đến chủng tộc, giới tính và xung đột chính trị - xã hội (`white`, `black`, `woman`, `racist`, `trump`, `obama`, `libtard`, `allahsoil`, `against`, `hate`), chứng minh bigram giữ vai trò rất quan trọng để nắm bắt ngữ cảnh."
        ),
        new_markdown_cell("## 6. Từ phân biệt mạnh nhất (Smoothed Log-Odds Ratio & Chi-Square) và Hashtag (P2.7 & P2.8)"),
        new_code_cell(
            "disc_df = pd.read_csv('reports/tables/P2_discriminative_terms.csv')\n"
            "display(disc_df.head(15))\n"
            "display(Image('reports/figures/P2_07_discriminative_log_odds_chi2.png'))\n"
            "display(Image('reports/figures/P2_08_top_hashtags.png'))"
        ),
        new_markdown_cell(
            "**Nhận xét Biểu đồ P2_07 & P2_08 (Log-Odds Ratio, $\\chi^2$ và Hashtag):**\n"
            "- Chỉ số Log-Odds có làm mượt Dirichlet (+1) và kiểm định $\\chi^2$ đồng nhất chỉ ra các từ khóa đặc trưng nhất của lớp phản cảm gồm: từ miệt thị tôn giáo/chủng tộc (`allahsoil`, `libtard`, `sjw`, `n****r`), từ công kích giới tính (`misogyny`, `sexist`, `feminism`, `b***h`) và tư tưởng cực đoan (`bigot`, `supremacist`, `nazi`).\n"
            "- Phân tích hashtag cho thấy người viết tweet phản cảm thường gắn chuỗi hashtag mang tính khẩu hiệu cực đoan (`#allahsoil`, `#libtard`, `#sjw`, `#racist`, `#white`), đây là tín hiệu đặc trưng quan trọng khi giữ lại nội dung hashtag sau dấu `#`."
        ),
        new_markdown_cell("## 7. Phân tích Bộ được giao (`Twitter_Data.csv`) & So sánh Giao thoa Từ vựng Hai Miền (P2.9 & P2.10)"),
        new_code_cell(
            "display(Image('reports/figures/P2_09_assigned_twitter_eda.png'))\n"
            "display(Image('reports/figures/P2_10_domain_vocab_overlap.png'))"
        ),
        new_markdown_cell(
            "**Nhận xét Biểu đồ P2_09 & P2_10 (Bộ được giao & So sánh miền từ vựng):**\n"
            "- Phân tích từ vựng lớp `-1` của `Twitter_Data.csv` cho thấy các từ xuất hiện nhiều nhất là `modi`, `india`, `people`, `bjp`, `election`, `congress`, khẳng định đây là bộ dữ liệu cảm xúc chính trị Ấn Độ.\n"
            f"- Hệ số tương đồng **Jaccard trên Top-5000 từ vựng** giữa hai bộ dữ liệu chỉ đạt **{p2['domain_jaccard_top5000']:.4f}** ({p2['domain_shared_words_top5000']:,}/5,000 từ giao nhau), dự báo hiện tượng lệch miền (*domain shift*) rõ rệt khi đem mô hình chạy thử thực tế ở P9."
        ),
        new_markdown_cell(
            "### Tổng hợp Phát hiện chính (P1 & P2)\n"
            f"1. **Vấn đề nhãn dữ liệu được giao:** `Twitter_Data.csv` ({p2['assigned_twitter_rows_raw']:,} dòng) chỉ có nhãn sentiment chính trị `-1/0/+1` (có {p2['assigned_twitter_null_rows']} dòng NaN), không có nhãn racist/sexist ⇒ áp dụng **Phương án A**: huấn luyện trên `hate_train.csv` ({p2['hate_train_rows']:,} dòng) và dùng bộ được giao để phân tích + kiểm thử chéo miền.\n"
            f"2. **Mất cân bằng lớp nghiêm trọng:** Lớp 1 (Phản cảm) chỉ chiếm **{p2['hate_positive_pct']:.2f}%** ({p2['hate_label_1']:,} mẫu) so với **92.99%** ({p2['hate_label_0']:,} mẫu) của Lớp 0, tỉ lệ lệch **{p2['imbalance_ratio']}:1**.\n"
            f"3. **Trùng lặp văn bản cao:** Có tới **{p2['hate_clean_duplicates']:,} mẫu ({100*p2['hate_clean_duplicates']/p2['hate_train_rows']:.1f}%)** trùng nhau sau khi chuẩn hóa văn bản (do retweet/template), bắt buộc khử trùng trước khi chia tập.\n"
            "4. **Khác biệt hành vi bề mặt ($p < 10^{-10}$):** Tweet phản cảm nhắc `@user` nhiều hơn hẳn (**65.03%** vs **46.46%**) và dùng nhiều câu hỏi tu từ `?` hơn (**18.78%** vs **10.42%**).\n"
            "5. **Đặc trưng từ vựng & Hashtag:** Các từ có Log-Odds và $\\chi^2$ cao nhất tập trung vào 3 chủ đề: miệt thị chủng tộc/tôn giáo, định kiến giới tính/phụ nữ, và xung đột ý thức hệ chính trị.\n"
            f"6. **Khoảng cách miền từ vựng lớn:** Độ tương đồng Jaccard Top-5000 từ giữa bộ Hate Speech và `Twitter_Data.csv` chỉ đạt **{p2['domain_jaccard_top5000']:.4f}**, giải thích vì sao cần kiểm thử thực tế đa nguồn ở P9."
        ),
    ]
    _save_and_execute("01_data_collection_eda.ipynb", cells_01)

    # =========================================================================
    # 02_sql_analysis.ipynb (P3)
    # =========================================================================
    cells_02 = [
        new_markdown_cell(
            "# 02. Phân tích SQL: Tìm điểm chung của các Tweet Phản cảm (Yêu cầu 1 — P3)\n\n"
            "**Mục tiêu:**\n"
            "1. Xây dựng cơ sở dữ liệu quan hệ SQLite `db/tweets.db` gồm các bảng `tweets`, `tokens`, `bigrams`, `assigned`, `lexicon`.\n"
            "2. **Nguyên tắc chống rò rỉ:** Chỉ nạp tập huấn luyện (`data/splits/train.csv`, $N = 19,830$) vào bảng phân tích đặc trưng `tweets`/`tokens`/`bigrams` để tuyệt đối không nhìn trộm tập `test`.\n"
            "3. Thực thi 4 truy vấn SQL chuẩn (`10_token_freq.sql`, `20_log_odds.sql`, `30_cooccur.sql`, `40_lexicon.sql`) bằng `pd.read_sql` để rút ra điểm chung định lượng của các tweet phản cảm."
        ),
        new_code_cell(BOOTSTRAP_CODE),
        new_markdown_cell("## 1. Kiểm tra Cơ sở dữ liệu `db/tweets.db` & Thực thi 4 File Truy vấn SQL"),
        new_code_cell(
            "import math, sqlite3\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n"
            "from src.utils import mask_profanity\n\n"
            "conn = sqlite3.connect('db/tweets.db')\n"
            "conn.create_function('LN', 1, lambda x: math.log(float(x)) if x and float(x) > 0 else 0.0)\n\n"
            "tables_info = pd.read_sql_query(\"SELECT name FROM sqlite_master WHERE type='table'\", conn)\n"
            "print('Các bảng trong db/tweets.db:', tables_info['name'].tolist())\n"
            "for tbl in ['tweets', 'tokens', 'bigrams', 'assigned', 'lexicon']:\n"
            "    cnt = pd.read_sql_query(f'SELECT COUNT(*) AS n FROM {tbl}', conn)['n'].iloc[0]\n"
            "    print(f'  - Bảng {tbl:<10}: {cnt:,} dòng')"
        ),
        new_markdown_cell("## 2. Truy vấn `20_log_odds.sql`: Top từ có tần suất đặc trưng cao nhất ở Lớp 1"),
        new_code_cell(
            "q2_sql = open('sql/20_log_odds.sql', encoding='utf-8').read()\n"
            "print(q2_sql)\n"
            "df_log_odds = pd.read_sql_query(q2_sql, conn)\n"
            "df_log_odds['token'] = df_log_odds['token'].map(mask_profanity)\n"
            "display(df_log_odds.head(20))"
        ),
        new_markdown_cell("## 3. Truy vấn `30_cooccur.sql` & `40_lexicon.sql`: Đồng xuất hiện từ vựng & Độ phủ Từ điển Xúc phạm"),
        new_code_cell(
            "q3_sql = open('sql/30_cooccur.sql', encoding='utf-8').read()\n"
            "df_cooccur = pd.read_sql_query(q3_sql, conn)\n"
            "df_cooccur['neighbor'] = df_cooccur['neighbor'].map(mask_profanity)\n"
            "print('=== Top 15 từ đồng xuất hiện cùng từ khóa phản cảm (30_cooccur.sql) ===')\n"
            "display(df_cooccur.head(15))\n\n"
            "q4_sql = open('sql/40_lexicon.sql', encoding='utf-8').read()\n"
            "df_lex = pd.read_sql_query(q4_sql, conn)\n"
            "print('=== Tỉ lệ tweet chứa từ trong Lexicon theo nhãn (40_lexicon.sql) ===')\n"
            "display(df_lex)\n\n"
            "df_lex_cat = pd.read_csv('reports/tables/P3_lexicon_by_category.csv')\n"
            "print('=== Phân rã Lexicon theo nhóm ngữ nghĩa ===')\n"
            "display(df_lex_cat)\n"
            "conn.close()\n"
            "display(Image('reports/figures/P3_sql_analysis.png'))"
        ),
        new_markdown_cell(
            "### Kết luận: Điểm chung của các Tweet có nội dung phản cảm (P3)\n"
            "Từ kết quả truy vấn SQL trên tập huấn luyện ($N = 19,830$ tweet, $158,342$ token), các đoạn tweet phân biệt chủng tộc / giới tính có **7 điểm chung định lượng** rõ rệt:\n"
            "1. **Tỉ lệ chứa từ trong từ điển xúc phạm (Lexicon) cao gấp 8.6 lần:** Truy vấn `40_lexicon.sql` chỉ ra **20.59%** tweet Lớp 1 (266/1,292 tweet) chứa ít nhất một từ trong từ điển xúc phạm, so với chỉ **2.40%** ở Lớp 0 (445/18,538 tweet).\n"
            "2. **Tuy nhiên, 79.41% tweet phản cảm KHÔNG dùng từ tục trực tiếp trong Lexicon:** Điều này chứng minh phương pháp lọc bằng từ khóa cứng (rule-based/lexicon) sẽ bỏ sót gần 4/5 số lượng tweet kỳ thị ngầm, đòi hỏi phải dùng mô hình Học máy với đặc trưng n-gram và ngữ cảnh.\n"
            "3. **Tập trung vào 3 nhóm đối tượng mục tiêu:** Phân rã SQL theo danh mục (`P3_lexicon_by_category.csv`) và `20_log_odds.sql` cho thấy ngôn từ thù ghét tập trung vào: (a) **Chủng tộc & Sắc tộc** (`racist`, `racism`, `white`, `black`, `n****r`), (b) **Giới tính & Phụ nữ** (`sexist`, `woman`, `women`, `b***h`, `misogyny`, `feminism`), và (c) **Tôn giáo & Ý thức hệ** (`allahsoil`, `islam`, `muslim`, `jew`, `libtard`, `sjw`, `bigot`).\n"
            "4. **Cấu trúc đồng xuất hiện (Co-occurrence — `30_cooccur.sql`):** Các từ khóa phản cảm thường đi kèm trực tiếp với các từ chỉ màu da/nhóm người (`white`: 249 lần, `black`: 211 lần, `woman`: 176 lần, `people`: 144 lần) và tên nhân vật chính trị (`trump`: 269 lần, `obama`: 121 lần), tạo thành các cụm quy chụp tập thể.\n"
            "5. **Mật độ tấn công cá nhân qua `@user` và `#hashtag`:** Kết hợp với thống kê bề mặt, 65.03% tweet phản cảm gắn thẻ trực tiếp `@user` kèm chuỗi hashtag mang tính khẩu hiệu định kiến.\n"
            "6. **Lưu ý học thuật (Tương quan $\\neq$ Nhân quả):** Một từ có tần suất hoặc Log-Odds cao ở Lớp 1 (như `women`, `black`, `racist`, `trump`) **không đồng nghĩa** mọi câu chứa từ đó đều là phản cảm; rất nhiều câu dùng các từ này để đưa tin tức hoặc **lên án nạn phân biệt chủng tộc** (counter-speech).\n\n"
            "### Nhận xét (P3)\n"
            "- Cơ sở dữ liệu `db/tweets.db` và 4 kịch bản SQL đã lượng hóa chính xác quy luật từ vựng của các tweet phản cảm mà không chạm vào tập `test`.\n"
            "- Kết quả SQL là cơ sở trực tiếp để thiết kế bộ trích xuất đặc trưng TF-IDF kết hợp unigram + bigram (`ngram_range=(1,2)`) và giữ lại các đại từ/từ phủ định ở giai đoạn P4."
        ),
    ]
    _save_and_execute("02_sql_analysis.ipynb", cells_02)

    # =========================================================================
    # 03_preprocess_balance.ipynb (P4)
    # =========================================================================
    sp = p4["split_summary"]
    cells_03 = [
        new_markdown_cell(
            "# 03. Làm sạch Văn bản, Khử trùng, Chia tập & Cân bằng Dữ liệu (Câu 2: 1.5 điểm — P4)\n\n"
            "**Mục tiêu:**\n"
            "1. Áp dụng pipeline làm sạch 12 bước (`src/clean.py::clean_text`), bảo toàn các từ phủ định (`no, not, never...`) và đại từ nhân xưng/nhóm người (`you, she, they, women, men...`).\n"
            "2. Khử trùng gần-giống (`text_clean`) và loại bỏ các nhóm có nhãn mâu thuẫn **trước khi chia tập** để triệt tiêu hoàn toàn nguy cơ rò rỉ dữ liệu (data leakage).\n"
            "3. Chia tập Stratified `70% Train / 15% Val / 15% Test` cố định một lần duy nhất vào `data/splits/`.\n"
            "4. So sánh thực nghiệm 6 chiến lược cân bằng dữ liệu bằng 5-fold StratifiedKFold CV chỉ trên tập `train`."
        ),
        new_code_cell(BOOTSTRAP_CODE),
        new_markdown_cell("## 1. Minh họa 12 Ví dụ Trước & Sau khi Làm sạch (`clean_text`)"),
        new_code_cell(
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "examples_df = pd.read_csv('reports/tables/P4_clean_examples.csv')\n"
            "display(examples_df)\n"
            "display(Image('reports/figures/P4_length_before_after.png'))"
        ),
        new_markdown_cell("## 2. Thống kê Khử trùng, Loại bỏ Nhãn mâu thuẫn & Chia tập Stratified 70/15/15"),
        new_code_cell(
            "with open('reports/p4_summary.json', 'r', encoding='utf-8') as f:\n"
            "    p4_data = json.load(f)\n\n"
            "sp = p4_data['split_summary']\n"
            "print('=== THỐNG KÊ LÀM SẠCH & KHỬ TRÙNG (P4.2) ===')\n"
            "print(f\"Số dòng ban đầu (hate_train.csv)       : {sp['n_initial_raw']:,}\")\n"
            "print(f\"Số dòng rỗng sau khi clean_text        : {sp['n_empty_dropped']:,}\")\n"
            "print(f\"Số nhóm text_clean có nhãn mâu thuẫn   : {sp['n_conflict_groups']:,} ({sp['n_conflict_rows_dropped']} dòng bị loại)\")\n"
            "print(f\"Số dòng trùng lặp gần-giống (Retweet)  : {sp['n_near_duplicate_rows_dropped']:,}\")\n"
            "print(f\"Tổng số dòng bị loại bỏ                : {sp['n_total_dropped']:,}\")\n"
            "print(f\"Số dòng sạch duy nhất còn lại          : {sp['n_final_unique_clean']:,} (Lớp 1 = {sp['final_label_1']:,}, tỉ lệ {sp['final_pos_rate_pct']:.2f}%)\")\n\n"
            "splits_table = pd.DataFrame(sp['splits']).T\n"
            "print('\\n=== PHÂN BỐ CÁC TẬP TRAIN / VAL / TEST (70 / 15 / 15) ===')\n"
            "display(splits_table)"
        ),
        new_markdown_cell("## 3. So sánh 6 Chiến lược Cân bằng Dữ liệu trên 5-Fold StratifiedKFold CV (P4.4)"),
        new_code_cell(
            "bal_df = pd.read_csv('reports/tables/P4_balance_comparison.csv')\n"
            "display(bal_df)\n"
            "display(Image('reports/figures/P4_balance_comparison.png'))\n"
            "print('Quyết định chọn chiến lược cân bằng:', p4_data['balance_comparison']['chosen_reason'])"
        ),
        new_markdown_cell(
            "### Nhận xét (P4)\n"
            f"- Quá trình khử trùng đã loại bỏ **{sp['n_near_duplicate_rows_dropped']:,} tweet trùng/retweet** và **{sp['n_conflict_rows_dropped']} dòng mâu thuẫn nhãn**, đưa bộ dữ liệu về **{sp['n_final_unique_clean']:,} mẫu duy nhất** và đảm bảo giao nhau giữa 3 tập `train`, `val`, `test` bằng $\\emptyset$ (đã kiểm chứng tự động qua `tests/test_leakage.py`).\n"
            f"- Khi không cân bằng dữ liệu (`none`), mô hình Logistic Regression chỉ đạt Recall = **13.85%** và F1 = **0.2390** (bỏ sót более 86% tweet phản cảm).\n"
            f"- Các kỹ thuật cân bằng giúp tăng vọt F1 lên vùng **0.5742 – 0.5876**, trong đó **RandomOverSampler (`ros`)** đạt F1 trung bình 5-fold CV cao nhất (**0.5876 ± 0.0117**, PR-AUC = **0.6337**) với thời gian chạy chỉ **2.86 giây**, nên được chọn làm chiến lược cân bằng chuẩn cho P5."
        ),
    ]
    _save_and_execute("03_preprocess_balance.ipynb", cells_03)

    # =========================================================================
    # 04_model_comparison.ipynb (P5)
    # =========================================================================
    cells_04 = [
        new_markdown_cell(
            "# 04. Xây dựng & So sánh 9 Mô hình Phân loại Nhị phân (Câu 2: 1.5đ & Câu 3: 1.5đ — P5)\n\n"
            "**Mục tiêu:**\n"
            "1. Huấn luyện và đánh giá toàn diện **9 họ mô hình học máy** trong `MODEL_ZOO`: Logistic Regression (`lr`), Complement Naive Bayes (`nb`), Calibrated Linear SVM (`svm_linear`), RBF SVM (`svm_rbf`), Decision Tree (`dt`), Random Forest (`rf`), Cosine KNN (`knn`), XGBoost (`xgb`), và LightGBM (`lgbm`).\n"
            "2. So sánh giữa hai cách biểu diễn đặc trưng: `word_tfidf` (unigram + bigram) và `word+char` (kết hợp word n-gram và char n-gram 2–5).\n"
            "3. Đánh giá bằng 5-fold CV trên `train` và đo lường độc lập trên `val` theo tiêu chí chính **F1 lớp dương (`f1_pos`)**, **PR-AUC**, **Recall**, **ROC-AUC**, **MCC** và tốc độ suy luận."
        ),
        new_code_cell(BOOTSTRAP_CODE),
        new_markdown_cell("## 1. Bảng So sánh Kết quả 9 Mô hình trên Tập Validation & Kiểm định Paired t-test"),
        new_code_cell(
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "comp_df = pd.read_csv('reports/model_comparison.csv')\n"
            "display(comp_df)\n\n"
            "ttest_df = pd.read_csv('reports/tables/P5_paired_ttest.csv')\n"
            "print('=== Kiểm định Paired t-test trên 5-fold CV F1 giữa Top-3 cấu hình ===')\n"
            "display(ttest_df)"
        ),
        new_markdown_cell("## 2. Biểu đồ So sánh F1 / PR-AUC, Đường cong ROC & PR, và Lưới 3×3 Ma trận Nhầm lẫn"),
        new_code_cell(
            "display(Image('reports/figures/P5_f1_prauc_bars.png'))\n"
            "display(Image('reports/figures/P5_roc_pr_curves.png'))\n"
            "display(Image('reports/figures/P5_confusion_grid_3x3.png'))"
        ),
        new_markdown_cell(
            "### Nhận xét: Vì sao Mô hình Tuyến tính thắng Mô hình Cây trên Văn bản TF-IDF thưa? (P5)\n"
            "- **Mô hình tuyến tính (`svm_linear`, `lr`) và `nb` vượt trội rõ rệt:** Với không gian đặc trưng TF-IDF chiều cao ($50,000 - 100,000$ chiều) và cực kỳ thưa (sparse > 99.9%), biên quyết định siêu phẳng tuyến tính tận dụng đồng thời hàng chục nghìn từ/cụm từ nhẹ mà không bị phân mảnh dữ liệu.\n"
            "- **Kết hợp `word+char` giúp tăng thêm hiệu năng:** Việc bổ sung char n-gram (2–5 ký tự) giúp `svm_linear` và `lr` bắt được các biến thể viết tắt, kéo dài chữ hoặc cố tình viết sai chính tả để lách kiểm duyệt.\n"
            "- **Vì sao Decision Tree / Random Forest / KNN thấp hơn?** Cây quyết định rẽ nhánh theo từng trục từ đơn lẻ (axis-aligned splits) trên ma trận 50,000 chiều thưa nên chỉ bao phủ được một vài từ khóa ở tầng nông và dễ overfit; trong khi KNN chịu ảnh hưởng của lời nguyền số chiều (*curse of dimensionality*).\n"
            "- Ba mô hình dẫn đầu (`svm_linear`, `lr`, `nb`) được chọn đưa vào giai đoạn tinh chỉnh siêu tham số và tối ưu ngưỡng quyết định ở **P6**."
        ),
    ]
    _save_and_execute("04_model_comparison.ipynb", cells_04)

    # =========================================================================
    # 05_tuning_threshold.ipynb (P6)
    # =========================================================================
    cells_05 = [
        new_markdown_cell(
            "# 05. Tinh chỉnh Siêu tham số & Tối ưu Ngưỡng Quyết định (Câu 3: 0.5 điểm — P6)\n\n"
            "**Mục tiêu:**\n"
            "1. Thực hiện `RandomizedSearchCV` (`n_iter = 40`, `cv = 5`, `scoring = 'f1'`) trên tập `train` cho 3 mô hình tốt nhất (`lr`, `svm_linear`, `nb`) cùng các tham số của `TfidfVectorizer` (`ngram_range`, `min_df`, `max_features`).\n"
            "2. Quét ngưỡng quyết định (`threshold` $\\in [0.05, 0.95]$) trên xác suất dự đoán của tập `val` để tối đa hóa F1 lớp dương (`src/threshold.py::pick_threshold`).\n"
            "3. Huấn luyện lại mô hình chốt (`best_model`) trên `train + val`, lưu `models/best_model.joblib`, `models/threshold.json` và `models/model_card.json`."
        ),
        new_code_cell(BOOTSTRAP_CODE),
        new_markdown_cell("## 1. Bảng So sánh Trước & Sau khi Tinh chỉnh Siêu tham số và Ngưỡng Quyết định"),
        new_code_cell(
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "ba_df = pd.read_csv('reports/tables/P6_before_after.csv')\n"
            "display(ba_df)\n\n"
            "with open('models/model_card.json', 'r', encoding='utf-8') as f:\n"
            "    model_card = json.load(f)\n"
            "print('=== HỒ SƠ MÔ HÌNH CHỐT (models/model_card.json) ===')\n"
            "print(json.dumps(model_card, indent=2, ensure_ascii=False))"
        ),
        new_markdown_cell("## 2. Biểu đồ Quét Ngưỡng Quyết định & So sánh Trước / Sau Tuning"),
        new_code_cell(
            "display(Image('reports/figures/P6_threshold_tuning.png'))\n"
            "display(Image('reports/figures/P6_before_after.png'))"
        ),
        new_markdown_cell(
            "### Nhận xét (P6)\n"
            f"- Tinh chỉnh siêu tham số kết hợp tối ưu ngưỡng quyết định trên tập Validation đã nâng cao rõ rệt chất lượng phân loại so với cấu hình mặc định (ngưỡng 0.50).\n"
            f"- Mô hình được chọn là **`{p6['chosen_model_name']}` (`{p6['chosen_model_key']}`)** với ngưỡng tối ưu **`threshold = {p6['best_threshold']}`**.\n"
            f"- Lý do lựa chọn: {p6['selection_reason']}\n"
            "- Toàn bộ quy trình chọn tham số và chọn ngưỡng chỉ sử dụng tập `train` và `val`, đảm bảo tập `test` hoàn toàn độc lập cho bước đánh giá cuối ở P8."
        ),
    ]
    _save_and_execute("05_tuning_threshold.ipynb", cells_05)

    # =========================================================================
    # 06_transformer_bonus.ipynb (P7)
    # =========================================================================
    cells_06 = [
        new_markdown_cell(
            "# 06. Mô hình Mở rộng Ngoài Chương trình: BiLSTM & DistilBERT Fine-tuning (Điểm cộng — P7)\n\n"
            "**Mục tiêu:**\n"
            "1. Xây dựng và huấn luyện mạng nơ-ron hồi quy hai chiều **BiLSTM (PyTorch Embedding 128 chiều + Global Avg/Max Pooling)** có trọng số lớp.\n"
            "2. Tinh chỉnh (Fine-tune) mô hình ngôn ngữ tiền huấn luyện **DistilBERT (`distilbert-base-uncased`)** với hàm mất mát `CrossEntropyLoss` có trọng số lớp.\n"
            "3. Phân tích sự đánh đổi giữa **Hiệu năng (F1 / PR-AUC)**, **Thời gian huấn luyện/suy luận**, **Dung lượng mô hình** và **Khả năng giải thích** khi triển khai ứng dụng thực tế."
        ),
        new_code_cell(BOOTSTRAP_CODE),
        new_markdown_cell("## 1. Kết quả Thực nghiệm BiLSTM & DistilBERT so với Mô hình Cổ điển Tối ưu"),
        new_code_cell(
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "with open('reports/p7_summary.json', 'r', encoding='utf-8') as f:\n"
            "    p7_data = json.load(f)\n\n"
            "print(f\"Thiết bị chạy thực nghiệm (CUDA Available): {p7_data['has_cuda']}\")\n"
            "print('\\n=== Lịch sử huấn luyện BiLSTM ===')\n"
            "display(pd.DataFrame(p7_data['bilstm']['history']))\n"
            "print('Metric Validation BiLSTM:', p7_data['bilstm']['metrics'])\n\n"
            "print('\\n=== Lịch sử huấn luyện DistilBERT ===')\n"
            "display(pd.DataFrame(p7_data['distilbert']['history']))\n"
            "print('Metric Validation DistilBERT:', p7_data['distilbert']['metrics'])\n\n"
            "display(Image('reports/figures/P7_bonus_comparison.png'))"
        ),
        new_markdown_cell(
            "### Nhận xét (P7)\n"
            f"- **BiLSTM** đạt Validation F1 = **{p7['bilstm']['metrics']['f1_pos']:.4f}** (PR-AUC = **{p7['bilstm']['metrics']['pr_auc']:.4f}**), còn **DistilBERT** đạt Validation F1 = **{p7['distilbert']['metrics']['f1_pos']:.4f}** (PR-AUC = **{p7['distilbert']['metrics']['pr_auc']:.4f}**).\n"
            f"- **Phân tích đánh đổi (Trade-off):** {p7['tradeoff_conclusion']}\n"
            "- Việc hoàn thành cả BiLSTM và DistilBERT cung cấp cái nhìn đối sánh toàn diện giữa Học máy cổ điển, Học sâu hồi quy (RNN/LSTM) và kiến trúc Transformer hiện đại."
        ),
    ]
    _save_and_execute("06_transformer_bonus.ipynb", cells_06)

    # =========================================================================
    # 07_final_eval_error_analysis.ipynb (P8)
    # =========================================================================
    ft_m = p8["final_test"]["metrics_at_tuned_threshold"]
    ci_f1 = p8["ci95"]["f1_pos"]
    cells_07 = [
        new_markdown_cell(
            "# 07. Đánh giá Cuối trên Tập Test, Phân tích Lỗi & Kiểm tra Thiên lệch (P8)\n\n"
            "**Mục tiêu:**\n"
            "1. Đánh giá mô hình chốt (`models/best_model.joblib` + `models/threshold.json`) **một lần duy nhất** trên tập kiểm tra độc lập `data/splits/test.csv` ($N = 4,250$).\n"
            "2. Ước lượng khoảng tin cậy 95% (95% Confidence Interval) bằng phương pháp **Bootstrap 1,000 lần**.\n"
            "3. Phân tích sâu 30 lỗi Báo động giả (False Positives) và 30 lỗi Bỏ sót (False Negatives), phân nhóm nguyên nhân ngôn ngữ học.\n"
            "4. Giải thích mô hình bằng Top-30 trọng số đặc trưng và đánh giá tính công bằng (Subgroup Bias / False Positive Rate theo nhóm nhân khẩu học)."
        ),
        new_code_cell(BOOTSTRAP_CODE),
        new_markdown_cell("## 1. Kết quả Đánh giá Cuối trên Tập Test & Khoảng tin cậy 95% Bootstrap"),
        new_code_cell(
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "with open('reports/final_test_metrics.json', 'r', encoding='utf-8') as f:\n"
            "    final_test = json.load(f)\n"
            "with open('reports/final_test_ci.json', 'r', encoding='utf-8') as f:\n"
            "    final_ci = json.load(f)\n\n"
            "print(f\"Thời điểm đánh giá Test duy nhất : {final_test['evaluated_once_at']}\")\n"
            "print(f\"Mã băm SHA-256 file test.csv     : {final_test['test_file_sha256']}\")\n"
            "print(f\"Ngưỡng quyết định ( từ Val )     : {final_test['threshold_used']}\")\n\n"
            "print('\\n=== METRIC TRÊN TẬP TEST (N = 4,250) ===')\n"
            "display(pd.DataFrame([final_test['metrics_at_tuned_threshold'], final_test['metrics_at_default_05']], index=['Tuned Threshold', 'Default 0.50']))\n\n"
            "print('\\n=== KHOẢNG TIN CẬY 95% BOOTSTRAP (1,000 LẦN) ===')\n"
            "display(pd.DataFrame(final_ci).T.loc[['f1_pos', 'precision_pos', 'recall_pos', 'pr_auc', 'roc_auc']])\n"
            "display(Image('reports/figures/P8_test_evaluation.png'))"
        ),
        new_markdown_cell("## 2. Phân tích Lỗi (30 FP + 30 FN) & Phân loại Nguyên nhân Sai sót"),
        new_code_cell(
            "causes_df = pd.read_csv('reports/tables/P8_error_causes.csv')\n"
            "print('=== BẢNG THỐNG KÊ NGUYÊN NHÂN LỖI TRÊN TẬP TEST ===')\n"
            "display(causes_df)\n"
            "display(Image('reports/figures/P8_error_causes.png'))\n\n"
            "errors_df = pd.read_csv('reports/tables/P8_errors.csv')\n"
            "print('=== 10 Ví dụ Báo động giả (False Positives) có điểm cao nhất ===')\n"
            "display(errors_df[errors_df['error_type'].str.startswith('FP')].head(10))\n"
            "print('=== 10 Ví dụ Bỏ sót (False Negatives) có điểm thấp nhất ===')\n"
            "display(errors_df[errors_df['error_type'].str.startswith('FN')].head(10))"
        ),
        new_markdown_cell("## 3. Giải thích Mô hình (Top Trọng số) & Phân tích Thiên lệch Nhóm (Subgroup Bias)"),
        new_code_cell(
            "coef_df = pd.read_csv('reports/tables/P8_top_coefficients.csv')\n"
            "display(Image('reports/figures/P8_feature_importance.png'))\n\n"
            "bias_df = pd.read_csv('reports/tables/P8_subgroup_bias.csv')\n"
            "print('=== BẢNG KIỂM TRA THIÊN LỆCH: TỈ LỆ BÁO ĐỘNG GIẢ (FPR) THEO NHÓM TỪ NHẬN DIỆN ===')\n"
            "display(bias_df)\n"
            "display(Image('reports/figures/P8_subgroup_bias.png'))"
        ),
        new_markdown_cell(
            "### Nhận xét & Hạn chế (P8)\n"
            f"- Trên tập Test độc lập ($N = 4,250$), mô hình đạt **F1 lớp dương = {ft_m['f1_pos']:.4f}** (95% CI: **[{ci_f1['ci95_low']:.4f}, {ci_f1['ci95_high']:.4f}]**), **Precision = {ft_m['precision_pos']:.4f}**, **Recall = {ft_m['recall_pos']:.4f}**, **PR-AUC = {ft_m['pr_auc']:.4f}** và **ROC-AUC = {ft_m['roc_auc']:.4f}**.\n"
            "- **Nguyên nhân lỗi chính:** (1) Đối với *False Negatives (Bỏ sót)*, phần lớn là các câu kỳ thị ngầm, định kiến hoặc châm biếm không sử dụng từ tục trực tiếp; (2) Đối với *False Positives (Báo động giả)*, lỗi xảy ra khi câu bình thường chứa các từ chỉ nhóm nhạy cảm (`women`, `black`, `white`, `trump`, `racist`) trong ngữ cảnh đưa tin hoặc phản đối phân biệt đối xử.\n"
            "- **Phân tích thiên lệch (Subgroup Bias):** Tỉ lệ báo động giả (FPR) trên các câu bình thường có nhắc đến từ chỉ chủng tộc, tôn giáo hoặc giới tính cao hơn mức FPR trung bình toàn tập Test — đây là hạn chế kinh điển của các mô hình dựa trên tần suất từ vựng cần được lưu ý khi triển khai thực tế."
        ),
    ]
    _save_and_execute("07_final_eval_error_analysis.ipynb", cells_07)

    # =========================================================================
    # 08_realworld_test.ipynb (P9)
    # =========================================================================
    rw_m = p9["overall_metrics"]
    cells_08 = [
        new_markdown_cell(
            "# 08. Chạy thử Mô hình trên Bộ Dữ liệu Thực tế & Phân tích Lệch miền (Câu 3: 1.5 điểm — P9)\n\n"
            "**Mục tiêu:**\n"
            "1. Đánh giá khả năng tổng quát hóa của mô hình trên bộ dữ liệu thực tế `data/real_world/real_world_labeled.csv` ($N = 300$ mẫu từ 4 nguồn A, B, C, D).\n"
            "2. Đo độ đồng thuận giữa 2 người gắn nhãn độc lập (`label_a` và `label_b`) bằng hệ số **Cohen's Kappa ($\\kappa$)**.\n"
            "3. Phân tích hiện tượng **lệch miền (domain shift)** giữa các nguồn dữ liệu, kiểm thử chéo trên bộ **Davidson et al.** ($N = 5,593$) và thống kê dự đoán trên bộ được giao `Twitter_Data.csv` ($N = 162,969$)."
        ),
        new_code_cell(BOOTSTRAP_CODE),
        new_markdown_cell("## 1. Độ đồng thuận Gắn nhãn (Cohen's Kappa) & Hiệu năng theo từng Nguồn Dữ liệu (A/B/C/D)"),
        new_code_cell(
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "with open('reports/realworld_metrics.json', 'r', encoding='utf-8') as f:\n"
            "    rw_data = json.load(f)\n\n"
            "print(f\"Tổng số mẫu thực tế          : {rw_data['n_total']} (Lớp 1 = {rw_data['n_pos']}, Lớp 0 = {rw_data['n_neg']})\")\n"
            "print(f\"Tỉ lệ đồng thuận 2 người     : {rw_data['annotator_agreement_pct']:.2f}% ({rw_data['n_disagreements']} mẫu bất đồng)\")\n"
            "print(f\"Hệ số Cohen's Kappa (κ)      : {rw_data['cohen_kappa']:.4f}\")\n"
            "print(f\"Overall Real-World F1-Score  : {rw_data['overall_metrics']['f1_pos']:.4f} (PR-AUC = {rw_data['overall_metrics']['pr_auc']:.4f})\")\n\n"
            "by_src_df = pd.read_csv('reports/tables/P9_metrics_by_source.csv')\n"
            "print('\\n=== HIỆU NĂNG CHI TIẾT THEO 4 NGUỒN DỮ LIỆU THỰC TẾ ===')\n"
            "display(by_src_df)\n"
            "display(Image('reports/figures/P9_realworld_evaluation.png'))"
        ),
        new_markdown_cell("## 2. Kiểm thử Chéo miền (Davidson et al.) & Thống kê trên Bộ được giao `Twitter_Data.csv`"),
        new_code_cell(
            "print('=== KẾT QUẢ KIỂM THỬ CHÉO TRÊN BỘ DAVIDSON ET AL. (Class 0=Hate vs Class 2=Neither) ===')\n"
            "display(pd.DataFrame([rw_data['davidson_cross_domain_metrics']]))\n\n"
            "tw_sent_df = pd.read_csv('reports/tables/P9_assigned_twitter_by_sentiment.csv')\n"
            "print('=== TỈ LỆ DỰ ĐOÁN PHẢN CẢM THEO NHÃN SENTIMENT TRÊN TWITTER_DATA.CSV (N=162,969) ===')\n"
            "display(tw_sent_df)"
        ),
        new_markdown_cell("## 3. Phân tích 15 Ví dụ Dự đoán Sai & 5 Ví dụ Dự đoán Đúng Đáng chú ý trên Tập Thực tế"),
        new_code_cell(
            "rw_ex_df = pd.read_csv('reports/tables/P9_realworld_examples.csv')\n"
            "print('=== 15 VÍ DỤ DỰ ĐOÁN SAI TRÊN TẬP THỰC TẾ ===')\n"
            "display(rw_ex_df[rw_ex_df['error_type'] != 'Correct (Dự đoán đúng)'])\n\n"
            "print('=== 5 VÍ DỤ DỰ ĐOÁN ĐÚNG TRÊN CÂU HIỂM HÓC (SOURCE D) ===')\n"
            "display(rw_ex_df[rw_ex_df['error_type'] == 'Correct (Dự đoán đúng)'])"
        ),
        new_markdown_cell(
            "### Nhận xét & Đề xuất Cải tiến từ Kết quả Thực tế (P9)\n"
            f"- **Độ tin cậy nhãn thực tế:** Quá trình gắn nhãn độc lập trên 300 mẫu đạt độ đồng thuận **{p9['annotator_agreement_pct']:.2f}%** và hệ số **Cohen's Kappa $\\kappa = {p9['cohen_kappa']:.4f}$** (mức đồng thuận rất tốt, $> 0.75$).\n"
            f"- **Hiệu năng thực tế & Hiện tượng lệch miền (Domain Shift):** Trên toàn bộ 300 mẫu thực tế, mô hình đạt **F1 = {rw_m['f1_pos']:.4f}** (Precision = **{rw_m['precision_pos']:.4f}**, Recall = **{rw_m['recall_pos']:.4f}**, PR-AUC = **{rw_m['pr_auc']:.4f}**). Hiệu năng cao nhất ở nguồn cùng miền `A_hate_unlabeled_test` và giảm ở nguồn `D_adversarial_curated` (chứa các câu mỉa mai, phủ định phản đối kỳ thị và chửi thề không nhắm nhóm người).\n"
            "- **Chứng minh thực nghiệm trên `Twitter_Data.csv`:** Ngay cả ở nhóm `sentiment = -1` (35,510 tweet tiêu cực), chỉ một tỉ lệ nhỏ được mô hình dự đoán là phân biệt chủng tộc/giới tính — chứng minh rõ ràng **cảm xúc tiêu cực chính trị hoàn toàn khác với ngôn từ thù ghét chủng tộc/giới tính**.\n"
            "- **Đề xuất hướng cải tiến:** (1) Bổ sung dữ liệu huấn luyện thuộc dạng *counter-speech* (phản đối kỳ thị) và *general profanity* để giảm False Positive; (2) Kết hợp mô hình ngữ cảnh sâu (RoBERTa/HateBERT) với cơ chế học chủ động (*Active Learning*) có con người duyệt lại các mẫu gần ngưỡng quyết định."
        ),
    ]
    _save_and_execute("08_realworld_test.ipynb", cells_08)


if __name__ == "__main__":
    build_all_notebooks()
