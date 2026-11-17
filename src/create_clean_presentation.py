"""Create a clean, natural student presentation version of the notebook, app, report, and slides
with all rubric/internal markers ('điểm cộng', 'bonus', 'P0..P12', 'V0..V8', 'B0..B6') removed."""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BOOTSTRAP_LINES = [
    "# === THIẾT LẬP MÔI TRƯỜNG & NẠP THƯ VIỆN DỰ ÁN ===\n",
    "import os, sys, subprocess\n",
    "from pathlib import Path\n",
    "\n",
    "os.environ.setdefault(\"OPENBLAS_NUM_THREADS\", \"1\")\n",
    "os.environ.setdefault(\"OMP_NUM_THREADS\", \"1\")\n",
    "REPO_URL = \"https://github.com/nhungoc-1/DoAnKHDL_DeTai_Tweet-Hate.git\"\n",
    "\n",
    "IN_COLAB = \"google.colab\" in sys.modules\n",
    "if IN_COLAB:\n",
    "    print(\"[SETUP] Đang khởi tạo môi trường Google Colab...\", flush=True)\n",
    "    for candidate in [\n",
    "        Path(\"/content/DoAnKHDL_DeTai_Tweet-Hate\"),\n",
    "        Path(\"/content/drive/MyDrive/DoAnKHDL_DeTai_Tweet-Hate\"),\n",
    "        Path(\"/content/drive/MyDrive/DO_AN_KHDL\"),\n",
    "        Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd(),\n",
    "    ]:\n",
    "        if (candidate / \"config.yaml\").exists():\n",
    "            PROJECT_DIR = candidate\n",
    "            subprocess.run([\"git\", \"-C\", str(PROJECT_DIR), \"pull\", \"--ff-only\"], check=False)\n",
    "            break\n",
    "    else:\n",
    "        PROJECT_DIR = Path(\"/content/DoAnKHDL_DeTai_Tweet-Hate\")\n",
    "        print(f\"[SETUP] Đang tải dự án từ GitHub vào {PROJECT_DIR} (mất ~3 giây)...\", flush=True)\n",
    "        subprocess.run([\"git\", \"clone\", \"--depth\", \"1\", REPO_URL, str(PROJECT_DIR)], check=True)\n",
    "    print(\"[SETUP] Đang kiểm tra & cài đặt các thư viện bổ sung (mất ~5-10 giây)...\", flush=True)\n",
    "    subprocess.run([\n",
    "        sys.executable, \"-m\", \"pip\", \"install\", \"-q\",\n",
    "        \"imbalanced-learn\", \"emoji\", \"kagglehub\", \"datasets\",\n",
    "        \"pyvi\", \"langdetect\", \"python-docx\", \"python-pptx\", \"wordcloud\"\n",
    "    ], check=False)\n",
    "    import nltk\n",
    "    for pkg in [\"stopwords\", \"wordnet\", \"omw-1.4\"]:\n",
    "        nltk.download(pkg, quiet=True)\n",
    "else:\n",
    "    PROJECT_DIR = Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd()\n",
    "\n",
    "os.chdir(PROJECT_DIR)\n",
    "if str(PROJECT_DIR) not in sys.path:\n",
    "    sys.path.insert(0, str(PROJECT_DIR))\n",
    "\n",
    "from src.utils import set_seed, load_config, mask_profanity\n",
    "set_seed(42)\n",
    "CFG = load_config()\n",
    "(PROJECT_DIR / \"db\").mkdir(parents=True, exist_ok=True)\n",
    "print(f\"[HOÀN TẤT] IN_COLAB = {IN_COLAB} | PROJECT_DIR = {PROJECT_DIR}\")",
]

# Ordered replacements to make prose natural academic Vietnamese for instructor presentation
TEXT_REPLACEMENTS: list[tuple[str, str]] = [
    # Remove "điểm cộng" / "bonus" phrasing
    (r"\(\s*[Đđ]iểm cộng\s*[—\-–]?\s*[A-Za-z0-9]*\s*\)", "(Mở rộng nâng cao)"),
    (r"\(\s*[Đđ]iểm cộng[^)]*\)", "(Mở rộng nâng cao)"),
    (r"[Đđ]iểm cộng\s*\([A-Za-z0-9]+\)", "Mở rộng nâng cao"),
    (r"mô hình điểm cộng", "mô hình học sâu mở rộng"),
    (r"Mô hình điểm cộng", "Mô hình học sâu mở rộng"),
    (r"Mô hình Điểm cộng", "Mô hình Học sâu Mở rộng"),
    (r"[Đđ]iểm cộng ngoài chương trình", "phần mở rộng nâng cao"),
    (r"[Đđ]iểm cộng", "phần mở rộng"),
    (r"Bonus Deep Learning", "Deep Learning"),
    (r"PyTorch Bonus", "PyTorch Deep Learning"),
    (r"Fine-tuned Bonus", "Fine-tuned Transformer"),
    (r"\+\s*Bonus", "+ Mở rộng"),
    (r"\[Bonus\]", "[Mở rộng]"),
    # Remove internal spec codes P0..P12, V0..V8, B0..B6 in headings/text
    (r"\s*[—\-–]\s*P\d+(?:\.\d+)?(?:\s*&\s*P\d+)?", ""),
    (r"\s*[—\-–]\s*V\d+(?:\s*&\s*V\d+)?", ""),
    (r"\s*[—\-–]\s*B\d+(?:\s*/\s*B\d+)?", ""),
    (r"\(\s*P\d+(?:\.\d+)?(?:\s*[&,\-→]\s*P\d+(?:\.\d+)?)*\s*\)", ""),
    (r"\(\s*V\d+(?:\.\d+)?(?:\s*[&,\-→]\s*V\d+(?:\.\d+)?)*\s*\)", ""),
    (r"\bP0\s*→\s*P12\b", "toàn bộ quy trình"),
    (r"\bP1\s*→\s*P12\b", "toàn bộ quy trình"),
    (r"\bP0\s*->\s*P12\b", "toàn bộ quy trình"),
    (r"\bV0\s*→\s*V8\b", "phần mở rộng Tiếng Việt"),
    (r"\bV0\s*->\s*V8\b", "phần mở rộng Tiếng Việt"),
    (r"\bV1\s*→\s*V7\b", "phần thực nghiệm Tiếng Việt"),
    (r"\bP0\s*→\s*P10\b", "từ thu thập dữ liệu đến xây dựng ứng dụng"),
    (r"\bở\s+P(\d+(?:\.\d+)?)\b", "ở bước thực nghiệm \\1"),
    (r"\bgiai đoạn\s+P(\d+)\b", "giai đoạn \\1"),
    (r"\bbước\s+V(\d+)\b", "bước \\1"),
    (r"\bở\s+V(\d+)\b", "ở phần thực nghiệm \\1"),
    # Clean bootstrap / AI-sounding headings
    (r"# === CELL BOOTSTRAP CHUẨN CHO MỌI NOTEBOOK TRÊN GOOGLE COLAB & LOCAL ===", "# === THIẾT LẬP MÔI TRƯỜNG & NẠP THƯ VIỆN DỰ ÁN ==="),
    (r"# === CELL BOOTSTRAP CHUẨN CHO MỌI NOTEBOOK TRONG DỰ ÁN ===", "# === THIẾT LẬP MÔI TRƯỜNG & NẠP THƯ VIỆN DỰ ÁN ==="),
    (r"\[BOOTSTRAP\]", "[SETUP]"),
    (r"\[BOOTSTRAP HOÀN TẤT\]", "[HOÀN TẤT]"),
    (r"Thiết lập cell `bootstrap` chuẩn", "Thiết lập môi trường thực thi"),
    (r"Cell `bootstrap`", "Ô khởi tạo môi trường"),
    (r"Phương án A", "hướng tiếp cận kết hợp dữ liệu chuẩn"),
    (r"câu hiểm hóc", "câu tình huống phức tạp"),
    (r"Câu hiểm hóc", "Câu tình huống phức tạp"),
    (r"CÂU HIỂM HÓC", "CÂU TÌNH HUỐNG PHỨC TẠP"),
    (r"hiểm hóc", "phức tạp"),
    (r"более", "hơn"),
    (r"\(Đã che từ tục\.6\)", "(Đã che từ tục)"),
    (r"Biểu đồ P2_0(\d)", r"Biểu đồ \1"),
    (r"Biểu đồ P2_10", "Biểu đồ 10"),
    (r"P2_05, P2_05b & P2_06", "5a, 5b & 6"),
    (r"P2_07 & P2_08", "7 & 8"),
    (r"P2_09 & P2_10", "9 & 10"),
    (r"\(`V1_01`\s*→\s*`V1_08`\)", "(từ Biểu đồ 1 đến Biểu đồ 8)"),
    (r"\bcho P5\b", "cho bước huấn luyện mô hình"),
    (r"ở \*\*P6\*\*", "ở bước tiếp theo"),
    (r"# 01\. Thu thập Dữ liệu & Thống kê Mô tả — EDA \(Câu 1: 2\.0 điểm\)", "# Câu 1 (2,0 điểm): Thu thập và phân tích dữ liệu — Phần 1: Phân tích bộ dữ liệu & Thống kê mô tả"),
    (r"# 02\. Phân tích SQL: Tìm điểm chung của các Tweet Phản cảm \(Yêu cầu 1\)", "# Câu 1 (2,0 điểm): Thu thập và phân tích dữ liệu — Phần 2: Truy vấn SQL tìm điểm chung của các tweet phản cảm"),
    (r"# 03\. Làm sạch Văn bản, Khử trùng, Chia tập & Cân bằng Dữ liệu \(Câu 2: 1\.5 điểm\)", "# Câu 2 (3,0 điểm): Thiết kế mô hình học máy — Phần 1: Làm sạch dữ liệu, Chuẩn hóa và Cân bằng dữ liệu"),
    (r"# 04\. Xây dựng & So sánh 9 Mô hình Phân loại Nhị phân \(Câu 2: 1\.5đ & Câu 3: 1\.5đ\)", "# Câu 2 & Câu 3: Xây dựng các mô hình dữ liệu, tập huấn luyện, tiêu chí đánh giá & So sánh hiệu quả các mô hình"),
    (r"# 05\. Tinh chỉnh Siêu tham số & Tối ưu Ngưỡng Quyết định \(Câu 3: 0\.5 điểm\)", "# Câu 3 (3,0 điểm): So sánh và lựa chọn mô hình học máy — Điều chỉnh các tham số để tối ưu hóa mô hình"),
    (r"# 06\. Mô hình Mở rộng Ngoài Chương trình: BiLSTM & DistilBERT Fine-tuning \(Mở rộng nâng cao\)", "# Câu 3 (Mở rộng Học sâu): Thực nghiệm mô hình mạng nơ-ron BiLSTM & DistilBERT"),
    (r"# 07\. Đánh giá Cuối trên Tập Test, Phân tích Lỗi & Kiểm tra Thiên lệch[ \t]*", "# Câu 3: Đánh giá mô hình được chọn trên tập Test độc lập & Phân tích lỗi"),
    (r"# 08\. Chạy thử Mô hình trên Bộ Dữ liệu Thực tế & Phân tích Lệch miền \(Câu 3: 1\.5 điểm\)", "# Câu 3 (3,0 điểm): Chạy thử mô hình trong thực tế"),
    (r"# VI_01\. Thu thập Dữ liệu, Thống kê Mô tả \(EDA\) & Phân tích SQL Tiếng Việt trên ViHSD[ \t]*", "# Mở rộng Tiếng Việt (ViHSD) — Phần 1: Thu thập, Thống kê mô tả & Phân tích SQL"),
    (r"# VI_02\. Làm sạch Tách từ Tiếng Việt & So sánh Mô hình Học máy Cổ điển[ \t]*", "# Mở rộng Tiếng Việt (ViHSD) — Phần 2: Làm sạch, Tách từ & So sánh các mô hình học máy"),
    (r"# VI_03\. Mô hình Transformer Tiếng Việt: PhoBERT-base-v2 & XLM-RoBERTa-base qua 3 Seeds[ \t]*", "# Mở rộng Tiếng Việt (ViHSD) — Phần 3: Huấn luyện mô hình PhoBERT-base-v2 & XLM-RoBERTa-base"),
    (r"# VI_04\. Đánh giá Cuối trên Tập Test ViHSD, Kiểm tra Độ bền Không dấu/Teencode & Phân tích Lỗi[ \t]*", "# Mở rộng Tiếng Việt (ViHSD) — Phần 4: Đánh giá trên tập Test & Kiểm tra độ bền không dấu / Teencode"),
    (r"# VI_05\. Kiểm thử Thực tế Tiếng Việt \(\$N = 210\$\) & Tích hợp Đa ngôn ngữ `MultiPredictor`[ \t]*", "# Câu 4 (2,0 điểm): Thiết kế ứng dụng sử dụng mô hình được chọn & Chạy thử thực tế đa ngôn ngữ"),
    (r"(?<!\n)\*\*Mục tiêu:\*\*", "\n\n**Mục tiêu:**"),
]


def clean_str(text: str, is_code: bool = False) -> str:
    s = text
    for pattern, repl in TEXT_REPLACEMENTS:
        s = re.sub(pattern, repl, s)
    if not is_code:
        # Clean double spaces before punctuation only in prose/markdown, NEVER in Python code
        s = re.sub(r"  +\)", ")", s)
        s = re.sub(r"\(\s*\)", "", s)
    return s


def _to_lines(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines if lines else [""]


def _clean_cell(cell: dict) -> None:
    src_str = "".join(cell["source"]) if isinstance(cell["source"], list) else str(cell["source"])
    if cell.get("cell_type") == "code":
        if "IN_COLAB" in src_str and "REPO_URL" in src_str:
            cell["source"] = list(BOOTSTRAP_LINES)
        else:
            cell["source"] = _to_lines(clean_str(src_str, is_code=True))
        if "outputs" in cell:
            for out in cell["outputs"]:
                if "text" in out:
                    if isinstance(out["text"], list):
                        out["text"] = [clean_str(t, is_code=True) for t in out["text"]]
                    elif isinstance(out["text"], str):
                        out["text"] = clean_str(out["text"], is_code=True)
    else:
        cell["source"] = _to_lines(clean_str(src_str, is_code=False))


def build_clean_presentation_notebook() -> Path:
    src_nb_path = PROJECT_ROOT / "notebooks" / "MASTER_ALL_IN_ONE_COLAB.ipynb"
    nb = json.loads(src_nb_path.read_text(encoding="utf-8"))

    top_prompt_md = (
        "# 1. Twitter Sentimental Analysis\n\n"
        "**Câu 1 (2,0 điểm): Thu thập và phân tích dữ liệu:**\n"
        "- Phân tích bộ dữ liệu được giao hoặc thu thập dữ liệu dựa trên các API (1đ),\n"
        "- Thống kê mô tả dữ liệu (1đ).\n\n"
        "**Câu 2 (3,0 điểm): Thiết kế mô hình học máy:**\n"
        "- Làm sạch dữ liệu và import dữ liệu lên Google Colab hoặc Python(1đ)\n"
        "- Chuẩn hóa và cân bằng dữ liệu (nếu cần thiết) (0.5đ)\n"
        "- Xây dựng các mô hình dữ liệu, các tập dữ liệu huấn luyện, các tiêu chí đánh giá.(1,5đ)\n\n"
        "**Câu 3 (3,0 điểm): So sánh và lựa chọn mô hình học máy:**\n"
        "- So sánh hiệu quả của các mô hình khác nhau trên bộ dữ liệu được giao (1.5đ)\n"
        "- Điều chỉnh các tham số để tối ưu hóa mô hình (0.5đ)\n"
        "- Chạy thử mô hình trong thực tế.(1.5đ)\n\n"
        "**Câu 4 (2,0 điểm): Thiết kế một ứng dụng sử dụng mô hình được chọn:**\n"
        "- Thiết kế giao diện ứng dụng (1,0đ),\n"
        "- Trình bày báo cáo và trả lời câu hỏi (1,0đ)"
    )

    nb["cells"][0]["source"] = _to_lines(top_prompt_md)

    if len(nb["cells"]) > 1 and nb["cells"][1].get("cell_type") == "markdown":
        nb["cells"][1]["source"] = _to_lines(
            "## Khởi tạo môi trường Google Colab & Import dữ liệu lên Python\n"
            "- Thiết lập môi trường thực thi tự động trên **Google Colab** và **Python**.\n"
            "- Cố định `seed = 42` trên toàn bộ thư viện (`random`, `numpy`, `sklearn`, `torch`) để đảm bảo tính tái lập kết quả."
        )

    for cell in nb["cells"][1:]:
        _clean_cell(cell)

    out_nb_path = PROJECT_ROOT / "notebooks" / "BAO_CAO_DO_AN_TWEET_HATE_CLASSIFICATION.ipynb"
    nb_json = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    out_nb_path.write_text(nb_json, encoding="utf-8")
    src_nb_path.write_text(nb_json, encoding="utf-8")

    for p in sorted(glob.glob(str(PROJECT_ROOT / "notebooks" / "*.ipynb"))):
        if Path(p).name in ("BAO_CAO_DO_AN_TWEET_HATE_CLASSIFICATION.ipynb", "MASTER_ALL_IN_ONE_COLAB.ipynb"):
            continue
        sub_nb = json.loads(Path(p).read_text(encoding="utf-8"))
        if Path(p).name == "00_colab_bootstrap.ipynb" and len(sub_nb["cells"]) > 0 and sub_nb["cells"][0].get("cell_type") == "markdown":
            sub_nb["cells"][0]["source"] = _to_lines(top_prompt_md)
        for c in sub_nb["cells"]:
            _clean_cell(c)
        Path(p).write_text(json.dumps(sub_nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Updated all 16 notebooks cleanly ({out_nb_path.name})")
    return out_nb_path


def clean_docx_and_pptx() -> None:
    # Clean app/app.py using is_code=True so Python syntax () is never altered
    app_fp = PROJECT_ROOT / "app" / "app.py"
    if app_fp.exists():
        app_fp.write_text(clean_str(app_fp.read_text(encoding="utf-8"), is_code=True), encoding="utf-8")
        print("Cleaned app/app.py safely")

    # Clean markdown files
    for md_rel in [
        "README.md",
        "reports/speaker_script.md",
        "reports/demo_script.md",
        "reports/cheat_sheet.md",
        "reports/qa_bank.md",
    ]:
        fp = PROJECT_ROOT / md_rel
        if fp.exists():
            fp.write_text(clean_str(fp.read_text(encoding="utf-8"), is_code=False), encoding="utf-8")
            print(f"Cleaned {md_rel}")


if __name__ == "__main__":
    build_clean_presentation_notebook()
    clean_docx_and_pptx()
