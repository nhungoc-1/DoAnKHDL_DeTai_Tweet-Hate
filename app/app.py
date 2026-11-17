"""Multilingual (English & Vietnamese) Streamlit Web Application for Hate Speech Classification (P10 & V7)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.predict import MultiPredictor
from src.utils import mask_profanity
from src.vi_clean import vi_mask_profanity

st.set_page_config(
    page_title="Hệ thống Phát hiện Tweet & Bình luận Phản cảm Đa ngôn ngữ (EN / VI)",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def get_cached_multi_predictor() -> MultiPredictor:
    """Load and cache both English (best_model.joblib) and Vietnamese (best_vi_bin_a.joblib) models."""
    return MultiPredictor()


EXAMPLE_SAMPLES = [
    (
        "1. [EN] Bình thường (Đời sống & Công nghệ)",
        "Just finished reading an amazing book on machine learning and natural language processing with my friends!",
        "Tự động",
    ),
    (
        "2. [EN] Phản cảm — Racist/Sexist (Đã che)",
        "Women do not belong in leadership or engineering roles because they are too emotional and inferior to men.",
        "Tự động",
    ),
    (
        "3. [VI có dấu] Bình thường (Khen ngợi & Thể thao)",
        "Đội tuyển bóng đá Việt Nam hôm qua thi đấu rất nỗ lực và đoàn kết, chúc mừng các cầu thủ trẻ!",
        "Tự động",
    ),
    (
        "4. [VI có dấu] Phản cảm — Thù ghét/Xúc phạm (Đã che)",
        "Đúng là lũ súc vật óc chó vô học, sống chỉ làm rác rưởi bẩn xã hội chứ không được tích sự gì!",
        "Tự động",
    ),
    (
        "5. [VI không dấu] Cần chọn tay hoặc thử nhận diện",
        "thang oc cho nay ngu dot va mat day kinh khung cut khoi day ngay",
        "Tự động",
    ),
    (
        "6. [Quá ngắn / Không rõ ngôn ngữ] Kiểm tra cảnh báo",
        "ok bro",
        "Tự động",
    ),
]


def main() -> None:
    st.title("🛡️ Hệ thống Phân loại Văn bản Phản cảm Đa ngôn ngữ (English & Tiếng Việt)")
    st.markdown(
        "**Đề tài:** Gắn nhãn các đoạn tweet có nội dung phân biệt chủng tộc, phân biệt giới tính (*English*) "
        "& Mở rộng phát hiện ngôn từ thù ghét / xúc phạm Mạng xã hội (*Tiếng Việt — ViHSD*)"
    )

    st.info(
        "ℹ️ **Lưu ý định nghĩa nhãn & nhận diện ngôn ngữ:**\n"
        "- **English Model (`models/best_model.joblib`):** Nhãn `1` = *Racist / Sexist Hate Speech* (Phân biệt chủng tộc / giới tính).\n"
        "- **Vietnamese Model (`models/vi/best_vi_bin_a.joblib`):** Nhãn `1` = *Ngôn từ thù ghét / xúc phạm cá nhân hoặc nhóm* (theo chuẩn bộ dữ liệu **ViHSD — Luu et al., 2021**).\n"
        "- ⚠️ *Tiếng Việt không dấu hoặc văn bản quá ngắn có thể bị nhận diện sai ngôn ngữ và dự đoán kém chính xác — khuyến nghị chọn ngôn ngữ thủ công trong trường hợp này.*"
    )

    try:
        mp = get_cached_multi_predictor()
    except Exception as e:
        st.error(f"Không thể nạp mô hình: {e}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(
        [
            "🔍 Tab 1: Phân loại 1 Câu (EN / VI)",
            "📂 Tab 2: Phân loại Hàng loạt (CSV Đa ngôn ngữ)",
            "📊 Tab 3: Thống kê & Hồ sơ 2 Mô hình (EN & VI)",
        ]
    )

    # =========================================================================
    # TAB 1: PHÂN LOẠI 1 CÂU (ĐA NGÔN NGỮ)
    # =========================================================================
    with tab1:
        st.subheader("Phân tích & Giải thích từng đoạn Tweet / Bình luận (Tự động nhận diện EN / VI)")
        st.caption("Bấm chọn nhanh 1 trong 6 câu mẫu kiểm thử (2 câu Anh, 2 câu Việt có dấu, 1 câu Việt không dấu, 1 câu quá ngắn):")

        if "input_text" not in st.session_state:
            st.session_state["input_text"] = EXAMPLE_SAMPLES[0][1]
        if "lang_radio" not in st.session_state:
            st.session_state["lang_radio"] = "Tự động"

        cols_ex = st.columns(3)
        for idx, (btn_label, sample_txt, sample_lang) in enumerate(EXAMPLE_SAMPLES):
            if cols_ex[idx % 3].button(btn_label, use_container_width=True):
                st.session_state["input_text"] = sample_txt
                st.session_state["lang_radio"] = sample_lang

        lang_choice = st.radio(
            "Ngôn ngữ (Chế độ chọn tay luôn ghi đè nhận diện tự động):",
            options=["Tự động", "English", "Tiếng Việt"],
            horizontal=True,
            key="lang_radio",
        )

        raw_input = st.text_area(
            "Nội dung Tweet / Bình luận cần kiểm tra (Tối đa 1,000 ký tự):",
            value=st.session_state["input_text"],
            height=110,
            max_chars=1000,
        )

        analyze_clicked = st.button("🚀 Phân tích Nội dung", type="primary")
        if analyze_clicked or raw_input.strip():
            res = mp.predict(raw_input, lang=lang_choice)
            if res["status"] != "ok":
                st.warning(f"⚠️ **CẢNH BÁO KHÔNG DỰ ĐOÁN:** {res.get('warning')}")
            else:
                score = float(res["score"])
                thr = float(res["threshold"])
                label = int(res["label"])
                lang_det = res["lang"]
                mask_fn = vi_mask_profanity if lang_det == "vi" else mask_profanity

                col_res1, col_res2 = st.columns([1.15, 1.0])
                with col_res1:
                    st.markdown(
                        f"**Ngôn ngữ xử lý:** `{lang_det.upper()}` | **Mô hình đã dùng:** `{res['model_used']}`  \n"
                        f"**Định nghĩa nhãn:** *{res['definition']}*"
                    )
                    if label == 1:
                        pos_title = (
                            "PHẢN CẢM — THÙ GHÉT / XÚC PHẠM (Nhãn = 1)"
                            if lang_det == "vi"
                            else "PHẢN CẢM — PHÂN BIỆT CHỦNG TỘC / GIỚI TÍNH (Nhãn = 1)"
                        )
                        st.error(
                            f"🚨 **KẾT QUẢ: {pos_title}**\n\n"
                            f"- **Điểm xác suất mô hình:** `{score:.4f}` (Ngưỡng quyết định: `{thr:.4f}`)\n"
                            f"- **Đánh giá:** Điểm vượt ngưỡng `{thr:.4f}` (+`{score - thr:.4f}`)."
                        )
                    else:
                        st.success(
                            f"✅ **KẾT QUẢ: BÌNH THƯỜNG / SẠCH (Nhãn = 0)**\n\n"
                            f"- **Điểm xác suất mô hình:** `{score:.4f}` (Ngưỡng quyết định: `{thr:.4f}`)\n"
                            f"- **Đánh giá:** Điểm thấp hơn ngưỡng `{thr:.4f}` (`{score - thr:.4f}`)."
                        )

                    st.markdown(f"**Thanh mức độ rủi ro so với ngưỡng quyết định (`{thr:.3f}`):**")
                    st.progress(min(max(score, 0.0), 1.0))
                    st.markdown(f"**Văn bản sau làm sạch / tách từ:** `{mask_fn(res['clean_text'])}`")

                with col_res2:
                    st.markdown("#### 🔎 Giải thích đóng góp đặc trưng (Top-8 Terms: `TF-IDF × Hệ số`)")
                    top_terms = res.get("top_terms", [])
                    if not top_terms:
                        st.info("Không có từ khóa đặc trưng nào nằm trong từ điển hoặc mô hình không hỗ trợ.")
                    else:
                        for term, weight in top_terms:
                            if weight > 0:
                                st.markdown(f"- :red[**+{term}** (`+{weight:.3f}` ↑ Phản cảm / Thù ghét)]")
                            else:
                                st.markdown(f"- :green[**{term}** (`{weight:.3f}` ↓ Bình thường)]")

    # =========================================================================
    # TAB 2: PHÂN LOẠI HÀNG LOẠT (CSV ĐA NGÔN NGỮ)
    # =========================================================================
    with tab2:
        st.subheader("Phân loại Hàng loạt từ File CSV Đa ngôn ngữ (Tối đa 20,000 dòng)")
        st.markdown(
            "Tải lên file `.csv` có cột `text` (hoặc `tweet` / `free_text`). "
            "Hệ thống tự động nhận diện ngôn ngữ (`lang`), ghi rõ mô hình sử dụng (`model_used`), "
            "và đánh dấu `NA` đối với các dòng không xác định được ngôn ngữ (`unknown`)."
        )

        batch_lang_mode = st.radio(
            "Chế độ ngôn ngữ cho File CSV:",
            options=["Tự động", "English", "Tiếng Việt"],
            horizontal=True,
            key="batch_lang_mode",
        )
        uploaded_file = st.file_uploader("Chọn file CSV:", type=["csv"])
        col_b1, col_b2 = st.columns(2)
        use_sample_en = col_b1.button("📋 Chạy thử 100 mẫu Thực tế Tiếng Anh (`real_world_labeled.csv`)")
        use_sample_vi = col_b2.button("🇻🇳 Chạy thử 100 mẫu Thực tế Tiếng Việt (`vi_real_world_labeled.csv`)")

        df_in = None
        if uploaded_file is not None:
            df_in = pd.read_csv(uploaded_file)
        elif use_sample_en:
            p_en = PROJECT_ROOT / "data" / "real_world" / "real_world_labeled.csv"
            if p_en.exists():
                df_in = pd.read_csv(p_en).head(100)
        elif use_sample_vi:
            p_vi = PROJECT_ROOT / "data" / "real_world" / "vi_real_world_labeled.csv"
            if p_vi.exists():
                df_in = pd.read_csv(p_vi).head(100)

        if df_in is not None:
            text_col = None
            for cand in ["text", "tweet", "free_text", "clean_text", "clean_comment"]:
                if cand in df_in.columns:
                    text_col = cand
                    break

            if text_col is None:
                st.error(f"Không tìm thấy cột văn bản trong file! Các cột hiện có: {list(df_in.columns)}")
            else:
                if len(df_in) > 20000:
                    st.warning("File vượt quá 20,000 dòng. Hệ thống tự động cắt 20,000 dòng đầu tiên.")
                    df_in = df_in.head(20000).copy()

                prog = st.progress(0, text="Đang phân loại hàng loạt...")
                texts = df_in[text_col].fillna("").astype(str).tolist()
                batch_out = mp.predict_batch(texts, lang=batch_lang_mode)
                prog.progress(100, text="Hoàn tất phân loại!")

                batch_display = batch_out.copy()
                batch_display["text"] = [
                    vi_mask_profanity(t) if l == "vi" else mask_profanity(t)
                    for t, l in zip(batch_display["text"], batch_display["lang"])
                ]
                batch_display["text_clean"] = [
                    vi_mask_profanity(t) if l == "vi" else mask_profanity(t)
                    for t, l in zip(batch_display["text_clean"], batch_display["lang"])
                ]

                n_total = len(batch_display)
                n_pos = int((batch_display["label"] == 1).sum())
                n_neg = int((batch_display["label"] == 0).sum())
                n_na = int((batch_display["label"] == "NA").sum())

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Tổng số dòng", f"{n_total:,}")
                m2.metric("Bình thường (Nhãn 0)", f"{n_neg:,}")
                m3.metric("Phản cảm / Thù ghét (Nhãn 1)", f"{n_pos:,}")
                m4.metric("Không rõ ngôn ngữ (NA)", f"{n_na:,}")

                st.dataframe(batch_display, use_container_width=True, height=340)

                csv_bytes = batch_out.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ Tải xuống kết quả phân loại (.csv)",
                    data=csv_bytes,
                    file_name="ket_qua_phan_loai_da_ngon_ngu.csv",
                    mime="text/csv",
                )

    # =========================================================================
    # TAB 3: THỐNG KÊ & HỒ SƠ 2 MÔ HÌNH (KHÔNG TRỘN SỐ)
    # =========================================================================
    with tab3:
        st.subheader("Hồ sơ & Kết quả Thực nghiệm Độc lập cho 2 Mô hình (English vs Tiếng Việt)")

        col_en, col_vi = st.columns(2)

        with col_en:
            st.markdown("### 🇬🇧 Mô hình 1: Tiếng Anh (Racist / Sexist Tweet Detection)")
            mc_en_path = PROJECT_ROOT / "models" / "model_card.json"
            ft_en_path = PROJECT_ROOT / "reports" / "final_test_metrics.json"
            if mc_en_path.exists() and ft_en_path.exists():
                mc_en = json.loads(mc_en_path.read_text(encoding="utf-8"))
                ft_en = json.loads(ft_en_path.read_text(encoding="utf-8"))
                tm_en = ft_en["metrics_at_tuned_threshold"]
                e1, e2, e3 = st.columns(3)
                e1.metric("EN Test F1 (Lớp 1)", f"{tm_en['f1_pos']:.4f}")
                e2.metric("EN Test PR-AUC", f"{tm_en['pr_auc']:.4f}")
                e3.metric("EN Threshold", f"{mc_en['best_threshold']:.3f}")
                st.write(f"- **Mô hình chốt:** `{mc_en['model_name']}` | **ROC-AUC:** `{tm_en['roc_auc']:.4f}`")
                st.write(f"- **Định nghĩa nhãn:** Phân biệt chủng tộc / giới tính (`0/1`).")

            comp_en_path = PROJECT_ROOT / "reports" / "model_comparison.csv"
            if comp_en_path.exists():
                st.markdown("**Bảng so sánh các mô hình Tiếng Anh (Validation):**")
                st.dataframe(pd.read_csv(comp_en_path), use_container_width=True, height=260)

            fig_en_test = PROJECT_ROOT / "reports" / "figures" / "P8_test_evaluation.png"
            if fig_en_test.exists():
                st.image(str(fig_en_test), caption="Kết quả Đánh giá Tập Test Tiếng Anh ")

        with col_vi:
            st.markdown("### 🇻🇳 Mô hình 2: Tiếng Việt (ViHSD Hate / Offensive Speech)")
            mc_vi_path = PROJECT_ROOT / "models" / "vi" / "vi_model_card.json"
            ft_vi_path = PROJECT_ROOT / "reports" / "vi_final_test_metrics.json"
            if mc_vi_path.exists() and ft_vi_path.exists():
                mc_vi = json.loads(mc_vi_path.read_text(encoding="utf-8"))
                ft_vi = json.loads(ft_vi_path.read_text(encoding="utf-8"))
                tm_vi = ft_vi["bin_A"]["metrics_tuned"]
                v1, v2, v3 = st.columns(3)
                v1.metric("VI Test F1 (bin_A)", f"{tm_vi['f1_pos']:.4f}")
                v2.metric("VI Test PR-AUC", f"{tm_vi['pr_auc']:.4f}")
                v3.metric("VI multi3 Macro-F1", f"{ft_vi['multi3']['f1_macro']:.4f}")
                st.write(
                    f"- **Mô hình chốt:** `{mc_vi['best_bin_A']['model_name']} ({mc_vi['best_bin_A']['vectorizer']})` | "
                    f"**Threshold:** `{ft_vi['bin_A']['threshold']:.3f}`"
                )
                st.write("- **Định nghĩa nhãn:** Ngôn từ thù ghét / xúc phạm cá nhân hoặc nhóm (ViHSD — Luu et al., 2021).")

            comp_vi_path = PROJECT_ROOT / "reports" / "vi_model_comparison.csv"
            if comp_vi_path.exists():
                st.markdown("**Bảng so sánh các mô hình Tiếng Việt (Dev — bin_A, bin_B, multi3):**")
                st.dataframe(pd.read_csv(comp_vi_path), use_container_width=True, height=260)

            fig_vi_test = PROJECT_ROOT / "reports" / "figures" / "V5_test_and_robustness.png"
            if fig_vi_test.exists():
                st.image(str(fig_vi_test), caption="Kết quả Đánh giá Tập Test & Độ bền Không dấu Tiếng Việt ")

    st.divider()
    st.caption(
        "⚠️ **Cảnh báo cố định:** *Tiếng Việt không dấu hoặc quá ngắn có thể bị nhận diện sai ngôn ngữ và dự đoán kém chính xác. "
        "Mô hình có thể sai; chỉ hỗ trợ sàng lọc sơ bộ, cần con người kiểm tra lại.*"
    )


if __name__ == "__main__":
    main()
