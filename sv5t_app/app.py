# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
from joblib import load

# ✅ st.set_page_config MUST be the first Streamlit command
st.set_page_config(page_title="SV5T Insight & Coach", layout="wide")

from src.activities import (  # noqa: E402
    load_or_create_activities,
    CRITERIA,
    CRITERIA_LABEL,
    criteria_from_activities,
    recommend_activities,
)

from src.recommend import (  # noqa: E402
    pdf_rule_checklist,
    evidence_by_missing,
    macro_solutions,
)


# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("data/sv5t.csv")


@st.cache_data
def load_activities() -> pd.DataFrame:
    return load_or_create_activities("data/activities.csv")


@st.cache_resource
def load_model_bundle():
    return load("model/sv5t_model.joblib")


# -----------------------------
# Prediction (safe)
# -----------------------------
def _predict_prob(df: pd.DataFrame, bundle, x_row: dict) -> float:
    """
    Tương thích cả model mới dạng dict và model cũ dạng Pipeline.
    - Nếu bundle là dict: cần có keys "model" và "numeric_cols"
    - Nếu bundle là Pipeline: tự suy ra numeric_cols từ df (numeric, bỏ sv5t)
    - Luôn fill thiếu cột bằng 0, drop thừa cột, và giữ đúng thứ tự cột model cần
    """
    b = bundle
    if isinstance(b, dict) and "model" in b and "numeric_cols" in b:
        model = b["model"]
        numeric_cols = list(b["numeric_cols"])
    else:
        model = b
        numeric_cols = (
            df.select_dtypes(include=[np.number])
            .drop(columns=["sv5t"], errors="ignore")
            .columns.tolist()
        )

    X_in = pd.DataFrame([x_row])

    # fill thiếu cột
    for col in numeric_cols:
        if col not in X_in.columns:
            X_in[col] = 0

    # chỉ giữ đúng cột model cần + đúng thứ tự
    X_in = X_in[numeric_cols]

    prob = float(model.predict_proba(X_in)[0, 1])
    return prob


# -----------------------------
# App UI
# -----------------------------
def main():
    st.title("SV5T Insight & Coach")
    st.caption("Phân tích – dự đoán – gợi ý hành động & hoạt động thực tế để nâng tỷ lệ đạt SV5T (demo).")

    # Version marker (để bạn biết chắc đang chạy file mới)
    with st.expander("🔧 Debug / Version", expanded=False):
        st.info("🔥 APP VERSION – 2025-12-23 – ACTIVITY MAPPING")
        st.write("RUNNING FILE:", __file__)

    # Check required files
    if not (os.path.exists("data/sv5t.csv") and os.path.exists("model/sv5t_model.joblib")):
        st.warning("Chưa có dữ liệu/mô hình. Hãy chạy: `python -m src.train` để tạo data, activities và train model.")
        st.stop()

    # Load resources
    df = load_data()
    acts = load_activities()
    bundle = load_model_bundle()

    tab1, tab2, tab3 = st.tabs(
        ["🎓 Cá nhân hoá cho Sinh viên", "📅 Hoạt động & Tiêu chí SV5T", "🧑‍💼 Dashboard Đoàn–Hội"]
    )

    # -------------------------
    # TAB 1: STUDENT
    # -------------------------
    with tab1:
        st.subheader("1) Nhập hồ sơ → dự đoán → checklist tiêu chí (bám PDF)")

        with st.sidebar:
            st.header("Hồ sơ SV (demo)")

            school_level = st.selectbox(
                "Bậc học",
                ["daihoc_hocvien", "caodang"],
                format_func=lambda v: "Đại học/Học viện" if v == "daihoc_hocvien" else "Cao đẳng",
            )
            grading_scale = st.selectbox("Hệ điểm", [4, 10])

            gpa = st.number_input(
                "GPA",
                min_value=0.0,
                max_value=float(grading_scale),
                value=3.4 if grading_scale == 4 else 8.5,
                step=0.01,
            )

            st.markdown("### Đạo đức")
            ren_luyen = st.slider("Điểm rèn luyện (0–100)", 50, 100, 90, 1)
            no_violation = st.selectbox("Không vi phạm", [1, 0], format_func=lambda v: "Có" if v == 1 else "Không")
            marx_team_member = st.selectbox(
                "Tham gia đội thi Mác–Lênin/HCM", [0, 1], format_func=lambda v: "Có" if v == 1 else "Không"
            )
            good_deed_awarded = st.selectbox(
                "Được biểu dương 'người tốt việc tốt' (huyện/tỉnh+)",
                [0, 1],
                format_func=lambda v: "Có" if v == 1 else "Không",
            )

            st.markdown("### Học tập (chỉ cần 1 tiêu chí học thuật)")
            nckh_good = st.checkbox("NCKH SV loại tốt")
            thesis_award = st.checkbox("Luận văn/đồ án đạt giải")
            journal_paper = st.checkbox("Có bài báo tạp chí")
            conference_proceeding = st.checkbox("Có kỷ yếu/ tham luận hội thảo")
            patent_or_creative_award = st.checkbox("Sáng chế / giải sáng tạo (tỉnh+)")
            academic_team_member = st.checkbox("Đội tuyển thi học thuật (QG/QT)")
            innovation_award = st.checkbox("Giải ý tưởng sáng tạo (tỉnh+)")

            st.markdown("### Thể lực (đạt 1/2)")
            sv_khoe_provincial_or_higher = st.checkbox("Đạt 'SV Khỏe' (từ tỉnh) / thể thao cấp TW")
            sport_award_school_or_higher = st.checkbox("Giải thể thao phong trào cấp trường+")

            st.markdown("### Tình nguyện")
            volunteer_days = st.slider("Ngày tình nguyện/năm", 0, 20, 5, 1)
            volunteer_award_prov_or_district = st.selectbox(
                "Có khen thưởng tình nguyện (huyện/tỉnh+)",
                [0, 1],
                format_func=lambda v: "Có" if v == 1 else "Không",
            )

            st.markdown("### Hội nhập")
            skill_course_or_youth_union_award = st.selectbox(
                "Có khóa kỹ năng / khen thưởng Hội–Đoàn cấp trường+",
                [0, 1],
                format_func=lambda v: "Có" if v == 1 else "Không",
            )
            integration_activity_count = st.slider("Số hoạt động hội nhập cấp trường+", 0, 10, 1, 1)
            english_b1_or_equivalent = st.selectbox(
                "Có chứng chỉ B1 (hoặc tương đương)",
                [0, 1],
                format_func=lambda v: "Có" if v == 1 else "Không",
            )
            foreign_language_gpa = st.number_input(
                "Điểm tích luỹ ngoại ngữ (nếu không có chứng chỉ)",
                min_value=0.0,
                max_value=float(grading_scale),
                value=3.2 if grading_scale == 4 else 8.0,
                step=0.01,
            )
            international_exchange = st.selectbox(
                "Có giao lưu quốc tế", [0, 1], format_func=lambda v: "Có" if v == 1 else "Không"
            )
            integration_competition_award = st.selectbox(
                "Đạt giải thi hội nhập/ngoại ngữ",
                [0, 1],
                format_func=lambda v: "Có" if v == 1 else "Không",
            )

            st.markdown("---")
            st.markdown("### 2) Hoạt động bạn đã tham gia (để tự tính đạt nhóm nào)")
            act_options = (acts["activity_id"] + " — " + acts["name"]).tolist()
            selected = st.multiselect("Chọn hoạt động đã tham gia/đạt", options=act_options, default=[])

        selected_ids = [s.split(" — ")[0].strip() for s in selected]

        # mapping hoạt động -> nhóm tiêu chí
        hit = criteria_from_activities(acts, selected_ids)
        missing = [c for c in CRITERIA if not hit.get(c, False)]

        # build row x
        x = {
            "school_level": school_level,
            "grading_scale": int(grading_scale),
            "gpa": float(gpa),
            "ren_luyen": int(ren_luyen),
            "no_violation": int(no_violation),
            "marx_team_member": int(marx_team_member),
            "good_deed_awarded": int(good_deed_awarded),
            "nckh_good": int(nckh_good),
            "thesis_award": int(thesis_award),
            "journal_paper": int(journal_paper),
            "conference_proceeding": int(conference_proceeding),
            "patent_or_creative_award": int(patent_or_creative_award),
            "academic_team_member": int(academic_team_member),
            "innovation_award": int(innovation_award),
            "sv_khoe_provincial_or_higher": int(sv_khoe_provincial_or_higher),
            "sport_award_school_or_higher": int(sport_award_school_or_higher),
            "volunteer_days": int(volunteer_days),
            "volunteer_award_prov_or_district": int(volunteer_award_prov_or_district),
            "skill_course_or_youth_union_award": int(skill_course_or_youth_union_award),
            "integration_activity_count": int(integration_activity_count),
            "english_b1_or_equivalent": int(english_b1_or_equivalent),
            "foreign_language_gpa": float(foreign_language_gpa),
            "international_exchange": int(international_exchange),
            "integration_competition_award": int(integration_competition_award),
            # proxy features từ hoạt động
            "act_dao_duc": int(hit.get("dao_duc", False)),
            "act_hoc_tap": int(hit.get("hoc_tap", False)),
            "act_the_luc": int(hit.get("the_luc", False)),
            "act_tinh_nguyen": int(hit.get("tinh_nguyen", False)),
            "act_hoi_nhap": int(hit.get("hoi_nhap", False)),
        }

        prob = _predict_prob(df, bundle, x)
        pred = int(prob >= 0.5)

        c1, c2, c3 = st.columns(3)
        c1.metric("Xác suất đạt (ML demo)", f"{prob*100:.1f}%")
        c2.metric("Dự đoán", "Có khả năng đạt" if pred == 1 else "Nguy cơ chưa đạt")
        c3.metric("Thiếu theo mapping hoạt động", f"{len(missing)}/5 nhóm")

        st.divider()
        st.markdown("### A) Kết quả theo hoạt động bạn đã tham gia")
        cols = st.columns(5)
        for i, c in enumerate(CRITERIA):
            _ = cols[i].success(CRITERIA_LABEL[c]) if hit.get(c, False) else cols[i].warning(CRITERIA_LABEL[c])


        st.markdown("### B) Checklist tiêu chí (bám PDF, dạng ngắn)")
        for t in pdf_rule_checklist(x):
            st.write("✅ " + str(t))

        st.markdown("### C) Minh chứng cần chuẩn bị (theo nhóm còn thiếu từ hoạt động)")
        if missing:
            for line in evidence_by_missing(missing):
                st.write(line)
        else:
            st.info("Bạn đã có hoạt động phủ đủ 5 nhóm (theo mapping). Hãy kiểm tra thêm ngưỡng điểm/khen thưởng theo PDF.")

        st.divider()
        st.markdown("### D) Gợi ý hoạt động nên tham gia (cá nhân hoá)")
        rec_df = recommend_activities(acts, missing, topk=8)
        st.dataframe(
            rec_df[["activity_id", "name", "organizer", "date", "criteria", "evidence"]],
            use_container_width=True,
        )

        st.divider()
        st.markdown("### E) What-if: nếu bạn tham gia thêm hoạt động → xác suất thay đổi?")
        st.caption("Chọn 1 hoạt động bất kỳ để mô phỏng tác động (demo).")

        pick = st.selectbox(
            "Chọn 1 hoạt động để thử",
            ["(không chọn)"] + (acts["activity_id"] + " — " + acts["name"]).tolist(),
        )
        if pick != "(không chọn)":
            pid = pick.split(" — ")[0].strip()
            hit2 = dict(hit)
            hit_add = criteria_from_activities(acts, [pid])
            for k in hit2:
                hit2[k] = bool(hit2.get(k, False) or hit_add.get(k, False))

            x2 = dict(x)
            x2["act_dao_duc"] = int(hit2.get("dao_duc", False))
            x2["act_hoc_tap"] = int(hit2.get("hoc_tap", False))
            x2["act_the_luc"] = int(hit2.get("the_luc", False))
            x2["act_tinh_nguyen"] = int(hit2.get("tinh_nguyen", False))
            x2["act_hoi_nhap"] = int(hit2.get("hoi_nhap", False))

            prob2 = _predict_prob(df, bundle, x2)
            st.write(f"Trước: **{prob*100:.1f}%** → Sau: **{prob2*100:.1f}%** (Δ = {(prob2-prob)*100:+.1f}%)")

    # -------------------------
    # TAB 2: ACTIVITIES
    # -------------------------
    with tab2:
        st.subheader("Danh mục hoạt động hiện tại & tiêu chí SV5T đạt được")
        st.caption("Bạn có thể thay `data/activities.csv` bằng dữ liệu thật của Đoàn–Hội.")

        left, right = st.columns([2, 1])

        with right:
            st.markdown("### Lọc nhanh")
            crit = st.multiselect(
                "Theo nhóm tiêu chí",
                options=CRITERIA,
                default=[],
                format_func=lambda c: CRITERIA_LABEL.get(c, c),
            )
            org = st.multiselect(
                "Đơn vị tổ chức",
                options=sorted(acts["organizer"].dropna().unique().tolist()),
                default=[],
            )

        view = acts.copy()
        if crit:
            view = view[view["criteria"].apply(lambda s: any(c in str(s).split(",") for c in crit))]
        if org:
            view = view[view["organizer"].isin(org)]

        with left:
            st.dataframe(
                view[["activity_id", "name", "organizer", "date", "criteria", "evidence"]],
                use_container_width=True,
            )

        st.divider()
        st.markdown("### Bản đồ hoạt động → tiêu chí")
        counts = []
        for c in CRITERIA:
            counts.append(
                dict(criteria=CRITERIA_LABEL[c], count=int(acts["criteria"].astype(str).str.contains(c).sum()))
            )
        fig = px.bar(pd.DataFrame(counts), x="criteria", y="count")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # TAB 3: DASHBOARD
    # -------------------------
    with tab3:
        st.subheader("Dashboard Đoàn–Hội (demo)")

        rate = float(df["sv5t"].mean())
        c1, c2, c3 = st.columns(3)
        c1.metric("Tỷ lệ đạt SV5T (demo)", f"{rate*100:.1f}%")
        c2.metric("Số mẫu", f"{len(df)}")
        c3.metric("Số hoạt động (demo)", f"{len(acts)}")

        st.divider()
        fig = px.histogram(df, x="sv5t", text_auto=True, title="Phân bố đạt/không đạt (0/1)")
        st.plotly_chart(fig, use_container_width=True)

        fail = df[df["sv5t"] == 0].copy()
        if fail.empty:
            st.info("Không có mẫu không đạt trong demo.")
            return

        missing_share = {
            "dao_duc": float(((fail["ren_luyen"] < 90) | (fail["no_violation"] == 0)).mean()),
            "hoc_tap": float((fail["nckh_good"] == 0).mean()),  # heuristic demo
            "the_luc": float((~((fail["sv_khoe_provincial_or_higher"] == 1) | (fail["sport_award_school_or_higher"] == 1))).mean()),
            "tinh_nguyen": float((~((fail["volunteer_days"] >= 5) & (fail["volunteer_award_prov_or_district"] == 1))).mean()),
            "hoi_nhap": float((~((fail["integration_activity_count"] >= 1))).mean()),
        }

        bars = (
            pd.DataFrame(
                {
                    "Nhóm tiêu chí": [CRITERIA_LABEL[k] for k in missing_share.keys()],
                    "Tỷ lệ thiếu (demo)": list(missing_share.values()),
                }
            )
            .sort_values("Tỷ lệ thiếu (demo)", ascending=False)
        )

        fig2 = px.bar(bars, x="Tỷ lệ thiếu (demo)", y="Nhóm tiêu chí", orientation="h", text_auto=".0%")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Gợi ý giải pháp theo dữ liệu")
        for s in macro_solutions(missing_share):
            st.write("✅ " + str(s))

        st.divider()
        st.download_button(
            "Download activities.csv",
            data=acts.to_csv(index=False).encode("utf-8"),
            file_name="activities.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
