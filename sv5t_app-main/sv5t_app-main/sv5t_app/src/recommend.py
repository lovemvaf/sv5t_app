from typing import Dict, List
from src.activities import CRITERIA_LABEL, EVIDENCE_GUIDE


def pdf_rule_checklist(x: Dict) -> List[str]:
    """
    Checklist ngắn gọn bám tiêu chí PDF (để demo).
    """
    tips: List[str] = []

    # Đạo đức
    if int(x["ren_luyen"]) < 90:
        tips.append("Đạo đức: Điểm rèn luyện cần ≥ 90.")
    if int(x["no_violation"]) != 1:
        tips.append("Đạo đức: Cần không vi phạm pháp luật/quy chế.")
    if int(x["marx_team_member"]) != 1 and int(x["good_deed_awarded"]) != 1:
        tips.append("Đạo đức: Cần thêm 1 tiêu chí (đội thi Mác–Lênin/HCM hoặc 'người tốt việc tốt' huyện/tỉnh/Đảng ủy/BGH).")

    # Học tập
    school_level = x["school_level"]
    scale = int(x["grading_scale"])
    gpa = float(x["gpa"])
    gpa_need = 3.4 if (school_level == "daihoc_hocvien" and scale == 4) else \
               8.5 if (school_level == "daihoc_hocvien" and scale == 10) else \
               3.2 if (school_level == "caodang" and scale == 4) else 8.0
    if gpa < gpa_need:
        tips.append(f"Học tập: GPA cần ≥ {gpa_need} theo bậc/hệ điểm.")
    academic_plus = any(int(x[k]) == 1 for k in [
        "nckh_good", "thesis_award", "journal_paper", "conference_proceeding",
        "patent_or_creative_award", "academic_team_member", "innovation_award"
    ])
    if not academic_plus:
        tips.append("Học tập: Cần 1 tiêu chí học thuật (NCKH tốt/bài báo/kỷ yếu/giải QG-QT/sáng tạo tỉnh+).")

    # Thể lực
    if int(x["sv_khoe_provincial_or_higher"]) != 1 and int(x["sport_award_school_or_higher"]) != 1:
        tips.append("Thể lực: Cần đạt 'SV khỏe' (tỉnh+) /thể thao TW hoặc giải thể thao trường+.")

    # Tình nguyện
    if int(x["volunteer_days"]) < 5:
        tips.append("Tình nguyện: Cần ≥ 5 ngày/năm.")
    if int(x["volunteer_award_prov_or_district"]) != 1:
        tips.append("Tình nguyện: Cần khen thưởng tình nguyện cấp huyện/tỉnh+ (theo quy định).")

    # Hội nhập
    if int(x["skill_course_or_youth_union_award"]) != 1:
        tips.append("Hội nhập: Cần hoàn thành khóa kỹ năng hoặc được khen thưởng Hội/Đoàn cấp trường+.")
    if int(x["integration_activity_count"]) < 1:
        tips.append("Hội nhập: Cần tham gia ≥ 1 hoạt động hội nhập cấp trường+.")
    lang_ok = (int(x["english_b1_or_equivalent"]) == 1) or (
        float(x["foreign_language_gpa"]) >= (3.2 if int(x["grading_scale"]) == 4 else 8.0)
    )
    if not lang_ok:
        tips.append("Hội nhập: Cần B1 (hoặc tương đương) hoặc điểm ngoại ngữ tích luỹ đạt ngưỡng.")
    if int(x["international_exchange"]) != 1 and int(x["integration_competition_award"]) != 1:
        tips.append("Hội nhập: Cần thêm 1 tiêu chí (giao lưu quốc tế hoặc giải Ba hội nhập/ngoại ngữ trường+).")

    if not tips:
        tips.append("Bạn đang đáp ứng đầy đủ 5 nhóm tiêu chí theo quy định. Hãy chuẩn bị minh chứng đầy đủ.")

    return tips


def evidence_by_missing(missing: List[str]) -> List[str]:
    out: List[str] = []
    for c in missing:
        out.append(f"**{CRITERIA_LABEL.get(c, c)}**")
        for e in EVIDENCE_GUIDE.get(c, []):
            out.append(f"- {e.what}: {e.note}")
    return out


def macro_solutions(missing_share: Dict[str, float]) -> List[str]:
    """
    Gợi ý giải pháp cấp Đoàn–Hội (demo).
    missing_share: tỷ lệ thiếu theo nhóm trong tập không đạt.
    """
    sol: List[str] = []
    # ngưỡng demo
    if missing_share.get("hoc_tap", 0) >= 0.25:
        sol.append("Học tập: mở chương trình hỗ trợ NCKH/bài báo/kỷ yếu; bồi dưỡng nhóm GPA sát ngưỡng.")
    if missing_share.get("tinh_nguyen", 0) >= 0.25:
        sol.append("Tình nguyện: thiết kế hoạt động đủ 'ngày công' + cơ chế ghi nhận/khen thưởng theo quy định.")
    if missing_share.get("hoi_nhap", 0) >= 0.25:
        sol.append("Hội nhập: mở khóa kỹ năng + workshop ngoại ngữ; tăng hoạt động giao lưu/thi hội nhập.")
    if missing_share.get("the_luc", 0) >= 0.25:
        sol.append("Thể lực: tổ chức giải phong trào cấp trường; hỗ trợ xét 'Sinh viên khỏe' theo chuẩn.")
    if missing_share.get("dao_duc", 0) >= 0.25:
        sol.append("Đạo đức: chuẩn hoá theo dõi rèn luyện; tăng sân chơi đội thi Mác–Lênin/HCM và phong trào 'người tốt việc tốt'.")
    if not sol:
        sol.append("Duy trì dashboard theo dõi + can thiệp sớm nhóm 'tiềm năng' để nâng tỷ lệ đạt.")
    return sol
