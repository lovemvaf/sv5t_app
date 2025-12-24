from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd


# 5 nhóm tiêu chí
CRITERIA = ["dao_duc", "hoc_tap", "the_luc", "tinh_nguyen", "hoi_nhap"]

CRITERIA_LABEL = {
    "dao_duc": "Đạo đức tốt",
    "hoc_tap": "Học tập tốt",
    "the_luc": "Thể lực tốt",
    "tinh_nguyen": "Tình nguyện tốt",
    "hoi_nhap": "Hội nhập tốt",
}


@dataclass
class Evidence:
    criteria: str
    what: str
    note: str


EVIDENCE_GUIDE: Dict[str, List[Evidence]] = {
    "dao_duc": [
        Evidence("dao_duc", "Điểm rèn luyện", "≥ 90; kèm bảng điểm rèn luyện/ xác nhận nhà trường."),
        Evidence("dao_duc", "Không vi phạm", "Không vi phạm pháp luật/quy chế/nội quy."),
        Evidence("dao_duc", "Tiêu chí bổ sung", "Đội thi Mác–Lênin/HCM hoặc 'người tốt việc tốt' (huyện/tỉnh hoặc Đảng ủy/BGH trường)."),
    ],
    "hoc_tap": [
        Evidence("hoc_tap", "GPA", "ĐH/HV: ≥3.4/4 hoặc ≥8.5/10; CĐ: ≥3.2/4 hoặc ≥8.0/10."),
        Evidence("hoc_tap", "Tiêu chí học thuật", "Chỉ cần 1: NCKH loại tốt / bài báo / kỷ yếu / giải QG-QT / đội tuyển QG-QT / sáng tạo tỉnh+"),
    ],
    "the_luc": [
        Evidence("the_luc", "SV Khỏe / Giải thể thao", "Đạt 'Sinh viên khỏe' (từ tỉnh) / t.thao TW hoặc giải thể thao trường+."),
    ],
    "tinh_nguyen": [
        Evidence("tinh_nguyen", "Ngày tình nguyện", "≥ 5 ngày/năm."),
        Evidence("tinh_nguyen", "Khen thưởng tình nguyện", "Khen thưởng cấp huyện/tỉnh+ (theo quy định)."),
    ],
    "hoi_nhap": [
        Evidence("hoi_nhap", "Kỹ năng / Khen thưởng Đoàn–Hội", "Hoàn thành ≥1 khóa kỹ năng hoặc được khen thưởng từ cấp trường."),
        Evidence("hoi_nhap", "Hoạt động hội nhập", "Tham gia ≥1 hoạt động hội nhập cấp trường+."),
        Evidence("hoi_nhap", "Ngoại ngữ", "B1 (hoặc tương đương) hoặc điểm tích luỹ ngoại ngữ đạt ngưỡng."),
        Evidence("hoi_nhap", "Tiêu chí bổ sung", "Giao lưu quốc tế hoặc đạt giải Ba hội nhập/thi ngoại ngữ trường+."),
    ],
}


def demo_activities() -> pd.DataFrame:
    """
    Dataset hoạt động demo (bạn có thể thay bằng data thật từ Đoàn/Hội).
    """
    rows = [
        # tình nguyện
        dict(activity_id="ACT001", name="Xuân tình nguyện", organizer="Đoàn trường", date="2025-12-28",
             type="tinh_nguyen", points=3, criteria=["tinh_nguyen"], evidence="Giấy xác nhận ngày công + tổng kết"),
        dict(activity_id="ACT002", name="Mùa hè xanh", organizer="Hội SV", date="2026-06-15",
             type="tinh_nguyen", points=5, criteria=["tinh_nguyen"], evidence="Xác nhận ngày công + khen thưởng (nếu có)"),

        # hội nhập
        dict(activity_id="ACT101", name="Workshop kỹ năng công dân toàn cầu", organizer="CLB Kỹ năng", date="2025-12-26",
             type="hoi_nhap", points=2, criteria=["hoi_nhap"], evidence="Chứng nhận hoàn thành khóa"),
        dict(activity_id="ACT102", name="Giao lưu sinh viên quốc tế", organizer="Phòng HTQT", date="2026-01-10",
             type="hoi_nhap", points=3, criteria=["hoi_nhap"], evidence="Giấy xác nhận tham gia giao lưu"),
        dict(activity_id="ACT103", name="Cuộc thi thuyết trình tiếng Anh", organizer="Khoa", date="2026-01-05",
             type="hoi_nhap", points=3, criteria=["hoi_nhap"], evidence="Giấy chứng nhận/giải thưởng (nếu có)"),

        # thể lực
        dict(activity_id="ACT201", name="Giải chạy Sinh viên khỏe", organizer="Trung tâm TDTT", date="2026-01-03",
             type="the_luc", points=2, criteria=["the_luc"], evidence="Giấy xác nhận/giải phong trào"),

        # học tập
        dict(activity_id="ACT301", name="Cuộc thi NCKH sinh viên", organizer="Phòng KH&CN", date="2026-03-20",
             type="hoc_tap", points=5, criteria=["hoc_tap"], evidence="Quyết định/biên bản nghiệm thu/giải thưởng"),
        dict(activity_id="ACT302", name="Seminar viết bài báo & kỷ yếu", organizer="Khoa", date="2026-02-15",
             type="hoc_tap", points=3, criteria=["hoc_tap"], evidence="Minh chứng bài báo/kỷ yếu"),

        # đạo đức (thường là tiêu chí theo rèn luyện + bổ sung)
        dict(activity_id="ACT401", name="Đội thi Olympic Mác–Lênin/HCM", organizer="Đoàn khoa", date="2026-01-18",
             type="dao_duc", points=3, criteria=["dao_duc"], evidence="Danh sách đội/giấy chứng nhận tham gia"),
        dict(activity_id="ACT402", name="Phong trào 'Người tốt việc tốt'", organizer="Đoàn trường", date="2026-04-01",
             type="dao_duc", points=2, criteria=["dao_duc"], evidence="Quyết định/giấy khen cấp huyện/tỉnh+"),
    ]
    df = pd.DataFrame(rows)
    df["criteria"] = df["criteria"].apply(lambda x: ",".join(x))
    return df


def load_or_create_activities(path: str = "data/activities.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if "criteria" not in df.columns:
            df["criteria"] = ""
        return df
    except Exception:
        df = demo_activities()
        import os
        os.makedirs("data", exist_ok=True)
        df.to_csv(path, index=False)
        return df


def criteria_from_activities(acts_df: pd.DataFrame, selected_ids: List[str]) -> Dict[str, bool]:
    hit = {c: False for c in CRITERIA}
    if not selected_ids:
        return hit

    sel = acts_df[acts_df["activity_id"].isin(selected_ids)].copy()
    if sel.empty:
        return hit

    for s in sel["criteria"].astype(str).tolist():
        for c in [x.strip() for x in s.split(",") if x.strip()]:
            if c in hit:
                hit[c] = True
    return hit


def recommend_activities(acts_df: pd.DataFrame, missing: List[str], topk: int = 6) -> pd.DataFrame:
    if not missing:
        return acts_df.sort_values("date").head(topk)

    def score_row(r):
        crits = [x.strip() for x in str(r["criteria"]).split(",") if x.strip()]
        # điểm nếu hoạt động giúp tiêu chí còn thiếu
        return sum(1 for m in missing if m in crits) * 10 + float(r.get("points", 0))

    df = acts_df.copy()
    df["rec_score"] = df.apply(score_row, axis=1)
    df = df.sort_values(["rec_score", "date"], ascending=[False, True])
    return df.head(topk)
