import os
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.activities import load_or_create_activities


def make_synthetic_sv5t_data(n: int = 450, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    school_level = rng.choice(["daihoc_hocvien", "caodang"], size=n, p=[0.75, 0.25])
    grading_scale = rng.choice([4, 10], size=n, p=[0.85, 0.15])

    # Đạo đức
    ren_luyen = np.clip(rng.normal(88, 8, n), 50, 100).round(0).astype(int)
    no_violation = (rng.random(n) > 0.07).astype(int)
    marx_team_member = (rng.random(n) > 0.85).astype(int)
    good_deed_awarded = (rng.random(n) > 0.92).astype(int)

    # GPA
    gpa = np.zeros(n, dtype=float)
    for i in range(n):
        if grading_scale[i] == 4:
            gpa[i] = np.clip(rng.normal(3.25, 0.40), 2.0, 4.0)
        else:
            gpa[i] = np.clip(rng.normal(8.2, 0.90), 5.0, 10.0)
    gpa = np.round(gpa, 2)

    # tiêu chí học thuật (chỉ cần 1)
    nckh_good = (rng.random(n) > 0.80).astype(int)
    thesis_award = (rng.random(n) > 0.95).astype(int)
    journal_paper = (rng.random(n) > 0.93).astype(int)
    conference_proceeding = (rng.random(n) > 0.88).astype(int)
    patent_or_creative_award = (rng.random(n) > 0.97).astype(int)
    academic_team_member = (rng.random(n) > 0.97).astype(int)
    innovation_award = (rng.random(n) > 0.96).astype(int)

    # Thể lực (1/2)
    sv_khoe_provincial_or_higher = (rng.random(n) > 0.92).astype(int)
    sport_award_school_or_higher = (rng.random(n) > 0.88).astype(int)

    # Tình nguyện
    volunteer_days = np.clip(rng.poisson(4, n), 0, 15).astype(int)
    volunteer_award_prov_or_district = (rng.random(n) > 0.80).astype(int)

    # Hội nhập
    skill_course_or_youth_union_award = (rng.random(n) > 0.55).astype(int)
    integration_activity_count = np.clip(rng.poisson(1, n), 0, 5).astype(int)
    english_b1_or_equivalent = (rng.random(n) > 0.65).astype(int)

    foreign_language_gpa = np.zeros(n, dtype=float)
    for i in range(n):
        if grading_scale[i] == 4:
            foreign_language_gpa[i] = np.clip(rng.normal(3.1, 0.35), 2.0, 4.0)
        else:
            foreign_language_gpa[i] = np.clip(rng.normal(7.8, 0.9), 5.0, 10.0)
    foreign_language_gpa = np.round(foreign_language_gpa, 2)

    international_exchange = (rng.random(n) > 0.85).astype(int)
    integration_competition_award = (rng.random(n) > 0.90).astype(int)

    # Rule “chuẩn PDF” (có thể rất ít 1 -> ta sẽ auto-fix ở main nếu cần)
    ethics_ok = (ren_luyen >= 90) & (no_violation == 1) & ((marx_team_member == 1) | (good_deed_awarded == 1))

    def gpa_need(level, scale):
        if level == "daihoc_hocvien":
            return 3.4 if scale == 4 else 8.5
        return 3.2 if scale == 4 else 8.0

    gpa_ok = np.array([gpa[i] >= gpa_need(school_level[i], int(grading_scale[i])) for i in range(n)])
    academic_plus_ok = (
        (nckh_good == 1) | (thesis_award == 1) | (journal_paper == 1) | (conference_proceeding == 1) |
        (patent_or_creative_award == 1) | (academic_team_member == 1) | (innovation_award == 1)
    )
    study_ok = gpa_ok & academic_plus_ok

    fitness_ok = (sv_khoe_provincial_or_higher == 1) | (sport_award_school_or_higher == 1)
    volunteer_ok = (volunteer_days >= 5) & (volunteer_award_prov_or_district == 1)

    lang_ok = (english_b1_or_equivalent == 1) | (foreign_language_gpa >= np.where(grading_scale == 4, 3.2, 8.0))
    integration_mandatory_ok = (skill_course_or_youth_union_award == 1) & (integration_activity_count >= 1) & lang_ok
    integration_extra_ok = (international_exchange == 1) | (integration_competition_award == 1)
    integration_ok = integration_mandatory_ok & integration_extra_ok

    sv5t = (ethics_ok & study_ok & fitness_ok & volunteer_ok & integration_ok).astype(int)

    return pd.DataFrame({
        "school_level": school_level,
        "grading_scale": grading_scale.astype(int),
        "gpa": gpa,

        "ren_luyen": ren_luyen,
        "no_violation": no_violation,
        "marx_team_member": marx_team_member,
        "good_deed_awarded": good_deed_awarded,

        "nckh_good": nckh_good,
        "thesis_award": thesis_award,
        "journal_paper": journal_paper,
        "conference_proceeding": conference_proceeding,
        "patent_or_creative_award": patent_or_creative_award,
        "academic_team_member": academic_team_member,
        "innovation_award": innovation_award,

        "sv_khoe_provincial_or_higher": sv_khoe_provincial_or_higher,
        "sport_award_school_or_higher": sport_award_school_or_higher,

        "volunteer_days": volunteer_days,
        "volunteer_award_prov_or_district": volunteer_award_prov_or_district,

        "skill_course_or_youth_union_award": skill_course_or_youth_union_award,
        "integration_activity_count": integration_activity_count,
        "english_b1_or_equivalent": english_b1_or_equivalent,
        "foreign_language_gpa": foreign_language_gpa,
        "international_exchange": international_exchange,
        "integration_competition_award": integration_competition_award,

        "sv5t": sv5t
    })


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    # tạo activities.csv demo (để app có “hoạt động hiện tại”)
    load_or_create_activities("data/activities.csv")

    df = make_synthetic_sv5t_data(n=450, seed=42)

    # HARD FIX: nếu chỉ có 1 class -> ép tạo positives để train
    if df["sv5t"].nunique() < 2:
        score = (
            (df["ren_luyen"] >= 90).astype(int) +
            (df["no_violation"] == 1).astype(int) +
            ((df["marx_team_member"] == 1) | (df["good_deed_awarded"] == 1)).astype(int) +
            (df["volunteer_days"] >= 5).astype(int) +
            (df["volunteer_award_prov_or_district"] == 1).astype(int) +
            (df["skill_course_or_youth_union_award"] == 1).astype(int) +
            (df["integration_activity_count"] >= 1).astype(int) +
            (
                (df["english_b1_or_equivalent"] == 1) |
                (df["foreign_language_gpa"] >= np.where(df["grading_scale"] == 4, 3.2, 8.0))
            ).astype(int) +
            ((df["international_exchange"] == 1) | (df["integration_competition_award"] == 1)).astype(int) +
            (df["sv_khoe_provincial_or_higher"] == 1).astype(int) +
            (df["sport_award_school_or_higher"] == 1).astype(int)
        )
        k = max(20, int(0.15 * len(df)))
        top_idx = score.sort_values(ascending=False).head(k).index
        df.loc[:, "sv5t"] = 0
        df.loc[top_idx, "sv5t"] = 1
        print(f"[AUTO-FIX] Forced {k} positives so training has 2 classes.")

    print(df["sv5t"].value_counts())

    df.to_csv("data/sv5t.csv", index=False)

    X = df.drop(columns=["sv5t"])
    y = df["sv5t"]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X[numeric_cols], y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("=== Quick Report ===")
    print(classification_report(y_test, y_pred))

    dump({"model": pipe, "numeric_cols": numeric_cols}, "model/sv5t_model.joblib")
    print("Saved model -> model/sv5t_model.joblib")
    print("Saved data  -> data/sv5t.csv")
    print("Saved activities -> data/activities.csv")


if __name__ == "__main__":
    main()
