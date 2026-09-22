
import streamlit as st
from pathlib import Path
import re
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from itertools import combinations

st.set_page_config(page_title="Community Insights Dashboard", page_icon="📊", layout="wide")

# Top logo/header removed as requested.

st.title("📊 Community Insights & Pattern Discovery Dashboard")
st.caption("Excel → Measurement → Pattern Discovery → Segmentation → Individual Profiles")

# -----------------------------
# Helpers
# -----------------------------
LIKERT = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5
}

ISSUES = [
    "Personal & Emotional",
    "Family & Marriage",
    "Employment & Career",
    "Financial & Economic",
    "Physical Health & Healthcare",
    "Emotional Wellbeing & Support",
    "Social & Community Connection",
    "Spiritual Growth & Discipleship",
]

# Map the current master survey (Q1-Q42) into the same eight core need areas
# used by the National dashboard.
ISSUE_QS = {
    "Personal & Emotional": ["Q7", "Q8", "Q9", "Q10", "Q11"],
    "Family & Marriage": ["Q12R", "Q13", "Q14R", "Q15", "Q16"],
    "Employment & Career": ["Q17", "Q18", "Q19", "Q20"],
    "Financial & Economic": ["Q22", "Q23", "Q24"],
    "Physical Health & Healthcare": ["Q26", "Q27", "Q28"],
    "Emotional Wellbeing & Support": ["Q29", "Q30", "Q31R"],
    "Social & Community Connection": ["Q32R", "Q33R", "Q34R"],
    "Spiritual Growth & Discipleship": ["Q35R", "Q36R", "Q37R", "Q38R", "Q39"],
}


def reverse_score(s):
    return 6 - pd.to_numeric(s, errors="coerce")


def qnum(df, n):
    col = f"Q{n}"
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    s = df[col]
    if s.dtype == "object":
        mapped = s.astype(str).str.strip().map(LIKERT)
        s = mapped.where(mapped.notna(), pd.to_numeric(s, errors="coerce"))
    return pd.to_numeric(s, errors="coerce")


def mean_score(series_list):
    m = pd.concat(series_list, axis=1).mean(axis=1, skipna=True)
    return ((m - 1) / 4 * 100).clip(0, 100)


def calculate_scores(df):
    out = df.copy()

    # Reverse-code the positive items so higher scores consistently mean
    # greater concern/support need.
    for n in [12, 14, 31, 32, 33, 34, 35, 36, 37, 38]:
        out[f"Q{n}R"] = reverse_score(qnum(out, n))

    def series_for(col):
        if col.endswith("R"):
            return out[col]
        return qnum(out, int(col[1:]))

    for issue, cols in ISSUE_QS.items():
        out[issue] = mean_score([series_for(c) for c in cols])

    def primary_issue(row):
        vals = row[ISSUES].dropna()
        return vals.idxmax() if not vals.empty else "Insufficient data"

    out["Primary Issue"] = out[ISSUES].apply(primary_issue, axis=1)

    def secondary(row):
        selected = [
            c for c in ISSUES
            if c != row["Primary Issue"]
            and pd.notna(row[c])
            and row[c] >= 50
        ]
        return ", ".join(selected)

    out["Secondary Issues"] = out.apply(secondary, axis=1)

    def band(x):
        if pd.isna(x):
            return "Not available"
        if x < 25:
            return "Low"
        if x < 50:
            return "Moderate"
        if x < 75:
            return "High"
        return "Very High"

    for c in ISSUES:
        out[c + " Level"] = out[c].apply(band)

    if "Person_ID" not in out.columns:
        out.insert(0, "Person_ID", [f"P{n:04d}" for n in range(1, len(out)+1)])

    return out


def run_clustering(df, k=4):
    """Cluster respondents using the eight core issue indices.

    Returns a dataframe containing complete cases used for clustering,
    a cluster profile table, and the silhouette score. Cluster labels are
    1-based for leader-facing display.
    """
    if df is None or df.empty:
        return None, pd.DataFrame(), np.nan

    work = df.dropna(subset=ISSUES).copy()
    if len(work) < max(3, k):
        return None, pd.DataFrame(), np.nan

    k = int(max(2, min(k, len(work) - 1)))
    X = work[ISSUES].astype(float)
    Xs = StandardScaler().fit_transform(X)

    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(Xs) + 1
    work["Cluster"] = labels

    profiles = work.groupby("Cluster")[ISSUES].mean().round(1)
    sil = silhouette_score(Xs, labels)
    return work, profiles, float(sil)

def top_combinations(df, threshold=50, max_size=3):
    """Return pair/triple problem combinations inside the supplied subset."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Combination", "Number of People", "Percentage"])
    flags = df[ISSUES].ge(threshold)
    rows = []
    for size in range(2, max_size + 1):
        for combo in combinations(ISSUES, size):
            mask = flags[list(combo)].all(axis=1)
            count = int(mask.sum())
            if count > 0:
                rows.append({
                    "Combination": " + ".join(combo),
                    "Number of People": count,
                    "Percentage": round(count / len(df) * 100, 1)
                })
    if not rows:
        return pd.DataFrame(columns=["Combination", "Number of People", "Percentage"])
    return pd.DataFrame(rows).sort_values(
        ["Number of People", "Combination"], ascending=[False, True]
    ).reset_index(drop=True)


def combination_people(frame, combination, threshold):
    parts = [x.strip() for x in str(combination).split(" + ")]
    flags = frame[ISSUES].ge(threshold)
    return frame.loc[flags[parts].all(axis=1)].copy()


def group_combination_summary(frame, group_col, threshold=50):
    """Find the strongest pair/triple combination within each demographic group."""
    rows = []
    if group_col not in frame.columns:
        return pd.DataFrame(columns=[group_col, "Group People", "Top Combination", "People", "% of Group"])
    for value, group in frame.groupby(group_col, dropna=True):
        combos = top_combinations(group, threshold=threshold, max_size=3)
        if not combos.empty:
            top = combos.iloc[0]
            rows.append({
                group_col: str(value),
                "Group People": len(group),
                "Top Combination": top["Combination"],
                "People": int(top["Number of People"]),
                "% of Group": float(top["Percentage"]),
            })
    return pd.DataFrame(rows).sort_values("% of Group", ascending=False) if rows else pd.DataFrame(
        columns=[group_col, "Group People", "Top Combination", "People", "% of Group"]
    )


def show_combination_analysis(frame, key_prefix, title_prefix="Selected group"):
    """Show combinations plus traceable Person_ID/full-record exports."""
    if frame is None or frame.empty:
        st.info("No respondents are available for this selection.")
        return

    threshold = st.slider(
        "Define a higher problem as score ≥",
        25, 80, 50,
        key=f"{key_prefix}_threshold"
    )
    combos = top_combinations(frame, threshold=threshold, max_size=3)

    if combos.empty:
        st.info("No pair or triple problem combinations meet the selected threshold.")
        return

    st.markdown("### 🔗 Problem Combinations")
    st.caption(
        f"Combinations are calculated within this selected group (n={len(frame)}), "
        "not against the full dataset."
    )
    st.dataframe(combos.head(20), use_container_width=True, hide_index=True)

    selected_combo = st.selectbox(
        "Select a problem combination to view/download people",
        combos["Combination"].tolist(),
        key=f"{key_prefix}_combo_choice"
    )
    people = combination_people(frame, selected_combo, threshold)

    st.markdown(f"#### 👥 {title_prefix}: {selected_combo}")
    st.metric(
        "People in selected combination",
        f"{len(people)} ({len(people)/len(frame)*100:.1f}% of selected group)"
    )
    show_people_export(
        people,
        f"People in {selected_combo}",
        f"{key_prefix}_combo_people",
        show_table=True
    )


# -----------------------------
# Data loading
# -----------------------------
MASTER_DATA_FILE = "Unified_Master_Community_Insights_Dataset_V4.xlsx"
APP_DIR = Path(__file__).resolve().parent

st.sidebar.subheader("📁 Data Source")
uploaded_excel = st.sidebar.file_uploader(
    "Upload your Excel survey data",
    type=["xlsx", "xls"],
    help="Excel must contain a sheet named 'Responses'. "
         "If no file is uploaded, the bundled V4 master dataset is used."
)

def load_responses():
    if uploaded_excel is not None:
        try:
            return pd.read_excel(uploaded_excel, sheet_name="Responses"), uploaded_excel.name
        except Exception as exc:
            st.error("The uploaded Excel file must contain a sheet named 'Responses'.")
            st.exception(exc)
            return None, None

    master_path = APP_DIR / MASTER_DATA_FILE
    if master_path.exists():
        return pd.read_excel(master_path, sheet_name="Responses"), MASTER_DATA_FILE

    for candidate in sorted(APP_DIR.glob("*.xlsx")):
        try:
            if "Responses" in pd.ExcelFile(candidate).sheet_names:
                return pd.read_excel(candidate, sheet_name="Responses"), candidate.name
        except Exception:
            continue
    return None, None

raw, loaded_source = load_responses()

if raw is None:
    st.error("No usable Excel workbook was found.")
    st.info("Upload your Excel file from the sidebar or keep the bundled V4 workbook beside app.py.")
    st.stop()

st.sidebar.success(f"Dataset: {loaded_source}")

df = calculate_scores(raw)

# Exact column matching helper. Defined before church/community display helpers
# because the sidebar uses those helpers during application startup.
def _find_column(frame, aliases):
    """Find an exact column match only; never use substring matching."""
    normalized = {str(c).strip().lower(): c for c in frame.columns}
    for alias in aliases:
        key = str(alias).strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _church_name_column(frame):
    """Return a genuine church/community name column if the dataset provides one."""
    return _find_column(
        frame,
        [
            "Church_Name", "Church Name", "ChurchName",
            "Community_Name", "Community Name", "CommunityName",
        ],
    )


def _church_display_name(frame, community_id):
    """Human-readable church/community label without inventing a church name."""
    if community_id == "All Communities":
        return "All Communities"

    name_col = _church_name_column(frame)
    if name_col:
        vals = (
            frame.loc[
                frame["Community_ID"].astype(str) == str(community_id),
                name_col
            ]
            .dropna().astype(str).str.strip()
        )
        vals = [v for v in vals if v]
        if vals:
            return f"{vals[0]} ({community_id})"

    # The current V4 dataset has Community_ID but Church_ID is blank in the
    # supplied records. Use available location information only as a descriptor,
    # never as a fabricated church name.
    sub = frame[frame["Community_ID"].astype(str) == str(community_id)]
    locations = []
    for col in ["Church_ID", "Local_Area", "District"]:
        if col in sub.columns:
            vals = (
                sub[col].dropna().astype(str).str.strip().replace("", np.nan)
                .dropna().drop_duplicates().tolist()
            )
            if col == "Church_ID" and vals:
                return f"Church ID {vals[0]} ({community_id})"
            if col != "Church_ID":
                locations.extend(vals)

    locations = list(dict.fromkeys(locations))
    if locations:
        return f"Community {community_id} — {', '.join(locations[:3])}"
    return f"Community {community_id} — church name not provided"


def _issue_people_export(frame, issue, threshold, key):
    """Export people meeting a selected issue threshold."""
    if frame is None or frame.empty or issue not in frame.columns:
        return
    people = frame.loc[frame[issue].ge(threshold)].copy()
    show_people_export(
        people,
        title=f"People with {issue} ≥ {threshold}",
        key=key,
        show_table=True,
    )



# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

filtered = df.copy()

# Optional country selector. If Country exists, every V2 FIXED analysis
# is performed inside the selected country.
if "Country" in df.columns:
    country_options = ["All Countries"] + sorted(
        df["Country"].dropna().astype(str).unique().tolist()
    )
    selected_country = st.sidebar.selectbox("Country", country_options)
    if selected_country != "All Countries":
        filtered = filtered[
            filtered["Country"].astype(str) == selected_country
        ]

if "Q1" in df.columns:
    age_options = ["All"] + [x for x in df["Q1"].dropna().astype(str).unique()]
    age = st.sidebar.selectbox("Age Group", age_options)
    if age != "All":
        filtered = filtered[filtered["Q1"].astype(str) == age]

if "Q2" in df.columns:
    life_options = ["All"] + [x for x in df["Q2"].dropna().astype(str).unique()]
    life = st.sidebar.selectbox("Life Stage", life_options)
    if life != "All":
        filtered = filtered[filtered["Q2"].astype(str) == life]

# Church / Community selector for the Church Pattern Discovery workflow.
if "Community_ID" in df.columns:
    community_values = sorted(
        df["Community_ID"].dropna().astype(str).unique().tolist()
    )
    community_options = ["All Communities"] + community_values
    community_labels = {
        cid: _church_display_name(df, cid) for cid in community_values
    }
    selected_community = st.sidebar.selectbox(
        "Church / Community",
        community_options,
        index=0,
        format_func=lambda x: (
            "All Communities — whole church/community population"
            if x == "All Communities" else community_labels.get(x, x)
        ),
        help="Choose the whole church/community population or a specific community."
    )
    if selected_community != "All Communities":
        filtered = filtered[
            filtered["Community_ID"].astype(str) == selected_community
        ]
else:
    selected_community = "All Communities"

# Pattern / concern threshold used throughout the church analysis.
issue_threshold = st.sidebar.slider(
    "Issue Threshold",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help="Change the score threshold used for People ≥ threshold, issue combinations, pattern discovery and worksheet evidence."
)

st.sidebar.caption(
    f"Showing {len(filtered)} of {len(df)} respondents"
    + (f" • {selected_country}" if "Country" in df.columns else "")
)


# -----------------------------
# Universal Cross-Analysis helpers
# -----------------------------
def _cross_variables(frame):
    """All usable variables from the V2 FIXED dataset."""
    variables = []

    # Keep the existing V2 FIXED questions exactly as supplied.
    for i in range(1, 43):
        c = f"Q{i}"
        if c in frame.columns:
            variables.append(c)

    # Metadata fields.
    for c in [
        "Country", "Region", "District", "Local_Area",
        "Community_ID", "Church_ID", "Leader_ID",
        "Survey_Level", "Urban_Rural",
        "Primary Issue", "Secondary Issues"
    ]:
        if c in frame.columns and c not in variables:
            variables.append(c)

    # Existing calculated issue indices.
    for c in ISSUES:
        if c in frame.columns and c not in variables:
            variables.append(c)

    for c in [
        "Community Connection", "Discipleship Engagement",
        "Relational Isolation"
    ]:
        if c in frame.columns and c not in variables:
            variables.append(c)

    return variables


def _cross_is_numeric(s):
    return pd.api.types.is_numeric_dtype(s)


def _cross_display_series(frame, variable):
    """
    Converts a selected variable into analysis categories.
    Numeric Likert questions become response categories.
    Numeric indices become severity bands.
    Categorical questions/metadata remain categories.
    Secondary Issues is treated as a multi-value field.
    """
    s = frame[variable]

    if variable == "Secondary Issues":
        return s.map(
            lambda x: ["Missing"] if pd.isna(x) or not str(x).strip()
            else [v.strip() for v in str(x).split(",") if v.strip()]
        )

    if _cross_is_numeric(s):
        clean = pd.to_numeric(s, errors="coerce").dropna()
        if len(clean) and clean.between(1, 5).all():
            mapping = {
                1: "1 - Strongly Disagree",
                2: "2 - Disagree",
                3: "3 - Neutral",
                4: "4 - Agree",
                5: "5 - Strongly Agree"
            }
            return s.map(lambda x: "Missing" if pd.isna(x) else mapping.get(int(x), str(x)))

        return s.map(
            lambda x: "Missing" if pd.isna(x) else
            ("Low (0-39)" if x < 40 else
             "Moderate (40-59)" if x < 60 else
             "High (60-74)" if x < 75 else
             "Very High (75-100)")
        )

    return s.map(lambda x: "Missing" if pd.isna(x) else str(x))


def _cross_exact_mask(frame, variable, value):
    """Return people matching a displayed cross-analysis category."""
    s = frame[variable]

    if variable == "Secondary Issues":
        return s.map(
            lambda x: False if pd.isna(x) else
            str(value).lower() in [v.strip().lower() for v in str(x).split(",")]
        )

    if _cross_is_numeric(s):
        numeric = pd.to_numeric(s, errors="coerce")

        response_map = {
            "1 - Strongly Disagree": numeric == 1,
            "2 - Disagree": numeric == 2,
            "3 - Neutral": numeric == 3,
            "4 - Agree": numeric == 4,
            "5 - Strongly Agree": numeric == 5,
        }
        if value in response_map:
            return response_map[value]

        if value == "Low (0-39)":
            return numeric < 40
        if value == "Moderate (40-59)":
            return (numeric >= 40) & (numeric < 60)
        if value == "High (60-74)":
            return (numeric >= 60) & (numeric < 75)
        if value == "Very High (75-100)":
            return numeric >= 75

        return numeric.astype(str) == str(value)

    return s.map(lambda x: "Missing" if pd.isna(x) else str(x)) == str(value)


def _universal_cross_table(frame, a, b, c=None):
    """
    Counts people for any two variables, optionally adding a third.
    Secondary Issues is exploded only for the cross-tab so a person can
    legitimately contribute to each secondary issue they reported.
    """
    work = frame.copy()
    work["_A"] = _cross_display_series(work, a)
    work["_B"] = _cross_display_series(work, b)

    if c:
        work["_C"] = _cross_display_series(work, c)

    # Explode list-valued variables one at a time.
    for col in ["_A", "_B", "_C"]:
        if col in work.columns:
            if work[col].apply(lambda x: isinstance(x, list)).any():
                work = work.explode(col)

    group_cols = ["_A", "_B"] + (["_C"] if c else [])
    result = (
        work.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="People")
    )

    # Percentage is based on the filtered respondent population.
    denominator = len(frame)
    result["Percentage"] = (
        result["People"] / denominator * 100 if denominator else 0
    ).round(1)

    rename = {"_A": a, "_B": b}
    if c:
        rename["_C"] = c
    return result.rename(columns=rename).sort_values(
        ["People"], ascending=False
    ).reset_index(drop=True)


def _show_cross_people(frame, a, va, b, vb, c=None, vc=None):
    mask = _cross_exact_mask(frame, a, va) & _cross_exact_mask(frame, b, vb)
    if c:
        mask &= _cross_exact_mask(frame, c, vc)

    people = frame.loc[mask].copy()

    st.subheader("👤 People behind the selected result")
    st.metric("Number of People", len(people))

    preferred = [
        "Person_ID", "Respondent_ID", "Name",
        "Country", "Region", "District", "Local_Area",
        "Community_ID", "Church_ID", "Leader_ID",
        "Q1", "Q2", "Primary Issue", "Secondary Issues"
    ]
    cols = [x for x in preferred if x in people.columns]
    if not cols:
        cols = list(people.columns)

    display = people[cols]
    st.dataframe(display, use_container_width=True, hide_index=True)

    if "Person_ID" in people.columns:
        ids = (
            people["Person_ID"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        st.markdown("#### 🆔 Person ID List")
        st.write(f"**{len(ids)} Person IDs**")
        safe_key = abs(hash((a, str(va), b, str(vb), str(vc))))
        st.text_area(
            "Person IDs",
            value="\n".join(ids),
            height=160,
            key=f"helper_ids_{safe_key}"
        )
        id_csv = pd.DataFrame({"Person_ID": ids}).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Person ID list (CSV)",
            data=id_csv,
            file_name="person_id_list.csv",
            mime="text/csv",
            key=f"helper_download_ids_{safe_key}"
        )

    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download selected people (CSV)",
        data=csv,
        file_name="cross_analysis_people.csv",
        mime="text/csv",
        key=f"cross_people_{abs(hash((a,str(va),b,str(vb),str(vc))))}"
    )
    return people


# -----------------------------
# Universal People Export
# -----------------------------
def show_people_export(frame, title="People represented by this result",
                       key="people_export", show_table=False):
    """Every people-counting view gets a Person_ID list + CSV export."""
    if frame is None:
        return
    people = frame.copy()
    st.markdown(f"#### 👤 {title}")
    st.metric("People", len(people))

    if "Person_ID" in people.columns:
        ids = (
            people["Person_ID"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        st.caption(f"{len(ids)} unique Person IDs")
        with st.expander("🆔 View / copy Person ID list"):
            st.text_area(
                "Person IDs",
                value="\\n".join(ids),
                height=160,
                key=f"{key}_ids"
            )
            id_csv = pd.DataFrame({"Person_ID": ids}).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Person ID list (CSV)",
                data=id_csv,
                file_name=f"{key}_person_ids.csv",
                mime="text/csv",
                key=f"{key}_id_download"
            )

    if show_table:
        preferred = [
            "Person_ID", "Leader_ID", "Country", "Region", "District",
            "Local_Area", "Community_ID", "Church_ID", "Survey_Level",
            "Urban_Rural", "Survey_Date", "Primary Issue", "Secondary Issues"
        ]
        cols = [c for c in preferred if c in people.columns]
        if not cols:
            cols = list(people.columns)
        st.dataframe(people[cols], use_container_width=True, hide_index=True)

    csv = people.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download complete filtered people list (CSV)",
        data=csv,
        file_name=f"{key}_people.csv",
        mime="text/csv",
        key=f"{key}_full_download"
    )


def show_group_row_export(frame, group_cols, selected_values, key,
                          title="People behind selected count"):
    """For a table containing group counts, select a row and export its people."""
    mask = pd.Series(True, index=frame.index)
    for col, val in zip(group_cols, selected_values):
        mask &= frame[col].astype(str) == str(val)
    people = frame.loc[mask]
    show_people_export(people, title=title, key=key, show_table=True)


# -----------------------------
# Church Pattern Worksheet — Answer Engine
# -----------------------------
def _issue_question_rows(frame, issue, top_n=3):
    """Return the survey questions that provide the clearest evidence for an issue."""
    rows = []
    for col in ISSUE_QS.get(issue, []):
        is_reverse = col.endswith("R")
        q = col[:-1] if is_reverse else col
        series = frame[col] if col in frame.columns else qnum(frame, int(q[1:]))
        if is_reverse and col not in frame.columns:
            series = reverse_score(series)
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        # 1–5 response scale converted to a 0–100 evidence score.
        score = ((numeric.mean() - 1) / 4 * 100)
        # For reverse-coded questions, the transformed series already means
        # "higher = greater concern/support need".
        pct_high = (numeric >= 4).mean() * 100
        rows.append({
            "Question": q,
            "Issue": issue,
            "Mean (1–5)": round(float(numeric.mean()), 2),
            "Evidence Score (0–100)": round(float(score), 1),
            "% Agree / Strongly Agree": round(float(pct_high), 1),
        })
    if not rows:
        return pd.DataFrame(
            columns=["Question", "Issue", "Mean (1–5)",
                     "Evidence Score (0–100)", "% Agree / Strongly Agree"]
        )
    return pd.DataFrame(rows).sort_values(
        ["Evidence Score (0–100)", "% Agree / Strongly Agree"],
        ascending=False
    ).head(top_n)


_THEME_RULES = {
    "Financial pressure / income": [
        "money", "financial", "finance", "income", "debt", "loan", "salary",
        "cost", "expense", "expenses", "poverty", "afford", "jobless"
    ],
    "Employment / career": [
        "job", "employment", "career", "work", "unemployed", "promotion",
        "business", "workplace", "profession", "skill"
    ],
    "Family / marriage": [
        "family", "marriage", "married", "spouse", "husband", "wife",
        "parent", "parents", "children", "child", "relationship"
    ],
    "Personal / identity / purpose": [
        "identity", "purpose", "meaning", "future", "lonely", "loneliness",
        "self", "confidence", "direction", "belong"
    ],
    "Spiritual growth / discipleship": [
        "faith", "bible", "scripture", "prayer", "pray", "god", "jesus",
        "christ", "church", "spiritual", "discipleship", "devotion"
    ],
    "Emotional wellbeing": [
        "stress", "anxiety", "worry", "depress", "sad", "fear", "emotion",
        "mental", "burnout", "pressure", "overwhelm"
    ],
    "Health / healthcare": [
        "health", "hospital", "doctor", "medical", "medicine", "illness",
        "disease", "treatment", "healthcare", "caregiving"
    ],
    "Community / social connection": [
        "community", "friend", "friends", "social", "isolation", "support",
        "connection", "belonging", "church group", "fellowship"
    ],
    "Education / learning": [
        "education", "study", "school", "college", "university", "learning",
        "training", "course", "knowledge"
    ],
}


def _text_themes(frame, column, top_n=5):
    """
    Lightweight, traceable theme extraction for Q41/Q42.
    It uses transparent keyword rules and reports counts plus representative
    responses. It does not diagnose or infer hidden attributes.
    """
    if column not in frame.columns:
        return pd.DataFrame(columns=["Theme", "Responses", "%", "Evidence"])

    texts = (
        frame[column].dropna().astype(str).str.strip()
    )
    texts = texts[texts != ""]
    if texts.empty:
        return pd.DataFrame(columns=["Theme", "Responses", "%", "Evidence"])

    rows = []
    lower = texts.str.lower()
    denominator = len(texts)

    for theme, keywords in _THEME_RULES.items():
        mask = lower.apply(
            lambda text: any(re.search(r"\b" + re.escape(k) + r"\b", text)
                             for k in keywords)
        )
        count = int(mask.sum())
        if count:
            examples = texts[mask].drop_duplicates().head(2).tolist()
            rows.append({
                "Theme": theme,
                "Responses": count,
                "%": round(count / denominator * 100, 1),
                "Evidence": " | ".join(examples),
            })

    if not rows:
        # Still answer the worksheet: show the most frequent non-empty
        # responses when no predefined theme is detected.
        freq = texts.value_counts().head(top_n)
        return pd.DataFrame({
            "Theme": ["Recurring response"] * len(freq),
            "Responses": freq.values,
            "%": [round(v / denominator * 100, 1) for v in freq.values],
            "Evidence": freq.index.tolist(),
        })

    return pd.DataFrame(rows).sort_values(
        ["Responses", "Theme"], ascending=[False, True]
    ).head(top_n)


def _strongest_group(frame, issues, threshold=50, min_n=2):
    """Step 3: identify who experiences the selected Step 2 pattern.

    A respondent is counted only when every issue in ``issues`` meets the
    threshold. Each demographic/context group is then ranked by the share of
    its members meeting that same pattern.
    """
    if isinstance(issues, str):
        issues = [issues]
    issues = [i for i in issues if i in frame.columns]

    empty_columns = [
        "Dimension", "Group", "People", "Group Size",
        "% of Group", "Mean Pattern Score"
    ]
    if frame.empty or not issues:
        return pd.DataFrame(columns=empty_columns)

    # This is the exact pattern selected in Step 2.
    pattern_hit = frame[issues].ge(threshold).all(axis=1)
    pattern_score = frame[issues].mean(axis=1)

    candidates = [
        ("Age Group", ["Age Group", "Q1"]),
        ("Life Stage", ["Life Stage", "Q2"]),
        ("Community Participation",
         ["Community Participation", "Community_Participation",
          "CommunityParticipation"]),
        ("Spiritual Stage",
         ["Spiritual Stage", "Spiritual_Stage", "SpiritualStage"]),
        ("Urban / Rural", ["Urban_Rural", "Urban/Rural", "Urban Rural"]),
    ]

    results = []
    for label, aliases in candidates:
        col = _find_column(frame, aliases)
        if not col:
            continue

        valid = frame[col].notna() & frame[col].astype(str).str.strip().ne("")
        if not valid.any():
            continue

        work = pd.DataFrame({
            "group_value": frame.loc[valid, col].astype(str),
            "pattern_hit": pattern_hit.loc[valid],
            "pattern_score": pattern_score.loc[valid],
        })

        for value, group in work.groupby("group_value", dropna=True):
            n = len(group)
            if n < min_n:
                continue

            count = int(group["pattern_hit"].sum())
            pct = count / n * 100
            results.append({
                "Dimension": label,
                "Group": str(value),
                "People": count,
                "Group Size": n,
                "% of Group": round(pct, 1),
                "Mean Pattern Score": round(float(group["pattern_score"].mean()), 1),
            })

    if not results:
        return pd.DataFrame(columns=empty_columns)

    # Rank by concentration first, then absolute number of people.
    return pd.DataFrame(results).sort_values(
        ["% of Group", "People", "Group Size"],
        ascending=[False, False, False]
    ).reset_index(drop=True)


def _location_summary(frame, issues, threshold=50, min_n=5):
    """Step 4: show all geographic locations ranked by pattern concentration.

    Each geographic level is calculated independently from the same selected
    Step 2 pattern. Locations with fewer than ``min_n`` respondents are omitted.
    Results are ranked within each level by percentage, then people count.
    """
    parts = [i for i in issues if i in frame.columns]
    if frame.empty or not parts:
        return pd.DataFrame()

    hit = frame[parts].ge(threshold).all(axis=1)
    base = hit.mean() * 100
    levels = [
        ("Region / State", "Region"),
        ("Country", "Country"),
        ("District / Local Area", "District"),
        ("Church / Community", "Community_ID"),
    ]

    rows = []
    for label, col in levels:
        if col not in frame.columns:
            continue

        valid = frame[col].notna() & frame[col].astype(str).str.strip().ne("")
        sub = pd.DataFrame({
            "Location": frame.loc[valid, col].astype(str).str.strip(),
            "Pattern Hit": hit.loc[valid],
        })

        grouped = sub.groupby("Location")["Pattern Hit"].agg(
            **{"People with pattern": "sum", "Respondents": "size"}
        )
        grouped = grouped[grouped["Respondents"] >= min_n].copy()
        if grouped.empty:
            continue

        grouped["% in location"] = (
            grouped["People with pattern"] / grouped["Respondents"] * 100
        )
        grouped = grouped.sort_values(
            ["% in location", "People with pattern", "Respondents"],
            ascending=[False, False, False]
        )

        for rank, (location, values) in enumerate(grouped.iterrows(), start=1):
            rows.append({
                "Level": label,
                "Rank": rank,
                "Location": location,
                "People with pattern": int(values["People with pattern"]),
                "Respondents": int(values["Respondents"]),
                "% in location": round(float(values["% in location"]), 1),
                "Overall %": round(float(base), 1),
            })

    return pd.DataFrame(rows)


def _action_considerations(issue, combo, q41_theme, q42_theme):
    """Generate non-diagnostic ministry considerations from observed evidence."""
    actions = []
    issue_actions = {
        "Personal & Emotional": "Create a safe listening and discipleship pathway around personal wellbeing, identity, purpose and life questions.",
        "Family & Marriage": "Strengthen family and relationship support through Scripture-based conversations, trusted leaders and appropriate referral pathways.",
        "Employment & Career": "Consider Scripture engagement connected to work, vocation, uncertainty, skills and economic pressure.",
        "Financial & Economic": "Consider practical support partnerships alongside Scripture engagement around financial pressure, stewardship and hope.",
        "Physical Health & Healthcare": "Consider accessible Scripture engagement and pastoral support for health, caregiving and healthcare experiences.",
        "Emotional Wellbeing & Support": "Provide safe pastoral listening and Scripture engagement, with referral to qualified professional support where appropriate.",
        "Social & Community Connection": "Strengthen small-group, mentoring and community-connection pathways around belonging and support.",
        "Spiritual Growth & Discipleship": "Strengthen Bible engagement, prayer, mentoring and discipleship pathways matched to the group's expressed needs.",
        "Employment & Career Support": "Explore practical mentoring and Scripture engagement around work and vocation.",
    }
    if issue in issue_actions:
        actions.append(issue_actions[issue])
    if combo:
        actions.append(
            f"Address the combination of {combo} rather than treating each issue in isolation."
        )
    if q41_theme:
        actions.append(
            f"Use the strongest Q41 theme — {q41_theme} — to shape listening and follow-up questions."
        )
    if q42_theme:
        actions.append(
            f"Compare the requested support theme — {q42_theme} — with what the church currently provides."
        )
    return actions[:3]



# -----------------------------
# Full survey-question wording for Step 5
# -----------------------------
# These descriptions are displayed alongside the question IDs in the worksheet.
# They are kept separate from scoring logic so adding readable wording does not
# change any existing calculations.
FULL_QUESTION_TEXT = {
    "Q7": "I feel overwhelmed by my responsibilities.",
    "Q8": "I often worry about the future.",
    "Q9": "I experience loneliness or a lack of meaningful companionship.",
    "Q10": "I find it difficult to cope with disappointment or discouragement.",
    "Q11": "I find it difficult to make important life decisions.",
    "Q12": "There is healthy and open communication in my family.",
    "Q13": "There are unresolved conflicts in my family.",
    "Q14": "My family spends sufficient quality time together.",
    "Q15": "I find it difficult to forgive or resolve relational hurts.",
    "Q16": "Family responsibilities sometimes feel overwhelming.",
    "Q17": "Work or study responsibilities create pressure in my daily life.",
    "Q18": "I am concerned about employment stability and my future career direction.",
    "Q19": "My work or career situation creates uncertainty or pressure.",
    "Q20": "Employment or career concerns affect my personal or family wellbeing.",
    "Q21": "I need practical guidance related to work, career or employability.",
    "Q22": "I have meaningful relationships with people around me.",
    "Q23": "There are people who support me when I face difficulties.",
    "Q24": "I feel comfortable discussing my challenges with someone I trust.",
    "Q25": "I sometimes feel disconnected from my faith community.",
    "Q26": "I would benefit from a small group, mentoring relationship or community support.",
    "Q27": "I experience physical-health concerns that affect my daily life.",
    "Q28": "Accessing healthcare or medical support is difficult when I need it.",
    "Q29": "Health-related responsibilities or caregiving create pressure for me.",
    "Q30": "Stress, anxiety or emotional pressure affects my daily wellbeing.",
    "Q31": "I would benefit from professional help or guidance for challenges I am facing.",
    "Q32": "I feel connected to people in my church or community.",
    "Q33": "I receive the practical and relational support I need from my community.",
    "Q34": "I feel disconnected from people or groups in my church or community.",
    "Q35": "I want to grow in prayer, Scripture engagement and everyday faith.",
    "Q36": "I would like deeper biblical teaching and spiritual mentoring.",
    "Q37": "I would benefit from a stronger discipleship relationship.",
    "Q38": "I have regular opportunities for spiritual growth and discipleship.",
    "Q39": "I need continued encouragement and support in my spiritual journey.",
}


# -----------------------------
# Branding / Footer
# -----------------------------
def render_footer():
    st.markdown("<hr style='margin-top:2rem;margin-bottom:1rem;'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        logo = Path(__file__).resolve().parent / "gracewell_technologies_logo.png"
        if logo.exists():
            st.image(str(logo), width=260)
        st.markdown(
            "<div style='text-align:center;color:#6b7280;font-size:0.82rem;'>"
            "Technology and analytics support by Gracewell Technologies"
            "</div>",
            unsafe_allow_html=True,
        )

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "1. Main Dashboard",
    "2. Issue Analysis",
    "3. Relationships",
    "4. Problem Combinations",
    "5. Pattern Discovery",
    "6. Segment Profiles",
    "7. Age Patterns",
    "8. Life-Stage Patterns",
    "9. Pattern Explorer",
    "10. Cross-Analysis",
    "11. Pattern Worksheet Answers",
    "12. Individual Profiles",
    "13. Data"
])

# 1 Main Dashboard
with tabs[0]:
    st.subheader("🏠 Main Church / Community Dashboard")
    st.caption(
        "See the whole selected church/community at a glance, then drill down into "
        "Issue × Issue, Age × Issue, Life Stage × Issue, patterns, questions and people."
    )

    scope_label = (
        "All Communities — whole church/community population"
        if selected_community == "All Communities"
        else _church_display_name(df, selected_community)
    )
    st.info(f"**Current analysis scope:** {scope_label}")
    st.caption(
        f"Analysis threshold: **{issue_threshold}/100**. Change it in the sidebar "
        "to see how the number and percentage of people meeting the threshold change."
    )

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Respondents", len(filtered))
    issue_means = filtered[ISSUES].mean()
    k2.metric("Top Issue", str(issue_means.idxmax()) if issue_means.notna().any() else "N/A")
    k3.metric(
        "Top Issue Score",
        f"{issue_means.max():.1f}" if issue_means.notna().any() else "N/A"
    )
    k4.metric(
        "Issues ≥50",
        f"{(filtered[ISSUES].ge(issue_threshold).sum(axis=1) > 0).mean()*100:.1f}%"
        if len(filtered) else "0.0%"
    )

    # A. Issue profile
    st.markdown("### 1️⃣ Issue Profile")
    main_issue = pd.DataFrame({
        "Issue": ISSUES,
        "Mean Score": [filtered[c].mean() for c in ISSUES],
        "People ≥ threshold": [int(filtered[c].ge(issue_threshold).sum()) for c in ISSUES],
        "% ≥ threshold": [filtered[c].ge(issue_threshold).mean()*100 for c in ISSUES],
    }).sort_values("Mean Score", ascending=False)

    st.dataframe(main_issue.round(1), use_container_width=True, hide_index=True)
    fig = px.bar(
        main_issue, x="Issue", y="Mean Score", text_auto=".1f",
        title="Issue Scores — Whole Selected Church / Community"
    )
    fig.update_yaxes(range=[0, 100], title="Score (0–100)")
    st.plotly_chart(fig, use_container_width=True)

    # B. Issue × Issue
    st.markdown("### 2️⃣ Issue × Issue")
    issue_corr = filtered[ISSUES].corr()
    fig = px.imshow(
        issue_corr, text_auto=".2f", zmin=-1, zmax=1,
        title="Issue × Issue Relationship"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Association only; correlation does not establish causation.")

    # C. Age × Issue
    st.markdown("### 3️⃣ Age × Issue")
    if "Q1" in filtered.columns:
        age_issue = filtered.groupby("Q1")[ISSUES].mean().round(1)
        st.dataframe(age_issue, use_container_width=True)
        age_long = age_issue.reset_index().melt(
            id_vars=["Q1"], value_vars=ISSUES,
            var_name="Issue", value_name="Score"
        )
        fig = px.bar(
            age_long, x="Q1", y="Score", color="Issue",
            barmode="group", text_auto=".0f",
            title="Age Group × Issue"
        )
        fig.update_yaxes(range=[0, 100], title="Mean Score")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Q1 Age Group is not available.")

    # D. Life Stage × Issue
    st.markdown("### 4️⃣ Life Stage × Issue")
    if "Q2" in filtered.columns:
        stage_issue = filtered.groupby("Q2")[ISSUES].mean().round(1)
        st.dataframe(stage_issue, use_container_width=True)
        stage_long = stage_issue.reset_index().melt(
            id_vars=["Q2"], value_vars=ISSUES,
            var_name="Issue", value_name="Score"
        )
        fig = px.bar(
            stage_long, x="Q2", y="Score", color="Issue",
            barmode="group", text_auto=".0f",
            title="Life Stage × Issue"
        )
        fig.update_yaxes(range=[0, 100], title="Mean Score")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Q2 Life Stage is not available.")

    # E. Compare up to two smaller groups
    st.markdown("### 5️⃣ Compare Groups")
    st.caption(
        "Select a dimension and up to two groups to compare their issue profiles. "
        "This is descriptive comparison, not a ranking."
    )

    compare_candidates = []
    for label, aliases in [
        ("Age Group", ["Q1"]),
        ("Life Stage", ["Q2"]),
        ("Country", ["Country"]),
        ("Region / State", ["Region", "State"]),
        ("District", ["District"]),
        ("Community", ["Community_ID"]),
        ("Church", ["Church_ID"]),
        ("Urban / Rural", ["Urban_Rural", "Urban/Rural", "Urban Rural"]),
        ("Community Participation", ["Community Participation", "Community_Participation"]),
        ("Spiritual Stage", ["Spiritual Stage", "Spiritual_Stage"]),
    ]:
        found = _find_column(filtered, aliases)
        if found and filtered[found].dropna().nunique() >= 2:
            compare_candidates.append((label, found))

    if compare_candidates:
        compare_label = st.selectbox(
            "Compare by",
            [x[0] for x in compare_candidates],
            key="main_compare_dimension"
        )
        compare_col = dict(compare_candidates)[compare_label]
        compare_values = sorted(
            filtered[compare_col].dropna().astype(str).unique().tolist()
        )
        selected_groups = st.multiselect(
            f"Select up to 2 {compare_label} groups",
            compare_values,
            max_selections=2,
            key="main_compare_groups"
        )

        if len(selected_groups) == 0:
            st.info("Select one or two groups to see their issue profiles.")
        else:
            comp_rows = []
            for group in selected_groups:
                mask = filtered[compare_col].astype(str) == group
                subset = filtered.loc[mask]
                row = {"Group": group, "People": len(subset)}
                for issue in ISSUES:
                    row[issue] = float(subset[issue].mean()) if len(subset) else float("nan")
                comp_rows.append(row)

            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df.round(1), use_container_width=True, hide_index=True)

            comp_long = comp_df.melt(
                id_vars=["Group", "People"],
                value_vars=ISSUES,
                var_name="Issue",
                value_name="Score"
            )
            fig_compare = px.bar(
                comp_long,
                x="Issue",
                y="Score",
                color="Group",
                barmode="group",
                text_auto=".1f",
                title=f"{compare_label} — Issue Comparison"
            )
            fig_compare.update_yaxes(range=[0, 100], title="Mean Score (0–100)")
            st.plotly_chart(fig_compare, use_container_width=True)

            if len(selected_groups) == 2:
                g1, g2 = selected_groups
                n1 = int((filtered[compare_col].astype(str) == g1).sum())
                n2 = int((filtered[compare_col].astype(str) == g2).sum())
                st.caption(
                    f"Comparison: **{g1} (n={n1})** vs **{g2} (n={n2})**. "
                    "Differences are descriptive; check sample sizes and context before interpretation."
                )
    else:
        st.info("No comparable grouping fields with at least two groups are available.")

    # F. Top combinations
    st.markdown("### 6️⃣ Top Problem Combinations")
    main_combos = top_combinations(filtered, threshold=issue_threshold, max_size=3)
    if main_combos.empty:
        st.info("No issue combinations meet the ≥50 threshold.")
    else:
        st.dataframe(main_combos.head(10), use_container_width=True, hide_index=True)
        fig = px.bar(
            main_combos.head(10),
            x="Number of People", y="Combination",
            orientation="h", text_auto=True,
            title="Most Frequent Issue Combinations"
        )
        st.plotly_chart(fig, use_container_width=True)

    # F. Q41/Q42 themes
    st.markdown("### 7️⃣ What People Are Saying")
    q41_main = _text_themes(filtered, "Q41")
    q42_main = _text_themes(filtered, "Q42")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Q41 — Significant Challenge Themes**")
        if q41_main.empty:
            st.info("No Q41 responses available.")
        else:
            st.dataframe(q41_main, use_container_width=True, hide_index=True)
            fig = px.bar(
                q41_main, x="Responses", y="Theme",
                orientation="h", text_auto=True,
                title="Q41 Themes"
            )
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Q42 — Requested Support Themes**")
        if q42_main.empty:
            st.info("No Q42 responses available.")
        else:
            st.dataframe(q42_main, use_container_width=True, hide_index=True)
            fig = px.bar(
                q42_main, x="Responses", y="Theme",
                orientation="h", text_auto=True,
                title="Q42 Themes"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Score bands are provisional for this prototype. The main dashboard is an "
        "overview; use the detailed tabs for drill-down and traceability to respondents."
    )

# 2 Issue Analysis
with tabs[1]:
    st.subheader("Which problems are most common?")
    rows = []
    for issue in ISSUES:
        rows.append({
            "Issue": issue,
            "Mean": filtered[issue].mean(),
            "Median": filtered[issue].median(),
            "Std Dev": filtered[issue].std(),
            "High/Very High": int(filtered[issue].ge(60).sum()),
            "High/Very High (%)": filtered[issue].ge(60).mean()*100
        })
    stats = pd.DataFrame(rows).sort_values("High/Very High (%)", ascending=False)
    st.dataframe(stats.round(1), use_container_width=True, hide_index=True)
    show_people_export(filtered, "People represented in Issue Analysis",
                       "church_issue_analysis")

    fig = px.bar(stats, x="Issue", y="High/Very High (%)", text_auto=".1f",
                 title="Prevalence of Higher Problem Scores")
    fig.update_yaxes(range=[0, 100], title="% of respondents")
    st.plotly_chart(fig, use_container_width=True)

# 3 Relationships
with tabs[2]:
    st.subheader("Which problems tend to occur together?")
    corr = filtered[ISSUES].corr()
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                    title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Correlation shows association, not causation.")

    pairs = []
    for a, b in combinations(ISSUES, 2):
        pairs.append({"Issue A": a, "Issue B": b, "Correlation": corr.loc[a, b]})
    pairs = pd.DataFrame(pairs).sort_values("Correlation", ascending=False)
    st.dataframe(pairs.round(3), use_container_width=True, hide_index=True)
    show_people_export(filtered, "People represented in Relationship Analysis",
                       "church_relationships")

# 4 Combinations
with tabs[3]:
    st.subheader("Which problem combinations repeatedly occur?")
    threshold = st.slider("Define 'higher problem' as score ≥", 25, 80, 50)
    combos = top_combinations(filtered, threshold=threshold, max_size=3)
    st.dataframe(combos, use_container_width=True, hide_index=True)

    if not combos.empty:
        combo_labels = combos["Combination"].tolist()
        selected_combo = st.selectbox(
            "Select a combination to view/download its people",
            combo_labels,
            key="church_combo_export_choice"
        )
        selected_combo_people = filtered.copy()
        flags = selected_combo_people[ISSUES].ge(threshold)
        parts = [x.strip() for x in selected_combo.split(" + ")]
        selected_combo_people = selected_combo_people[
            flags[parts].all(axis=1)
        ]
        show_people_export(
            selected_combo_people,
            f"People in: {selected_combo}",
            "church_selected_combination",
            show_table=True
        )

        fig = px.bar(combos.head(15), x="Number of People", y="Combination",
                     orientation="h", text_auto=True,
                     title="Most Frequent Problem Combinations")
        st.plotly_chart(fig, use_container_width=True)

# 5 Pattern Discovery
with tabs[4]:
    st.subheader("Statistical Pattern Discovery")
    st.write("K-Means groups people using their standardized eight-area problem profiles. Age and Life Stage are intentionally excluded from clustering so they can be used later to interpret the discovered segments.")

    valid_n = filtered[ISSUES].dropna().shape[0]
    max_k = min(8, max(2, valid_n - 1))

    if valid_n < 4:
        st.warning("Not enough complete records for clustering.")
    else:
        k = st.slider("Number of segments (K)", 2, max_k, min(4, max_k))
        clustered, profiles, sil = run_clustering(filtered, k)

        if clustered is not None:
            st.metric("Silhouette Score", f"{sil:.3f}")
            st.caption("Higher values generally indicate better separation, but cluster usefulness must also be interpreted substantively.")

            pca_input = clustered.dropna(subset=ISSUES + ["Cluster"]).copy()
            Xs = StandardScaler().fit_transform(pca_input[ISSUES])
            pca = PCA(n_components=2, random_state=42)
            xy = pca.fit_transform(Xs)
            pca_input["PC1"] = xy[:, 0]
            pca_input["PC2"] = xy[:, 1]
            pca_input["Segment"] = pca_input["Cluster"].astype(str)

            fig = px.scatter(pca_input, x="PC1", y="PC2", color="Segment",
                             hover_data=["Person_ID", "Primary Issue"],
                             title="PCA View of Discovered Segments")
            st.plotly_chart(fig, use_container_width=True)

            cluster_people = clustered.dropna(subset=["Cluster"]).sort_values("Cluster")
            st.dataframe(
                cluster_people[["Person_ID", "Cluster", "Primary Issue", "Secondary Issues"] + ISSUES],
                use_container_width=True, hide_index=True
            )
            show_people_export(
                cluster_people,
                "People included in Pattern Discovery",
                "church_pattern_discovery",
                show_table=False
            )

# 6 Segment Profiles
with tabs[5]:
    st.subheader("What characterizes each discovered group?")
    st.caption("Start with the cluster profile table: it shows the average score for each core need area within each discovered segment. The PCA chart is only a visual map of similarity; PC1 and PC2 are not ministry issues.")

    if "Cluster" not in filtered.columns:
        # calculate a default clustering for display
        valid_n = filtered[ISSUES].dropna().shape[0]
        if valid_n >= 5:
            clustered, profiles, sil = run_clustering(filtered, min(4, valid_n - 1))
        else:
            clustered = None
    else:
        clustered = filtered

    if clustered is not None and "Cluster" in clustered.columns:
        cp = clustered.dropna(subset=["Cluster"]).groupby("Cluster")[ISSUES].mean().round(1)
        st.dataframe(cp, use_container_width=True)

        fig = px.imshow(cp, text_auto=".1f", zmin=0, zmax=100,
                        title="Cluster Problem Profiles")
        st.plotly_chart(fig, use_container_width=True)

        sizes = clustered["Cluster"].value_counts().sort_index().reset_index()
        sizes.columns = ["Cluster", "People"]
        st.dataframe(sizes, use_container_width=True, hide_index=True)
        cluster_choice = st.selectbox(
            "Select cluster to view/download its people",
            sizes["Cluster"].tolist(),
            key="church_segment_cluster_export"
        )
        cluster_people = clustered[
            clustered["Cluster"].astype(str) == str(cluster_choice)
        ]
        show_people_export(
            cluster_people,
            f"People in Cluster {cluster_choice}",
            "church_cluster_people",
            show_table=True
        )
    else:
        st.info("At least five complete records are recommended for segment profiling.")

# 7 Age Patterns
with tabs[6]:
    st.subheader("How do patterns vary by age group?")
    st.caption("Age analysis shows issue scores AND problem combinations within each age group. Select a group, then a combination to see and download the underlying Person_IDs.")
    if "Q1" not in filtered.columns:
        st.warning("Q1 Age Group is not available.")
    else:
        age_profile = filtered.groupby("Q1")[ISSUES].mean().round(1)
        st.dataframe(age_profile, use_container_width=True)

        st.markdown("### 🔗 Top problem combination by age group")
        age_combo_summary = group_combination_summary(filtered, "Q1", threshold=50)
        if not age_combo_summary.empty:
            st.dataframe(age_combo_summary, use_container_width=True, hide_index=True)
            st.caption("The combination percentage uses the selected age group's respondent count as the denominator.")

        age_choice = st.selectbox(
            "Select age group to investigate",
            sorted(filtered["Q1"].dropna().astype(str).unique()),
            key="church_age_people_choice"
        )
        age_people = filtered[filtered["Q1"].astype(str) == age_choice].copy()

        st.markdown(f"### 👥 Age group: {age_choice}")
        st.metric("People in age group", len(age_people))

        # A. Individual issue analysis inside the selected age group
        st.markdown("### 🎯 Individual Issue → People")
        st.caption("Select one issue to see the people in this age group who meet the selected problem threshold.")
        age_issue = st.selectbox(
            "Select an individual issue",
            ISSUES,
            key="church_age_issue_choice"
        )
        age_issue_threshold = st.slider(
            "Problem threshold for the selected issue",
            25, 80, 50,
            key="church_age_issue_threshold"
        )
        age_issue_people = age_people[age_people[age_issue].ge(age_issue_threshold)].copy()
        age_issue_pct = (len(age_issue_people) / len(age_people) * 100) if len(age_people) else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("People meeting threshold", len(age_issue_people))
        c2.metric("% of age group", f"{age_issue_pct:.1f}%")
        c3.metric("Mean issue score", f"{age_people[age_issue].mean():.1f}")
        show_people_export(
            age_issue_people,
            f"{age_choice} — {age_issue} ≥ {age_issue_threshold}",
            "church_age_issue_people",
            show_table=True
        )

        # B. Combination analysis inside the selected age group
        show_combination_analysis(
            age_people,
            "church_age_pattern",
            f"Age group {age_choice}"
        )

        fig = px.imshow(
            age_profile, text_auto=".1f", zmin=0, zmax=100,
            title="Age Group × Problem Score"
        )
        st.plotly_chart(fig, use_container_width=True)

# 8 Life Stage
with tabs[7]:
    st.subheader("How do patterns vary by life stage?")
    st.caption("Life-stage analysis shows issue scores AND problem combinations within each life stage. Select a stage, then a combination to see and download the underlying Person_IDs.")
    if "Q2" not in filtered.columns:
        st.warning("Q2 Life Stage is not available.")
    else:
        life_profile = filtered.groupby("Q2")[ISSUES].mean().round(1)
        st.dataframe(life_profile, use_container_width=True)

        st.markdown("### 🔗 Top problem combination by life stage")
        life_combo_summary = group_combination_summary(filtered, "Q2", threshold=50)
        if not life_combo_summary.empty:
            st.dataframe(life_combo_summary, use_container_width=True, hide_index=True)
            st.caption("The combination percentage uses the selected life-stage respondent count as the denominator.")

        life_choice = st.selectbox(
            "Select life stage to investigate",
            sorted(filtered["Q2"].dropna().astype(str).unique()),
            key="church_life_people_choice"
        )
        life_people = filtered[filtered["Q2"].astype(str) == life_choice].copy()

        st.markdown(f"### 👥 Life stage: {life_choice}")
        st.metric("People in life stage", len(life_people))

        # A. Individual issue analysis inside the selected life stage
        st.markdown("### 🎯 Individual Issue → People")
        st.caption("Select one issue to see the people in this life stage who meet the selected problem threshold.")
        life_issue = st.selectbox(
            "Select an individual issue",
            ISSUES,
            key="church_life_issue_choice"
        )
        life_issue_threshold = st.slider(
            "Problem threshold for the selected issue",
            25, 80, 50,
            key="church_life_issue_threshold"
        )
        life_issue_people = life_people[life_people[life_issue].ge(life_issue_threshold)].copy()
        life_issue_pct = (len(life_issue_people) / len(life_people) * 100) if len(life_people) else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("People meeting threshold", len(life_issue_people))
        c2.metric("% of life-stage group", f"{life_issue_pct:.1f}%")
        c3.metric("Mean issue score", f"{life_people[life_issue].mean():.1f}")
        show_people_export(
            life_issue_people,
            f"{life_choice} — {life_issue} ≥ {life_issue_threshold}",
            "church_life_issue_people",
            show_table=True
        )

        # B. Combination analysis inside the selected life stage
        show_combination_analysis(
            life_people,
            "church_life_pattern",
            f"Life stage {life_choice}"
        )

        fig = px.imshow(
            life_profile, text_auto=".1f", zmin=0, zmax=100,
            title="Life Stage × Problem Score"
        )
        st.plotly_chart(fig, use_container_width=True)

# 9 Pattern Explorer
with tabs[8]:
    st.subheader("Pattern Explorer")
    st.write("Use the filters to move from population → life stage → problem pattern → people.")

    explorer = filtered.copy()

    if "Primary Issue" in explorer:
        issue_choice = st.selectbox("Primary Issue", ["All"] + ISSUES)
        if issue_choice != "All":
            explorer = explorer[explorer["Primary Issue"] == issue_choice]

    min_score = st.slider("Minimum score for selected problem", 0, 100, 60)
    selected_issue = st.selectbox("Inspect Problem", ISSUES)
    explorer = explorer[explorer[selected_issue] >= min_score]

    cols = [c for c in ["Person_ID", "Name", "Q1", "Q2", "Primary Issue", "Secondary Issues"] if c in explorer.columns]
    cols += ISSUES
    st.metric("People matching pattern", len(explorer))
    st.dataframe(explorer[cols], use_container_width=True, hide_index=True)
    show_people_export(
        explorer,
        "People matching the selected pattern",
        "church_pattern_explorer",
        show_table=False
    )


# 10 Universal Cross-Analysis
with tabs[9]:
    st.subheader("🔬 Universal Cross-Analysis")
    st.write(
        "Compare ANY available question, demographic field, issue index, "
        "Primary Issue or Secondary Issue with another variable. "
        "Optionally add a third variable."
    )

    variables = _cross_variables(filtered)

    if len(variables) < 2:
        st.warning("At least two variables are required.")
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            var_a = st.selectbox(
                "Variable A",
                variables,
                key="universal_cross_a"
            )

        with c2:
            b_options = [x for x in variables if x != var_a]
            var_b = st.selectbox(
                "Variable B",
                b_options,
                key="universal_cross_b"
            )

        with c3:
            c_options = ["None"] + [
                x for x in variables if x not in {var_a, var_b}
            ]
            var_c_choice = st.selectbox(
                "Optional Variable C",
                c_options,
                key="universal_cross_c"
            )
            var_c = None if var_c_choice == "None" else var_c_choice

        st.caption(
            "Examples: Q1 × Q2 • Q2 × Q8 • Q8 × Q22 • "
            "Q17 × Q29 • Q22 × Q23 • Primary Issue × Secondary Issues • "
            "Church × Issue × Age"
        )

        if st.button(
            "🔍 Analyze selected combination",
            type="primary",
            key="run_universal_cross"
        ):
            result = _universal_cross_table(
                filtered, var_a, var_b, var_c
            )
            st.session_state["universal_cross_result"] = result
            st.session_state["universal_cross_vars"] = (
                var_a, var_b, var_c
            )

        if "universal_cross_result" in st.session_state:
            result = st.session_state["universal_cross_result"]
            saved_a, saved_b, saved_c = st.session_state["universal_cross_vars"]

            st.markdown(
                f"### {saved_a} × {saved_b}"
                + (f" × {saved_c}" if saved_c else "")
            )

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download cross-analysis result (CSV)",
                data=csv,
                file_name="cross_analysis_result.csv",
                mime="text/csv",
                key="download_universal_cross_result"
            )

            if not result.empty and not saved_c:
                try:
                    fig = px.bar(
                        result.head(30),
                        x=saved_a,
                        y="People",
                        color=saved_b,
                        title=f"{saved_a} × {saved_b}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

            if not result.empty:
                labels = []
                for idx, row in result.iterrows():
                    text = (
                        f"{saved_a} = {row[saved_a]} | "
                        f"{saved_b} = {row[saved_b]}"
                    )
                    if saved_c:
                        text += f" | {saved_c} = {row[saved_c]}"
                    text += f" | {int(row['People'])} people"
                    labels.append((idx, text))

                selected_row_index = st.selectbox(
                    "Select a result to see the people behind it",
                    [idx for idx, _ in labels],
                    format_func=lambda x: next(
                        text for idx, text in labels if idx == x
                    ),
                    key="universal_cross_row"
                )

                row = result.loc[selected_row_index]
                selected_people = _show_cross_people(
                    filtered,
                    saved_a, row[saved_a],
                    saved_b, row[saved_b],
                    saved_c,
                    row[saved_c] if saved_c else None
                )

                # Explicit Person-ID list for quick operational use.
                if "Person_ID" in selected_people.columns:
                    st.markdown("### 🆔 Person ID List")
                    ids = (
                        selected_people["Person_ID"]
                        .dropna()
                        .astype(str)
                        .drop_duplicates()
                        .tolist()
                    )
                    st.write(
                        f"**{len(ids)} Person IDs** match this selected combination."
                    )
                    st.text_area(
                        "Copy Person IDs",
                        value="\n".join(ids),
                        height=180,
                        key=f"person_ids_{abs(hash((saved_a, str(row[saved_a]), saved_b, str(row[saved_b]), str(row[saved_c]) if saved_c else '')))}"
                    )

                    ids_df = pd.DataFrame({"Person_ID": ids})
                    ids_csv = ids_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Person ID list (CSV)",
                        data=ids_csv,
                        file_name="person_id_list.csv",
                        mime="text/csv",
                        key="download_person_id_list"
                    )

                st.info(
                    "Every displayed count can be traced to the people behind the selected result. "
                    "Use the Person ID list or CSV for authorized follow-up/research."
                )


# 11 Pattern Worksheet Answers
with tabs[10]:
    st.subheader("📝 Church / Community Pattern Discovery — Complete Answers")
    st.caption(
        "Steps 1–6 are calculated from the selected respondents; Steps 7–8 are "
        "evidence-based drafts for leader validation and discernment."
    )
    scope_label = (
        "All Communities (whole church/community population)"
        if selected_community == "All Communities"
        else _church_display_name(df, selected_community)
    )
    st.info(f"**Current Church / Community scope:** {scope_label}")
    if selected_community != "All Communities" and _church_name_column(df) is None:
        st.warning(
            "The uploaded dataset does not contain a Church Name / Community Name field. "
            "The dashboard therefore shows the Community_ID and available location information "
            "instead of inventing a church name."
        )
    st.caption(
        f"Pattern threshold: **{issue_threshold}/100**. All threshold-based "
        "counts, combinations and group analysis below use this value."
    )

    if filtered.empty:
        st.warning("No respondents are available for the selected church/community scope.")
    else:
        worksheet = filtered.copy()

        # STEP 1 — Top three issues
        issue_summary = pd.DataFrame({
            "Issue": ISSUES,
            "Mean Score": [worksheet[c].mean() for c in ISSUES],
            "People ≥ threshold": [int(worksheet[c].ge(issue_threshold).sum()) for c in ISSUES],
            "% ≥ threshold": [worksheet[c].ge(issue_threshold).mean() * 100 for c in ISSUES],
        }).sort_values("Mean Score", ascending=False).reset_index(drop=True)

        top3 = issue_summary.head(3)
        st.markdown(f"### STEP 1 — What are our top issues?  •  Threshold: {issue_threshold}")
        st.dataframe(top3.round(1), use_container_width=True, hide_index=True)

        top_issue = str(top3.iloc[0]["Issue"])
        top3_names = top3["Issue"].tolist()

        with st.expander("👤 View / download people behind a top issue"):
            selected_issue_for_export = st.selectbox(
                "Select issue",
                top3_names,
                key="worksheet_issue_people_export"
            )
            _issue_people_export(
                worksheet,
                selected_issue_for_export,
                issue_threshold,
                "worksheet_issue_people"
            )

        # Visual: top issue comparison
        st.markdown("#### 📊 Top Issues")
        fig_top = px.bar(
            top3,
            x="Issue",
            y="Mean Score",
            text_auto=".1f",
            title="Top 3 Issue Scores"
        )
        fig_top.update_yaxes(range=[0, 100], title="Score (0–100)")
        st.plotly_chart(fig_top, use_container_width=True)

        # STEP 2 — What occurs together?
        st.markdown(f"### STEP 2 — What occurs together?  •  Threshold: {issue_threshold}")
        combo = top_combinations(worksheet, threshold=issue_threshold, max_size=3)
        top_combo = combo.head(10)
        if top_combo.empty:
            st.info("No issue combinations meet the ≥50 threshold.")
            combo_text = ""
        else:
            st.dataframe(top_combo, use_container_width=True, hide_index=True)
            combo_text = str(top_combo.iloc[0]["Combination"])

        with st.expander("👤 View / download people behind a combination"):
            if not top_combo.empty:
                combo_options = top_combo["Combination"].astype(str).tolist()
                selected_combo_for_export = st.selectbox(
                    "Select combination",
                    combo_options,
                    key="worksheet_combo_people_export"
                )
                combo_parts = [
                    p.strip() for p in selected_combo_for_export.split(" + ")
                    if p.strip()
                ]
                if combo_parts:
                    mask = pd.Series(True, index=worksheet.index)
                    for issue_name in combo_parts:
                        if issue_name in worksheet.columns:
                            mask &= worksheet[issue_name].ge(issue_threshold)
                    combo_people = worksheet.loc[mask].copy()
                    show_people_export(
                        combo_people,
                        title=f"People behind: {selected_combo_for_export}",
                        key="worksheet_combo_people",
                        show_table=True
                    )

        # Visual: combination frequency
        if not top_combo.empty:
            fig_combo = px.bar(
                top_combo.head(10),
                x="Number of People",
                y="Combination",
                orientation="h",
                text_auto=True,
                title="Most Frequent Problem Combinations"
            )
            st.plotly_chart(fig_combo, use_container_width=True)

        # The selected Step 2 pattern is reused by Steps 3 and 4.
        pattern_issues = (
            [p.strip() for p in combo_text.split(" + ")]
            if combo_text else [top_issue]
        )

        # STEP 3 — Who experiences it?
        st.markdown(f"### STEP 3 — Who experiences it?  •  Threshold: {issue_threshold}")

        # Count the same respondents identified by the Step 2 pattern.
        # A respondent must meet the threshold for every issue in the pattern.
        pattern_hit = worksheet[pattern_issues].ge(issue_threshold).all(axis=1)
        pattern_count = int(pattern_hit.sum())

        st.info(
            f"Who among those {pattern_count} people identified in Step 2 "
            "experiences the selected pattern? The breakdown below is by "
            "age group, life stage, and urban/rural category."
        )
        st.caption(
            f"Pattern carried forward from Step 2: {' + '.join(pattern_issues)} "
            f"(each issue score ≥ {issue_threshold})"
        )
        group_results = _strongest_group(
            worksheet, pattern_issues, threshold=issue_threshold
        )
        if group_results.empty:
            st.info("No usable demographic/context grouping columns were found.")
            strongest_group_text = "No group available"
        else:
            st.dataframe(
                group_results.head(10),
                use_container_width=True,
                hide_index=True
            )
            best_group = group_results.iloc[0]
            strongest_group_text = (
                f"{best_group['Dimension']}: {best_group['Group']} "
                f"({int(best_group['People'])} people; "
                f"{best_group['% of Group']:.1f}% of that group)"
            )
            st.success(
                f"Highest threshold concentration: **{strongest_group_text}**"
            )

            with st.expander("👤 View / download people in a selected group"):
                selected_dimension = st.selectbox(
                    "Dimension",
                    group_results["Dimension"].drop_duplicates().tolist(),
                    key="worksheet_group_dimension_export"
                )
                available_groups = group_results.loc[
                    group_results["Dimension"] == selected_dimension, "Group"
                ].astype(str).tolist()
                selected_group = st.selectbox(
                    "Group",
                    available_groups,
                    key="worksheet_group_export"
                )
                dim_aliases = {
                    "Age Group": ["Q1"],
                    "Life Stage": ["Q2"],
                    "Community Participation": ["Community Participation", "Community_Participation", "CommunityParticipation"],
                    "Spiritual Stage": ["Spiritual Stage", "Spiritual_Stage", "SpiritualStage"],
                    "Urban / Rural": ["Urban_Rural", "Urban/Rural", "Urban Rural"],
                }
                group_col = _find_column(worksheet, dim_aliases.get(selected_dimension, []))
                if group_col:
                    group_people = worksheet[
                        worksheet[group_col].astype(str) == selected_group
                    ].copy()
                    show_people_export(
                        group_people,
                        title=f"People in {selected_dimension}: {selected_group}",
                        key="worksheet_group_people",
                        show_table=True
                    )

            # Visual: strongest demographic/context groups
            fig_group = px.bar(
                group_results.head(10),
                x="% of Group",
                y="Group",
                color="Dimension",
                orientation="h",
                text_auto=".1f",
                title="Groups experiencing the selected Step 2 pattern"
            )
            fig_group.update_xaxes(range=[0, 100], title="% meeting threshold")
            st.plotly_chart(fig_group, use_container_width=True)

        # Optional two-group comparison for the selected church/community.
        st.markdown("### 🔎 Optional — Compare Two Groups")
        st.caption(
            "Compare up to two groups within the current church/community scope "
            "to understand how the top issue varies."
        )
        compare_candidates_ws = []
        for label, aliases in [
            ("Age Group", ["Q1"]),
            ("Life Stage", ["Q2"]),
            ("Country", ["Country"]),
            ("Region / State", ["Region", "State"]),
            ("District", ["District"]),
            ("Community", ["Community_ID"]),
            ("Church", ["Church_ID"]),
            ("Urban / Rural", ["Urban_Rural", "Urban/Rural", "Urban Rural"]),
            ("Community Participation", ["Community Participation", "Community_Participation"]),
                ("Spiritual Stage", ["Spiritual Stage", "Spiritual_Stage"]),
        ]:
            found = _find_column(worksheet, aliases)
            if found and worksheet[found].dropna().nunique() >= 2:
                compare_candidates_ws.append((label, found))

        if compare_candidates_ws:
            compare_label_ws = st.selectbox(
                "Compare by",
                [x[0] for x in compare_candidates_ws],
                key="worksheet_compare_dimension"
            )
            compare_col_ws = dict(compare_candidates_ws)[compare_label_ws]
            values_ws = sorted(
                worksheet[compare_col_ws].dropna().astype(str).unique().tolist()
            )
            selected_groups_ws = st.multiselect(
                f"Select up to 2 {compare_label_ws} groups",
                values_ws,
                max_selections=2,
                key="worksheet_compare_groups"
            )
            if selected_groups_ws:
                rows_ws = []
                for group in selected_groups_ws:
                    sub = worksheet[worksheet[compare_col_ws].astype(str) == group]
                    rows_ws.append({
                        "Group": group,
                        "People": len(sub),
                        "Top Issue Score": round(float(sub[top_issue].mean()), 1) if len(sub) else float("nan"),
                        "People ≥ threshold": int(sub[top_issue].ge(issue_threshold).sum()) if len(sub) else 0,
                        "% ≥ threshold": round(float(sub[top_issue].ge(issue_threshold).mean() * 100), 1) if len(sub) else 0.0,
                    })
                ws_comp_df = pd.DataFrame(rows_ws)
                st.dataframe(ws_comp_df, use_container_width=True, hide_index=True)
                ws_comp_long = []
                for group in selected_groups_ws:
                    sub = worksheet[worksheet[compare_col_ws].astype(str) == group]
                    for issue in top3_names:
                        ws_comp_long.append({
                            "Group": group,
                            "Issue": issue,
                            "Score": float(sub[issue].mean()) if len(sub) else float("nan")
                        })
                ws_comp_long = pd.DataFrame(ws_comp_long)
                fig_ws_compare = px.bar(
                    ws_comp_long,
                    x="Issue",
                    y="Score",
                    color="Group",
                    barmode="group",
                    text_auto=".1f",
                    title=f"{compare_label_ws} — Top Issue Comparison"
                )
                fig_ws_compare.update_yaxes(range=[0, 100], title="Mean Score (0–100)")
                st.plotly_chart(fig_ws_compare, use_container_width=True)

        # STEP 4 — Where does it occur?
        st.markdown("### STEP 4 — Where does it occur?")
        st.caption(f"Pattern: {' + '.join(pattern_issues)} (score ≥ {issue_threshold})")
        loc = _location_summary(worksheet, pattern_issues, threshold=issue_threshold)
        if loc.empty:
            st.info("No geographic metadata is available.")
        else:
            st.dataframe(loc, use_container_width=True, hide_index=True)
            fig_loc = px.bar(
                loc,
                x="% in location",
                y="Location",
                color="Level",
                facet_row="Level",
                orientation="h",
                text="% in location",
                hover_data=["Rank", "People with pattern", "Respondents", "Overall %"],
                title="All locations ranked by pattern concentration",
            )
            fig_loc.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_loc.update_xaxes(range=[0, 100], title="% of respondents in location meeting the pattern")
            fig_loc.update_yaxes(title="Location", categoryorder="total ascending")
            fig_loc.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(fig_loc, use_container_width=True)

        # STEP 5 — Which survey questions explain the selected Step 2 pattern?
        st.markdown("### STEP 5 — Which survey questions explain the selected pattern?")
        st.caption(
            f"Underlying survey questions for the Step 2 pattern: "
            f"{' + '.join(pattern_issues)}"
        )
        question_frames = [
            _issue_question_rows(worksheet, issue, top_n=3)
            for issue in pattern_issues
        ]
        question_evidence = (
            pd.concat(question_frames, ignore_index=True)
            if question_frames else pd.DataFrame()
        )
        if question_evidence.empty:
            st.info("No underlying Likert question evidence is available.")
        else:
            # Add the complete readable question wording without changing
            # the existing scoring, ranking, charts or calculations.
            question_evidence.insert(
                1,
                "Full Survey Question",
                question_evidence["Question"].map(
                    lambda q: FULL_QUESTION_TEXT.get(
                        str(q),
                        "Full wording is not available in the current survey-question reference."
                    )
                )
            )

            st.markdown("#### Full survey questions supporting this pattern")
            st.dataframe(
                question_evidence,
                use_container_width=True,
                hide_index=True
            )
            st.caption(
                "The question ID, full wording, issue mapping and evidence score "
                "are shown together so leaders can trace each pattern back to the "
                "original survey questions. Evidence Score converts the underlying "
                "1–5 response mean to 0–100. Reverse-coded questions are already "
                "transformed so that higher values consistently represent greater "
                "concern/support need."
            )

            # Visual: underlying question evidence
            fig_q = px.bar(
                question_evidence.head(12),
                x="Question",
                y="Evidence Score (0–100)",
                color="Issue",
                text_auto=".1f",
                title="Underlying Survey Question Evidence"
            )
            fig_q.update_yaxes(range=[0, 100], title="Evidence Score (0–100)")
            st.plotly_chart(fig_q, use_container_width=True)

        # STEP 6 — What are people saying?
        st.markdown("### STEP 6 — What are people saying?")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Q41 — Most Significant Challenge**")
            q41 = _text_themes(worksheet, "Q41")
            if q41.empty:
                st.info("No Q41 responses available.")
                q41_theme = ""
            else:
                st.dataframe(q41, use_container_width=True, hide_index=True)
                q41_theme = str(q41.iloc[0]["Theme"])
                fig_q41 = px.bar(
                    q41,
                    x="Responses",
                    y="Theme",
                    orientation="h",
                    text_auto=True,
                    title="Q41 Challenge Themes"
                )
                st.plotly_chart(fig_q41, use_container_width=True)
        with c2:
            st.markdown("**Q42 — Most Helpful Help / Guidance / Support**")
            q42 = _text_themes(worksheet, "Q42")
            if q42.empty:
                st.info("No Q42 responses available.")
                q42_theme = ""
            else:
                st.dataframe(q42, use_container_width=True, hide_index=True)
                q42_theme = str(q42.iloc[0]["Theme"])
                fig_q42 = px.bar(
                    q42,
                    x="Responses",
                    y="Theme",
                    orientation="h",
                    text_auto=True,
                    title="Q42 Requested Support Themes"
                )
                st.plotly_chart(fig_q42, use_container_width=True)

        # STEP 7 — Write the pattern
        st.markdown("### STEP 7 — Write the pattern")
        top_supporting_qs = (
            ", ".join(question_evidence["Question"].astype(str).head(4).tolist())
            if not question_evidence.empty else "not available"
        )
        group_sentence = (
            strongest_group_text if strongest_group_text != "No group available"
            else "the available respondent population"
        )
        scope_name = (
            "the whole church/community population"
            if selected_community == "All Communities"
            else f"the Church / Community {selected_community}"
        )
        if combo_text:
            pattern_statement = (
                f"Among respondents in {scope_name}, "
                f"{combo_text} frequently occur together. "
                f"This is particularly visible in {group_sentence}. "
                f"Survey questions {top_supporting_qs} provide additional evidence, "
                f"while Q41/Q42 add the respondents' own expressed context."
            )
        else:
            pattern_statement = (
                f"Among respondents in {scope_name}, "
                f"{top_issue} is the highest observed need area. "
                f"This is particularly visible in {group_sentence}. "
                f"Survey questions {top_supporting_qs} provide additional evidence, "
                f"while Q41/Q42 add the respondents' own expressed context."
            )
        st.text_area(
            "Evidence-based pattern statement",
            value=pattern_statement,
            height=130,
            key="church_auto_pattern_statement"
        )

        # STEP 8 — What does this pattern mean for action?
        st.markdown("### STEP 8 — What does this pattern mean for action?")
        actions = _action_considerations(
            top_issue, combo_text, q41_theme, q42_theme
        )
        for i, action in enumerate(actions, 1):
            st.text_area(
                f"Possible action consideration {i}",
                value=action,
                height=80,
                key=f"church_action_{i}"
            )

        st.info(
            "These action considerations are generated from observed survey "
            "patterns. Leaders should validate them against local knowledge, "
            "pastoral context, safeguarding requirements and available support. "
            "The dashboard does not diagnose individuals or establish causation."
        )

        # One-click evidence export for the complete worksheet.
        export_rows = []
        for _, row in top3.iterrows():
            export_rows.append({
                "Section": "Step 1",
                "Item": row["Issue"],
                "Evidence": (
                    f"Mean={row['Mean Score']:.1f}; "
                    f"People≥threshold={int(row['People ≥ threshold'])}; "
                    f"%≥threshold={row['% ≥ threshold']:.1f}%"
                )
            })
        for _, row in top_combo.iterrows():
            export_rows.append({
                "Section": "Step 2",
                "Item": row["Combination"],
                "Evidence": (
                    f"{int(row['Number of People'])} people; "
                    f"{row['Percentage']:.1f}%"
                )
            })
        for _, row in question_evidence.iterrows():
            export_rows.append({
                "Section": "Step 5",
                "Item": row["Question"],
                "Evidence": (
                    f"{row['Issue']}; evidence score={row['Evidence Score (0–100)']:.1f}; "
                    f"agree/strongly agree={row['% Agree / Strongly Agree']:.1f}%"
                )
            })

        export_df = pd.DataFrame(export_rows)
        st.download_button(
            "⬇️ Download complete worksheet evidence (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="church_pattern_discovery_answers.csv",
            mime="text/csv",
            key="download_church_pattern_answers"
        )

        st.caption(
            "Pattern Check: supported by the selected responses; co-occurrence "
            "is not causation; underlying questions and Q41/Q42 are shown; "
            "privacy, consent and safeguarding remain the responsibility of the leader."
        )


# 10 Individual Profiles
with tabs[11]:
    st.subheader("Individual Problem Profiles")

    show_people_export(
        filtered,
        "People available for Individual Profiles",
        "church_individual_profiles",
        show_table=False
    )

    id_options = filtered["Person_ID"].astype(str).tolist()
    if not id_options:
        st.info("No respondents match the current filters.")
    else:
        selected_id = st.selectbox("Select Person_ID", id_options)
        person = filtered[filtered["Person_ID"].astype(str) == selected_id].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Primary Need Area", str(person["Primary Issue"]))
        c2.metric("Social & Community", f"{person['Social & Community Connection']:.1f}")
        c3.metric("Spiritual Growth", f"{person['Spiritual Growth & Discipleship']:.1f}")

        profile = pd.DataFrame({
            "Issue": ISSUES,
            "Score": [person[x] for x in ISSUES]
        })

        fig = px.bar(profile, x="Issue", y="Score", text_auto=".1f",
                     title="Individual Problem Profile")
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(profile.round(1), use_container_width=True, hide_index=True)

        if "Q41" in person.index:
            st.markdown("**Q41 — Most Significant Challenge**")
            st.write(person["Q41"])
        if "Q42" in person.index:
            st.markdown("**Q42 — Most Helpful Help / Guidance / Support**")
            st.write(person["Q42"])

# 13 Data
with tabs[12]:
    st.subheader("Analysed Data")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download analysed CSV",
        data=csv,
        file_name="community_insights_analysed.csv",
        mime="text/csv"
    )

st.divider()
st.caption("Prototype analytics only. Scores and thresholds should be validated with real survey data before research or operational use. Avoid diagnostic or stigmatizing labels.")
render_footer()
