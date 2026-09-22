"""
Unified Master Community Insights Dashboard
Standalone Streamlit application
- No Excel upload required
- Uses fixed question IDs
- Includes an "All Churches" option in the Church / Community filter
- Demonstration data is generated internally
- Replace build_demo_data() later with your database/API function if required

Run:
    pip install streamlit
    streamlit run church_dashboard_app.py
"""

import csv
import io
import random
from datetime import datetime

import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Unified Master Community Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# FIXED QUESTION CATALOG
# ---------------------------------------------------------

QUESTION_CATALOG = {
    "Q1": "Age",
    "Q2": "Gender",
    "Q3": "Country",
    "Q4": "State / Province",
    "Q5": "District",
    "Q6": "Church / Community",
    "Q7": "Life Stage",
    "Q8": "Community Participation",
    "Q9": "Primary Need",
    "Q10": "Secondary Need",
    "Q11": "Preferred Support",
    "Q12": "Financial & Economic",
    "Q13": "Employment / Livelihood",
    "Q14": "Emotional Wellbeing & Support",
    "Q15": "Family & Relationships",
    "Q16": "Education & Skills",
    "Q17": "Health & Wellness",
    "Q18": "Spiritual Growth",
    "Q19": "Bible Engagement",
    "Q20": "Prayer & Fellowship",
    "Q21": "Youth Participation",
    "Q22": "Children's Ministry",
    "Q23": "Women's Ministry",
    "Q24": "Men's Ministry",
    "Q25": "Elderly Support",
    "Q26": "Digital Access",
    "Q27": "Preferred Communication",
    "Q28": "Preferred Language",
    "Q29": "Scripture Format",
    "Q30": "Engagement Frequency",
    "Q31": "Current Challenge",
    "Q32": "Urgency Level",
    "Q33": "Desired Change",
    "Q34": "Barriers to Participation",
    "Q35": "Trusted Support Person",
    "Q36": "Follow-up Preference",
    "Q37": "Transformation Goal",
    "Q38": "Response Status",
    "Q39": "Follow-up Status",
    "Q40": "Support Provided",
    "Q41": "Outcome",
    "Q42": "Additional Comments",
}

QUESTION_GROUPS = {
    "Demographics": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"],
    "Needs Assessment": ["Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17"],
    "Spiritual Engagement": ["Q18", "Q19", "Q20", "Q21", "Q22", "Q23", "Q24", "Q25"],
    "Communication & Scripture": ["Q26", "Q27", "Q28", "Q29", "Q30"],
    "Patterns & Transformation": ["Q31", "Q32", "Q33", "Q34", "Q35", "Q36", "Q37"],
    "Follow-up & Outcomes": ["Q38", "Q39", "Q40", "Q41", "Q42"],
}


# ---------------------------------------------------------
# DEMONSTRATION DATA
# ---------------------------------------------------------

def build_demo_data(number_of_records=600):
    """Generate internal demonstration data without Excel or pandas."""

    random.seed(42)

    churches = [
        "L001-C01",
        "L001-C02",
        "L001-C03",
        "L002-C01",
        "L002-C02",
        "L003-C01",
        "L003-C02",
        "L004-C01",
    ]

    countries = ["India", "Nepal", "Sri Lanka", "Bangladesh"]
    states = {
        "India": ["Tamil Nadu", "Kerala", "Karnataka"],
        "Nepal": ["Bagmati", "Gandaki", "Lumbini"],
        "Sri Lanka": ["Western", "Central", "Southern"],
        "Bangladesh": ["Dhaka", "Chattogram", "Rajshahi"],
    }

    ages = ["13–17", "18–25", "26–35", "36–50", "51–65", "66+"]
    genders = ["Male", "Female", "Prefer not to say"]
    life_stages = [
        "Children",
        "Youth",
        "Young Adults",
        "Adults",
        "Parents",
        "Elders",
    ]
    participation = ["Low", "Moderate", "High"]
    primary_needs = [
        "Spiritual Growth",
        "Emotional Wellbeing & Support",
        "Financial & Economic",
        "Family & Relationships",
        "Education & Skills",
        "Health & Wellness",
    ]
    secondary_needs = [
        "Financial & Economic",
        "Emotional Wellbeing & Support",
        "Family & Relationships",
        "Education & Skills",
        "Spiritual Growth",
        "Community Belonging",
    ]
    support_options = [
        "Bible Study",
        "Prayer Support",
        "Counselling",
        "Training",
        "Financial Guidance",
        "Community Fellowship",
    ]
    urgency_levels = ["Low", "Medium", "High", "Critical"]
    response_statuses = ["New", "Reviewed", "Contacted", "Completed"]
    followup_statuses = ["Pending", "In Progress", "Completed", "Not Required"]
    outcomes = [
        "No outcome recorded",
        "Connected to support",
        "Participated in Bible engagement",
        "Received counselling",
        "Joined fellowship",
        "Needs further follow-up",
    ]

    records = []

    for index in range(number_of_records):
        country = random.choice(countries)
        state = random.choice(states[country])
        church = random.choice(churches)

        primary_need = random.choice(primary_needs)
        secondary_need = random.choice(secondary_needs)

        record = {
            "Person_ID": f"P{index + 1:05d}",
            "Church_ID": church,
            "Country": country,
            "State": state,
            "District": f"District {random.randint(1, 8)}",
            "Age": random.choice(ages),
            "Gender": random.choice(genders),
            "Life_Stage": random.choice(life_stages),
            "Community_Participation": random.choice(participation),
            "Primary_Issue": primary_need,
            "Secondary_Issue": secondary_need,
            "Preferred_Support": random.choice(support_options),
            "Urgency": random.choice(urgency_levels),
            "Response_Status": random.choice(response_statuses),
            "Followup_Status": random.choice(followup_statuses),
            "Support_Provided": random.choice(support_options),
            "Outcome": random.choice(outcomes),
            "Q12": random.choice(["Yes", "No", "Sometimes"]),
            "Q14": random.choice(["Yes", "No", "Needs support"]),
            "Q31": primary_need,
            "Q32": random.choice(urgency_levels),
            "Q33": random.choice(
                [
                    "Greater hope",
                    "Improved family relationships",
                    "Spiritual maturity",
                    "Financial stability",
                    "Emotional healing",
                ]
            ),
            "Q34": random.choice(
                [
                    "Time",
                    "Distance",
                    "Fear",
                    "Lack of information",
                    "Financial limitations",
                    "No major barrier",
                ]
            ),
            "Q35": random.choice(
                [
                    "Church leader",
                    "Family member",
                    "Friend",
                    "Counsellor",
                    "No one currently",
                ]
            ),
            "Q36": random.choice(
                [
                    "Phone call",
                    "WhatsApp",
                    "In-person meeting",
                    "Email",
                    "No follow-up",
                ]
            ),
            "Q37": random.choice(
                [
                    "Hope",
                    "Healing",
                    "Belonging",
                    "Faith development",
                    "Purpose",
                    "Restoration",
                ]
            ),
            "Q38": random.choice(response_statuses),
            "Q39": random.choice(followup_statuses),
            "Q40": random.choice(support_options),
            "Q41": random.choice(outcomes),
            "Q42": "",
        }

        records.append(record)

    return records


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def unique_values(records, field):
    return sorted(
        {
            str(row.get(field, "")).strip()
            for row in records
            if str(row.get(field, "")).strip()
        }
    )


def filter_records(records, filters):
    filtered = []

    for row in records:
        matches = True

        for field, selected_value in filters.items():
            if selected_value in (None, "", "All", "All Churches", "All Countries"):
                continue

            if str(row.get(field, "")) != str(selected_value):
                matches = False
                break

        if matches:
            filtered.append(row)

    return filtered


def count_by(records, field):
    counts = {}

    for row in records:
        value = str(row.get(field, "Not specified")).strip()
        if not value:
            value = "Not specified"
        counts[value] = counts.get(value, 0) + 1

    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def percentage(part, total):
    if total == 0:
        return 0
    return round((part / total) * 100, 1)


def records_to_csv(records):
    if not records:
        return ""

    output = io.StringIO()
    fieldnames = list(records[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()


def show_distribution(title, records, field):
    st.subheader(title)

    distribution = count_by(records, field)

    if not distribution:
        st.info("No records available for this selection.")
        return

    total = sum(distribution.values())

    for label, count in distribution.items():
        st.write(f"**{label}** — {count} ({percentage(count, total)}%)")
        st.progress(count / total)


def unavailable_analytics(title, explanation):
    st.warning(f"### {title}")
    st.write(explanation)
    st.caption(
        "This section is intentionally disabled because this standalone version "
        "does not connect to a live database or uploaded response dataset."
    )


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

if "records" not in st.session_state:
    st.session_state.records = build_demo_data(600)

records = st.session_state.records


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("📊 Community Insights")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Question Explorer",
        "Church Intelligence",
        "National Intelligence",
        "State / District Intelligence",
        "Pattern Discovery",
        "Segment Profiles",
        "Cross-Analysis",
        "Individual Support",
        "Follow-up & Outcomes",
        "Export Records",
        "Analytics Status",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("Filters")

country_options = ["All Countries"] + unique_values(records, "Country")
selected_country = st.sidebar.selectbox(
    "Country",
    country_options,
    index=0,
)

# IMPORTANT: All Churches is the first option.
church_options = ["All Churches"] + unique_values(records, "Church_ID")
selected_church = st.sidebar.selectbox(
    "Church / Community",
    church_options,
    index=0,
    help="Select All Churches to include respondents from every church.",
)

state_options = ["All"] + unique_values(records, "State")
selected_state = st.sidebar.selectbox(
    "State / Province",
    state_options,
    index=0,
)

life_stage_options = ["All"] + unique_values(records, "Life_Stage")
selected_life_stage = st.sidebar.selectbox(
    "Life Stage",
    life_stage_options,
    index=0,
)

filters = {
    "Country": selected_country,
    "Church_ID": selected_church,
    "State": selected_state,
    "Life_Stage": selected_life_stage,
}

filtered_records = filter_records(records, filters)

st.sidebar.caption(
    f"Showing {len(filtered_records)} of {len(records)} respondents"
)

if selected_church == "All Churches":
    st.sidebar.success("All Churches selected")


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("Unified Master Community Insights Dashboard")
st.caption(
    "Standalone demonstration version • Fixed question IDs • No Excel dependency"
)

st.divider()


# ---------------------------------------------------------
# DASHBOARD OVERVIEW
# ---------------------------------------------------------

if page == "Dashboard Overview":
    st.header("Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Respondents", len(filtered_records))

    with col2:
        st.metric(
            "Churches Represented",
            len(unique_values(filtered_records, "Church_ID")),
        )

    with col3:
        st.metric(
            "Countries",
            len(unique_values(filtered_records, "Country")),
        )

    with col4:
        st.metric(
            "High / Critical Urgency",
            sum(
                1
                for row in filtered_records
                if row.get("Urgency") in ["High", "Critical"]
            ),
        )

    st.divider()

    left, right = st.columns(2)

    with left:
        show_distribution(
            "Primary Issues",
            filtered_records,
            "Primary_Issue",
        )

    with right:
        show_distribution(
            "Life Stage Distribution",
            filtered_records,
            "Life_Stage",
        )

    st.divider()

    show_distribution(
        "Response Status",
        filtered_records,
        "Response_Status",
    )


# ---------------------------------------------------------
# QUESTION EXPLORER
# ---------------------------------------------------------

elif page == "Question Explorer":
    st.header("Question Explorer")

    group = st.selectbox(
        "Select Question Group",
        ["All Groups"] + list(QUESTION_GROUPS.keys()),
    )

    if group == "All Groups":
        question_ids = list(QUESTION_CATALOG.keys())
    else:
        question_ids = QUESTION_GROUPS[group]

    search_text = st.text_input(
        "Search question ID or question name",
        placeholder="Example: Q12 or Financial",
    ).lower()

    rows = []

    for question_id in question_ids:
        question_name = QUESTION_CATALOG[question_id]

        if (
            not search_text
            or search_text in question_id.lower()
            or search_text in question_name.lower()
        ):
            rows.append(
                {
                    "Question ID": question_id,
                    "Question": question_name,
                    "Group": next(
                        (
                            group_name
                            for group_name, ids in QUESTION_GROUPS.items()
                            if question_id in ids
                        ),
                        "Uncategorized",
                    ),
                }
            )

    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.info(
        "The question catalog is available without Excel. "
        "Actual response-level analysis requires a connected response dataset."
    )


# ---------------------------------------------------------
# CHURCH INTELLIGENCE
# ---------------------------------------------------------

elif page == "Church Intelligence":
    st.header("Church Intelligence")

    if selected_church == "All Churches":
        st.info(
            "All Churches is selected. The summaries below combine respondents "
            "from every church."
        )
    else:
        st.info(f"Showing church-level information for {selected_church}.")

    col1, col2 = st.columns(2)

    with col1:
        show_distribution(
            "Primary Issues by Selected Church Scope",
            filtered_records,
            "Primary_Issue",
        )

    with col2:
        show_distribution(
            "Urgency Distribution",
            filtered_records,
            "Urgency",
        )

    show_distribution(
        "Preferred Support",
        filtered_records,
        "Preferred_Support",
    )


# ---------------------------------------------------------
# NATIONAL INTELLIGENCE
# ---------------------------------------------------------

elif page == "National Intelligence":
    st.header("National Intelligence")

    show_distribution(
        "Country Distribution",
        filtered_records,
        "Country",
    )

    show_distribution(
        "State / Province Distribution",
        filtered_records,
        "State",
    )

    show_distribution(
        "Primary Issues Across Current Scope",
        filtered_records,
        "Primary_Issue",
    )


# ---------------------------------------------------------
# STATE / DISTRICT INTELLIGENCE
# ---------------------------------------------------------

elif page == "State / District Intelligence":
    st.header("State / District Intelligence")

    show_distribution(
        "State / Province Distribution",
        filtered_records,
        "State",
    )

    show_distribution(
        "District Distribution",
        filtered_records,
        "District",
    )

    st.info(
        "District-level comparisons are descriptive only in this standalone version. "
        "Advanced statistical testing is not enabled."
    )


# ---------------------------------------------------------
# PATTERN DISCOVERY
# ---------------------------------------------------------

elif page == "Pattern Discovery":
    st.header("Pattern Discovery")

    unavailable_analytics(
        "AI Pattern Discovery Not Connected",
        "The original AI-driven pattern discovery workflow requires response data, "
        "the eight pattern-discovery prompts, and an AI model or processing pipeline. "
        "This no-Excel version displays the framework but does not invent analytical findings.",
    )

    st.subheader("Fixed Pattern Discovery Inputs")

    pattern_questions = [
        "Q12",
        "Q14",
        "Q31",
        "Q32",
        "Q33",
        "Q34",
        "Q35",
        "Q36",
        "Q37",
    ]

    for question_id in pattern_questions:
        st.write(
            f"**{question_id} — {QUESTION_CATALOG.get(question_id, 'Not defined')}**"
        )


# ---------------------------------------------------------
# SEGMENT PROFILES
# ---------------------------------------------------------

elif page == "Segment Profiles":
    st.header("Segment Profiles")

    segment = st.selectbox(
        "Choose segment dimension",
        ["Age", "Life Stage", "Gender", "Community Participation"],
    )

    field_map = {
        "Age": "Age",
        "Life Stage": "Life_Stage",
        "Gender": "Gender",
        "Community Participation": "Community_Participation",
    }

    show_distribution(
        f"{segment} Distribution",
        filtered_records,
        field_map[segment],
    )

    unavailable_analytics(
        "Persona Generation",
        "Detailed persona generation, needs interpretation, biblical emphasis, "
        "and transformation recommendations require a validated analytical workflow.",
    )


# ---------------------------------------------------------
# CROSS-ANALYSIS
# ---------------------------------------------------------

elif page == "Cross-Analysis":
    st.header("Cross-Analysis")

    first_dimension = st.selectbox(
        "First dimension",
        ["Age", "Life Stage", "Gender", "Country", "State", "Primary Issue"],
    )

    second_dimension = st.selectbox(
        "Second dimension",
        [
            "Secondary Issue",
            "Primary Issue",
            "Urgency",
            "Community Participation",
            "Response Status",
        ],
    )

    st.info(
        f"Requested cross-analysis: **{first_dimension} × {second_dimension}**"
    )

    unavailable_analytics(
        "Cross-Tabulation Disabled",
        "A full cross-tabulation engine requires structured response data and "
        "a defined missing-value and aggregation policy. This standalone version "
        "does not present fabricated cross-analysis results.",
    )


# ---------------------------------------------------------
# INDIVIDUAL SUPPORT
# ---------------------------------------------------------

elif page == "Individual Support":
    st.header("Individual Support")

    search_id = st.text_input(
        "Search Person ID",
        placeholder="Example: P00001",
    ).strip()

    if search_id:
        matches = [
            row
            for row in filtered_records
            if row.get("Person_ID", "").lower() == search_id.lower()
        ]

        if matches:
            person = matches[0]
            st.success(f"Person found: {person['Person_ID']}")

            for key, value in person.items():
                st.write(f"**{key}:** {value}")
        else:
            st.warning("Person ID not found in the current filtered scope.")
    else:
        st.info("Enter a Person ID to view the individual profile.")


# ---------------------------------------------------------
# FOLLOW-UP & OUTCOMES
# ---------------------------------------------------------

elif page == "Follow-up & Outcomes":
    st.header("Follow-up & Outcomes")

    col1, col2 = st.columns(2)

    with col1:
        show_distribution(
            "Follow-up Status",
            filtered_records,
            "Followup_Status",
        )

    with col2:
        show_distribution(
            "Recorded Outcomes",
            filtered_records,
            "Outcome",
        )

    show_distribution(
        "Support Provided",
        filtered_records,
        "Support_Provided",
    )


# ---------------------------------------------------------
# EXPORT RECORDS
# ---------------------------------------------------------

elif page == "Export Records":
    st.header("Export Records")

    st.write(
        f"Records available for export: **{len(filtered_records)}**"
    )

    if filtered_records:
        csv_data = records_to_csv(filtered_records)

        st.download_button(
            label="⬇️ Download Complete Filtered Records",
            data=csv_data,
            file_name="filtered_community_records.csv",
            mime="text/csv",
        )

        preview_count = st.slider(
            "Preview number of records",
            min_value=5,
            max_value=min(100, len(filtered_records)),
            value=min(10, len(filtered_records)),
        )

        st.dataframe(
            filtered_records[:preview_count],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No records available for the selected filters.")


# ---------------------------------------------------------
# ANALYTICS STATUS
# ---------------------------------------------------------

elif page == "Analytics Status":
    st.header("Analytics Availability")

    status_rows = [
        ("Fixed question catalog", "Available"),
        ("All Churches filter", "Available"),
        ("Country filtering", "Available"),
        ("State filtering", "Available"),
        ("Life-stage filtering", "Available"),
        ("Descriptive distributions", "Available"),
        ("CSV export", "Available"),
        ("Live Excel connection", "Not connected"),
        ("Live database connection", "Not connected"),
        ("AI-generated pattern interpretation", "Disabled"),
        ("Advanced statistical testing", "Disabled"),
        ("Predictive modelling", "Disabled"),
        ("Validated persona generation", "Disabled"),
        ("Automated Scripture recommendation", "Disabled"),
        ("Longitudinal outcome analysis", "Disabled"),
    ]

    st.table(
        [
            {"Feature": feature, "Status": status}
            for feature, status in status_rows
        ]
    )

    st.warning(
        "Important: This application contains internally generated demonstration "
        "records. Do not use the demonstration results for ministry decisions, "
        "research conclusions, or official reporting."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()
st.caption(
    f"Gracewell Technologies • Standalone Dashboard • "
    f"Generated {datetime.now().strftime('%d %B %Y, %I:%M %p')}"
)
