import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Question Number Dashboard",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# QUESTION NUMBER AND CATEGORY MAPPING
# ============================================================

QUESTION_CATEGORIES = {
    "Q7": "Personal & Emotional",
    "Q8": "Personal & Emotional",
    "Q9": "Personal & Emotional",
    "Q10": "Personal & Emotional",
    "Q11": "Personal & Emotional",

    "Q12": "Family & Relationships",
    "Q13": "Family & Relationships",
    "Q14": "Family & Relationships",
    "Q15": "Family & Relationships",
    "Q16": "Family & Relationships",

    "Q17": "Employment & Career",
    "Q18": "Employment & Career",
    "Q19": "Employment & Career",
    "Q20": "Employment & Career",
    "Q21": "Employment & Career",

    "Q22": "Financial Wellbeing",
    "Q23": "Financial Wellbeing",
    "Q24": "Financial Wellbeing",
    "Q25": "Financial Wellbeing",

    "Q26": "Physical Health & Wellbeing",
    "Q27": "Physical Health & Wellbeing",
    "Q28": "Physical Health & Wellbeing",

    "Q29": "Emotional Wellbeing & Support",
    "Q30": "Emotional Wellbeing & Support",
    "Q31": "Emotional Wellbeing & Support",

    "Q32": "Community & Social Connection",
    "Q33": "Community & Social Connection",
    "Q34": "Community & Social Connection",

    "Q35": "Spiritual Growth",
    "Q36": "Spiritual Growth",
    "Q37": "Spiritual Growth",
    "Q38": "Spiritual Growth",
    "Q39": "Spiritual Growth",

    "Q40": "Priorities",
    "Q41": "Open Text",
    "Q42": "Open Text"
}

# ============================================================
# QUESTION LIST
# ============================================================

QUESTION_NUMBERS = list(QUESTION_CATEGORIES.keys())

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.question-box {
    padding: 12px 18px;
    margin: 6px 0;
    border: 1px solid #dddddd;
    border-radius: 8px;
    font-size: 18px;
    font-weight: 600;
}

.category-box {
    padding: 12px 18px;
    margin: 6px 0;
    border: 1px solid #dddddd;
    border-radius: 8px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.title("📋 Question Number Dashboard")

st.caption(
    "Question categories are stored internally. "
    "Only question numbers are displayed."
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Controls")

search_text = st.sidebar.text_input(
    "Search Question Number",
    placeholder="Example: Q31"
)

show_categories = st.sidebar.checkbox(
    "Show category information",
    value=False
)

# ============================================================
# CATEGORY FILTER
# ============================================================

all_categories = [
    "All Categories"
] + list(dict.fromkeys(QUESTION_CATEGORIES.values()))

selected_category = st.sidebar.selectbox(
    "Filter by Category",
    all_categories
)

# ============================================================
# FILTER QUESTIONS
# ============================================================

filtered_questions = QUESTION_NUMBERS.copy()

if search_text:

    filtered_questions = [
        question
        for question in filtered_questions
        if search_text.strip().upper() in question
    ]

if selected_category != "All Categories":

    filtered_questions = [
        question
        for question in filtered_questions
        if QUESTION_CATEGORIES[question] == selected_category
    ]

# ============================================================
# SUMMARY METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Questions",
        len(QUESTION_NUMBERS)
    )

with col2:
    st.metric(
        "Displayed Questions",
        len(filtered_questions)
    )

with col3:
    st.metric(
        "Categories",
        len(set(QUESTION_CATEGORIES.values()))
    )

st.divider()

# ============================================================
# DISPLAY QUESTIONS
# ============================================================

st.subheader("Available Questions")

if filtered_questions:

    for question in filtered_questions:

        # Display only question number by default
        st.markdown(
            f"""
            <div class="question-box">
                {question}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Category is shown only if checkbox is enabled
        if show_categories:

            st.caption(
                f"Category: {QUESTION_CATEGORIES[question]}"
            )

else:

    st.warning(
        "No matching question numbers found."
    )

# ============================================================
# QUESTION SELECTION
# ============================================================

st.divider()

st.subheader("Select a Question")

if filtered_questions:

    selected_question = st.selectbox(
        "Question Number",
        options=filtered_questions
    )

    st.success(
        f"Selected Question: {selected_question}"
    )

    # Category remains available internally
    selected_category_value = QUESTION_CATEGORIES[
        selected_question
    ]

    if show_categories:

        st.info(
            f"Category: {selected_category_value}"
        )

else:

    st.info("No question available for selection.")

# ============================================================
# CATEGORY SUMMARY
# ============================================================

st.divider()

st.subheader("Category Summary")

category_summary = {}

for category in QUESTION_CATEGORIES.values():

    category_summary[category] = (
        category_summary.get(category, 0) + 1
    )

for category, count in category_summary.items():

    st.write(f"**{category}:** {count} questions")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Gracewell Technologies"
)
