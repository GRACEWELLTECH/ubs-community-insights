import streamlit as st
import pandas as pd
import re

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Question Number Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 16px;
        color: #666;
        margin-bottom: 25px;
    }

    .question-box {
        padding: 12px 18px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 8px;
        font-size: 18px;
        font-weight: 600;
    }

    .metric-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 Question Number Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Display and explore question numbers from your Excel dataset</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel File",
    type=["xlsx", "xls"]
)

# ============================================================
# HELPER FUNCTION
# ============================================================

def extract_question_number(value):
    """
    Extract question numbers such as:
    Q1
    Q12
    Q31
    Q100
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    match = re.search(
        r"\bQ\s*(\d+)\b",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return f"Q{match.group(1)}"

    return None


def question_sort_key(question):
    """
    Sort Q1, Q2, Q10, Q31 correctly.
    """

    match = re.search(r"\d+", question)

    if match:
        return int(match.group())

    return 999999


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Please upload your Excel file from the sidebar."
    )

    st.markdown("### Expected question format")

    example_questions = [
        "Q1",
        "Q2",
        "Q12",
        "Q14",
        "Q31",
        "Q32",
        "Q33",
        "Q34",
        "Q35",
        "Q36",
        "Q37"
    ]

    for q in example_questions:

        st.markdown(
            f'<div class="question-box">{q}</div>',
            unsafe_allow_html=True
        )

else:

    # ========================================================
    # READ EXCEL
    # ========================================================

    try:

        df = pd.read_excel(uploaded_file)

    except Exception as e:

        st.error(
            f"❌ Unable to read the Excel file.\n\n{e}"
        )

        st.stop()

    # ========================================================
    # BASIC DATA INFORMATION
    # ========================================================

    st.success("✅ Excel file loaded successfully")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            len(df)
        )

    with col2:
        st.metric(
            "Columns",
            len(df.columns)
        )

    with col3:
        st.metric(
            "File",
            uploaded_file.name
        )

    st.divider()

    # ========================================================
    # FIND QUESTION NUMBERS
    # ========================================================

    question_numbers = set()

    # --------------------------------------------------------
    # CHECK COLUMN NAMES
    # --------------------------------------------------------

    for column in df.columns:

        question = extract_question_number(column)

        if question:
            question_numbers.add(question)

    # --------------------------------------------------------
    # CHECK CELL VALUES
    # --------------------------------------------------------

    for column in df.columns:

        for value in df[column].dropna():

            question = extract_question_number(value)

            if question:
                question_numbers.add(question)

    # ========================================================
    # SORT QUESTIONS
    # ========================================================

    question_numbers = sorted(
        question_numbers,
        key=question_sort_key
    )

    # ========================================================
    # DISPLAY QUESTION COUNT
    # ========================================================

    st.subheader("📋 Questions")

    st.write(
        f"**Total Questions Found: {len(question_numbers)}**"
    )

    # ========================================================
    # SEARCH QUESTION
    # ========================================================

    search_text = st.text_input(
        "🔎 Search Question Number",
        placeholder="Example: Q31"
    )

    if search_text:

        search_text = search_text.strip().upper()

        filtered_questions = [
            q for q in question_numbers
            if search_text in q
        ]

    else:

        filtered_questions = question_numbers

    # ========================================================
    # DISPLAY ONLY QUESTION NUMBERS
    # ========================================================

    if filtered_questions:

        for question in filtered_questions:

            st.markdown(
                f'<div class="question-box">{question}</div>',
                unsafe_allow_html=True
            )

    else:

        st.warning(
            "No matching question numbers found."
        )

    # ========================================================
    # DOWNLOAD QUESTION NUMBERS
    # ========================================================

    if question_numbers:

        question_df = pd.DataFrame(
            {
                "Question Number": question_numbers
            }
        )

        csv_data = question_df.to_csv(
            index=False
        )

        st.divider()

        st.download_button(
            label="⬇️ Download Question Numbers",
            data=csv_data,
            file_name="question_numbers.csv",
            mime="text/csv"
        )

    # ========================================================
    # OPTIONAL: SHOW ORIGINAL DATA
    # ========================================================

    st.divider()

    show_data = st.checkbox(
        "Show original Excel data"
    )

    if show_data:

        st.dataframe(
            df,
            use_container_width=True,
            height=500
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Question Number Dashboard | Gracewell Technologies"
)
