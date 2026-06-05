import streamlit as st
import pandas as pd
import joblib

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea, #764ba2);
    }

    .block-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
    }

    h1, h2, h3, p {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Student Performance Rating Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Rating Predictor")
st.write("Enter student details and compare predictions from different ML models.")

# ---------------------------
# RESET FUNCTION (IMPORTANT FIX)
# ---------------------------
def reset_form():
    st.session_state.project_made = "Select"
    st.session_state.presentation = "Select"
    st.session_state.project_deployed = "Select"
    st.session_state.participation = "Select"
    st.session_state.discipline = "Select"

    st.session_state.lab_work = 0
    st.session_state.viva = 0
    st.session_state.collaboration = 0
    st.session_state.assignment = 0


# ---------------------------
# Initialize state ONLY ON FIRST LOAD
# ---------------------------
if "init" not in st.session_state:
    reset_form()
    st.session_state.init = True


# ---------------------------
# Reset Button (FORCES CLEAN STATE)
# ---------------------------
if st.button("🔄 Reset Form"):
    reset_form()
    st.rerun()


# ---------------------------
# Load Models
# ---------------------------
try:
    rf_model = joblib.load("random_forest_model.pkl")
    lr_model = joblib.load("linear_regression_model.pkl")
    encoder = joblib.load("encoder.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


# ---------------------------
# Input Fields (ALL CONTROLLED BY session_state)
# ---------------------------
project_made = st.selectbox(
    "Project Made",
    ["Select", "Yes", "No"],
    key="project_made"
)

presentation = st.selectbox(
    "Presentation Grade",
    ["Select", "A", "B", "C", "D", "E"],
    key="presentation"
)

lab_work = st.slider(
    "Lab Work Completion (0-12)",
    0, 12,
    key="lab_work"
)

project_deployed = st.selectbox(
    "Project Deployed",
    ["Select", "Yes", "No"],
    key="project_deployed"
)

viva = st.slider(
    "Viva Marks (0-5)",
    0, 5,
    key="viva"
)

participation = st.selectbox(
    "Class Participation",
    ["Select", "Active", "Partially Active", "Not Participating"],
    key="participation"
)

discipline = st.selectbox(
    "Discipline Score",
    ["Select", "Excellent", "Good", "Average", "Below Average"],
    key="discipline"
)

collaboration = st.slider(
    "Collaboration Score (0-10)",
    0, 10,
    key="collaboration"
)

assignment = st.slider(
    "Assignment Quality (0-5)",
    0, 5,
    key="assignment"
)


# ---------------------------
# Prediction Button
# ---------------------------
if st.button("Predict Rating"):

    if (
        project_made == "Select"
        or presentation == "Select"
        or project_deployed == "Select"
        or participation == "Select"
        or discipline == "Select"
    ):
        st.error("⚠ Please select all dropdown fields before prediction.")
        st.stop()

    # Create DataFrame
    input_df = pd.DataFrame(
        [[
            project_made,
            presentation,
            project_deployed,
            participation,
            discipline,
            lab_work,
            viva,
            collaboration,
            assignment
        ]],
        columns=[
            "Project_Made",
            "Presentation_Grade",
            "Project_Deployed",
            "Class_Participation",
            "Discipline_Score",
            "Lab_Work_Completion",
            "Viva_Marks",
            "Collaboration_Score",
            "Assignment_Quality"
        ]
    )

    categorical_cols = [
        "Project_Made",
        "Presentation_Grade",
        "Project_Deployed",
        "Class_Participation",
        "Discipline_Score"
    ]

    numeric_cols = [
        "Lab_Work_Completion",
        "Viva_Marks",
        "Collaboration_Score",
        "Assignment_Quality"
    ]

    # Encode categorical data
    encoded = encoder.transform(input_df[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    final_input = pd.concat(
        [input_df[numeric_cols], encoded_df],
        axis=1
    )

    # Predictions
    rf_prediction = rf_model.predict(final_input)[0]
    lr_prediction = lr_model.predict(final_input)[0]

    rf_prediction = max(0, min(10, rf_prediction))
    lr_prediction = max(0, min(10, lr_prediction))

    final_prediction = (rf_prediction + lr_prediction) / 2

    # Results
    st.subheader("📊 Prediction Results")

    st.success(f"🌲 Random Forest Rating: {rf_prediction:.2f}/10")
    st.info(f"📈 Linear Regression Rating: {lr_prediction:.2f}/10")
    st.warning(f"⭐ Final Average Rating: {final_prediction:.2f}/10")

    st.metric(
        "Difference Between Models",
        f"{abs(rf_prediction - lr_prediction):.2f}"
    )