import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(
    page_title="Student Performance Rating Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Rating Predictor")

# Load Models
try:
    rf_model = joblib.load("random_forest_model.pkl")
    lr_model = joblib.load("linear_regression_model.pkl")
    encoder = joblib.load("encoder.pkl")
except Exception as e:
    st.error("Error loading model files")
    st.code(str(e))
    st.stop()

# Input Fields
project_made = st.selectbox(
    "Project Made",
    ["Yes", "No"]
)

presentation = st.selectbox(
    "Presentation Grade",
    ["A", "B", "C", "D", "E"]
)

lab_work = st.slider(
    "Lab Work Completion",
    0,
    12,
    6
)

project_deployed = st.selectbox(
    "Project Deployed",
    ["Yes", "No"]
)

viva = st.slider(
    "Viva Marks",
    0,
    5,
    2
)

participation = st.selectbox(
    "Class Participation",
    ["Active", "Partially Active", "Not Participating"]
)

discipline = st.selectbox(
    "Discipline Score",
    ["Excellent", "Good", "Average", "Below Average"]
)

collaboration = st.slider(
    "Collaboration Score",
    0,
    10,
    5
)

assignment = st.slider(
    "Assignment Quality",
    0,
    5,
    2
)

# Prediction
if st.button("Predict Rating"):

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

    encoded = encoder.transform(
        input_df[categorical_cols]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    final_input = pd.concat(
        [input_df[numeric_cols], encoded_df],
        axis=1
    )

    rf_prediction = rf_model.predict(final_input)[0]
    lr_prediction = lr_model.predict(final_input)[0]

    # Keep score in range
    rf_prediction = max(0, min(10, rf_prediction))
    lr_prediction = max(0, min(10, lr_prediction))

    st.subheader("📊 Prediction Results")

    st.success(
        f"🌲 Random Forest Rating: {rf_prediction:.2f}/10"
    )

    st.info(
        f"📈 Linear Regression Rating: {lr_prediction:.2f}/10"
    )

    st.divider()

    st.metric(
        label="Difference Between Models",
        value=f"{abs(rf_prediction - lr_prediction):.2f}"
    )