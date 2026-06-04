import streamlit as st
import pandas as pd
import joblib

model = joblib.load("student_rating_model.pkl")
encoder = joblib.load("encoder.pkl")

st.title("🎓 Student Performance Rating Predictor")

project_made = st.selectbox("Project Made", ["Yes", "No"])
presentation = st.selectbox("Presentation Grade", ["A","B","C","D","E"])
lab_work = st.slider("Lab Work Completion (0-12)", 0, 12)
project_deployed = st.selectbox("Project Deployed", ["Yes", "No"])
viva = st.slider("Viva Marks (0-5)", 0, 5)
participation = st.selectbox("Class Participation", ["Active","Partially Active","Not Participating"])
discipline = st.selectbox("Discipline Score", ["Excellent","Good","Average","Below Average"])
collaboration = st.slider("Collaboration Score (0-10)", 0, 10)
assignment = st.slider("Assignment Quality (0-5)", 0, 5)

if st.button("Predict Rating"):
    input_df = pd.DataFrame([[project_made, presentation, project_deployed,
                              participation, discipline, lab_work, viva,
                              collaboration, assignment]],
                            columns=["Project_Made","Presentation_Grade","Project_Deployed",
                                     "Class_Participation","Discipline_Score",
                                     "Lab_Work_Completion","Viva_Marks",
                                     "Collaboration_Score","Assignment_Quality"])
    
    categorical_cols = ["Project_Made","Presentation_Grade","Project_Deployed",
                        "Class_Participation","Discipline_Score"]
    numeric_cols = ["Lab_Work_Completion","Viva_Marks","Collaboration_Score","Assignment_Quality"]

    encoded = encoder.transform(input_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    final_input = pd.concat([input_df[numeric_cols], encoded_df], axis=1)

    prediction = model.predict(final_input)[0]
    st.success(f"Predicted Rating: {prediction:.2f}/10")
