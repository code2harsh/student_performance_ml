import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv("student_performance_dataset.csv")

categorical_cols = ["Project_Made", "Presentation_Grade", "Project_Deployed",
                    "Class_Participation", "Discipline_Score"]
numeric_cols = ["Lab_Work_Completion", "Viva_Marks", "Collaboration_Score", "Assignment_Quality"]

encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

X = pd.concat([df[numeric_cols], encoded_df], axis=1)
y = df["Rating"]

# Train/Val/Test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
print("Validation MAE:", mean_absolute_error(y_val, y_val_pred))
print("Validation R²:", r2_score(y_val, y_val_pred))

y_test_pred = model.predict(X_test)
print("Test MAE:", mean_absolute_error(y_test, y_test_pred))
print("Test R²:", r2_score(y_test, y_test_pred))

joblib.dump(model, "student_rating_model.pkl")
joblib.dump(encoder, "encoder.pkl")
print("✅ Model and encoder saved")


perfect_student = pd.DataFrame([{
    "Project_Made": "Yes",
    "Presentation_Grade": "A",
    "Project_Deployed": "Yes",
    "Class_Participation": "Active",
    "Discipline_Score": "Excellent",
    "Lab_Work_Completion": 12,
    "Viva_Marks": 5,
    "Collaboration_Score": 10,
    "Assignment_Quality": 5
}])