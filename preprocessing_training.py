import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("student_performance_dataset.csv")

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

# Encoding
encoder = OneHotEncoder(drop="first", sparse_output=False)

encoded = encoder.fit_transform(df[categorical_cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(categorical_cols)
)

X = pd.concat([df[numeric_cols], encoded_df], axis=1)
y = df["Rating"]

# Train/Test Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# ==========================
# Random Forest
# ==========================

rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n===== RANDOM FOREST =====")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("R² :", r2_score(y_test, rf_pred))

# ==========================
# Linear Regression
# ==========================

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\n===== LINEAR REGRESSION =====")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("R² :", r2_score(y_test, lr_pred))

# Save models

joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(lr_model, "linear_regression_model.pkl")
joblib.dump(encoder, "encoder.pkl")

print("\n✅ Models Saved Successfully")