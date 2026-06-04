import pandas as pd
import random
import os

project_made = ["Yes", "No"]
presentation_grade = ["A", "B", "C", "D", "E"]
project_deployed = ["Yes", "No"]
class_participation = ["Active", "Partially Active", "Not Participating"]
discipline_score = ["Excellent", "Good", "Average", "Below Average"]

def generate_row():
    row = {
        "Project_Made": random.choice(project_made),
        "Presentation_Grade": random.choice(presentation_grade),
        "Lab_Work_Completion": random.randint(0, 12),
        "Project_Deployed": random.choice(project_deployed),
        "Viva_Marks": random.randint(0, 5),
        "Class_Participation": random.choice(class_participation),
        "Discipline_Score": random.choice(discipline_score),
        "Collaboration_Score": random.randint(0, 10),
        "Assignment_Quality": random.randint(0, 5),
    }

    rating = (
        (row["Lab_Work_Completion"] / 12) * 2.0 +
        (row["Viva_Marks"] / 5) * 2.0 +
        (row["Collaboration_Score"] / 10) * 2.0 +
        (row["Assignment_Quality"] / 5) * 1.5
    )

    if row["Project_Made"] == "Yes": rating += 0.5
    if row["Project_Deployed"] == "Yes": rating += 0.5
    if row["Presentation_Grade"] == "A": rating += 1.0
    elif row["Presentation_Grade"] == "B": rating += 0.5
    if row["Class_Participation"] == "Active": rating += 0.5
    elif row["Class_Participation"] == "Partially Active": rating += 0.25
    if row["Discipline_Score"] == "Excellent": rating += 0.5
    elif row["Discipline_Score"] == "Good": rating += 0.25

    rating += random.uniform(-0.3, 0.3)
    row["Rating"] = round(min(max(rating, 0), 10), 2)

    return row

data = [generate_row() for _ in range(1000)]
df = pd.DataFrame(data)
filename = "student_performance_dataset.csv"
df.to_csv(filename, index=False)

print(f"✅ Dataset generated: {filename} with {len(df)} rows")
print("📂 Location:", os.path.abspath(filename))
