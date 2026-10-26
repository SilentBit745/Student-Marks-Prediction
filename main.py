
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Sample student data
data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,2,4,6,8,3,5,7,9,1,3,5,7],
    "Attendance": [50,55,60,65,70,75,80,90,60,70,85,95,65,75,85,95,45,60,75,90],
    "Previous_Marks": [40,45,50,55,60,65,70,80,42,55,68,85,48,62,75,90,35,50,65,78],
    "Final_Marks": [35,40,48,52,60,65,72,85,43,57,70,88,46,64,78,95,30,47,68,80]
}

df = pd.DataFrame(data)

# Inputs and target
X = df[["Study_Hours", "Attendance", "Previous_Marks"]]
y = df["Final_Marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Two models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel:", name)
    print("Actual marks:", list(y_test))
    print("Predicted marks:", predictions.round(2))
    print("MAE:", round(mae, 2))
    print("R2 Score:", round(r2, 2))

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "R2 Score": round(r2, 2)
    })

# Compare models
results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df)

# Save files
df.to_csv("student_data.csv", index=False)
results_df.to_csv("model_results.csv", index=False)

print("\nFiles saved successfully!")
