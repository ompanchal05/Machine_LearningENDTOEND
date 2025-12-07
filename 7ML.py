import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib


# -----------------------------
# Load cleaned dataset
# -----------------------------
df = pd.read_csv("SP_cleaned.csv")
TARGET = df.columns[-1]

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Detect task type
problem_type = "Regression" if y.dtype in ['float64', 'int64'] and len(y.unique()) > 15 else "Classification"
print(f"🔍 Problem Type Detected: {problem_type}")


# -----------------------------
# Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------
# Scale numeric values
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# Define models and parameters
# -----------------------------
if problem_type == "Classification":
    model = RandomForestClassifier()
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }
else:
    model = RandomForestRegressor()
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }


# -----------------------------
# Grid Search Tuning
# -----------------------------
print("\n⚙️ Hyperparameter Tuning Started... Please wait...")

grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy' if problem_type=="Classification" else 'neg_mean_squared_error')
grid.fit(X_train, y_train)

print("\n🏆 Best Parameters Found:")
print(grid.best_params_)


# -----------------------------
# Evaluate best model
# -----------------------------
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

if problem_type == "Classification":
    score = accuracy_score(y_test, y_pred)
    print(f"\n📊 Tuned Accuracy: {score}")
else:
    score = mean_squared_error(y_test, y_pred)
    print(f"\n📉 Tuned MSE: {score}")


# -----------------------------
# Save final model
# -----------------------------
joblib.dump(best_model, "final_model.pkl")
joblib.dump(scaler, "final_scaler.pkl")

print("\n💾 Saved model as: final_model.pkl")
print("\n🎯 Day 7 Completed!")
