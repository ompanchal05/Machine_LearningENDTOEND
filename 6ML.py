import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

# Models for testing
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("SP_cleaned.csv")
TARGET = df.columns[-1]
X = df.drop(columns=[TARGET])
y = df[TARGET]


# -------------------------
# Detect Classification or Regression
# -------------------------
problem_type = "Regression" if df[TARGET].dtype in ['float64', 'int64'] and len(df[TARGET].unique()) > 15 else "Classification"
print(f"\n🔍 Problem Type: {problem_type}")


# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# -------------------------
# Scale Data
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -------------------------
# Model Testing Function
# -------------------------
def evaluate_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if problem_type == "Classification":
        return accuracy_score(y_test, y_pred)
    else:
        return -mean_squared_error(y_test, y_pred)  # negative because higher is better


# -------------------------
# Model List
# -------------------------
if problem_type == "Classification":
    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }
else:
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(),
        "SVR": SVR(),
        "KNN Regressor": KNeighborsRegressor()
    }


# -------------------------
# Evaluate All Models
# -------------------------
results = {}

print("\n📊 Testing Models...\n")

for name, model in models.items():
    try:
        score = evaluate_model(model)
        results[name] = score
        print(f"{name}: {score}")
    except Exception as e:
        print(f"{name}: ❌ Failed ({e})")


# -------------------------
# Choose Best Model
# -------------------------
best_model_name = max(results, key=results.get)
best_score = results[best_model_name]

print("\n🏆 Best Model:", best_model_name, "→ Score:", best_score)


# -------------------------
# Train Best Model Fully and Save
# -------------------------
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "best_scaler.pkl")

print("\n💾 Saved best model as best_model.pkl and scaler as best_scaler.pkl")
print("\n day 6 ML Complete! 🎉\n")
