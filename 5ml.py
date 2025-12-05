import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error


def prepare_data(df):
    """Encodes, scales, and splits data automatically."""
    
    # Detect target column (last column)
    TARGET = df.columns[-1]
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Encode categorical columns
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Scale numeric values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    return train_test_split(X, y, test_size=0.2, random_state=42), TARGET


def evaluate(df, label):
    """Train model and print results based on problem type."""
    
    (X_train, X_test, y_train, y_test), TARGET = prepare_data(df)

    # Detect classification or regression
    problem_type = "Regression" if df[TARGET].dtype in ['float64', 'int64'] and len(df[TARGET].unique()) > 15 else "Classification"
    
    print(f"\n🧪 Evaluating {label} ({problem_type})")

    if problem_type == "Classification":
        model = LogisticRegression(max_iter=300)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"📊 Accuracy: {accuracy_score(y_test, y_pred)}")

    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"📈 MSE: {mean_squared_error(y_test, y_pred)}")



# -------------------------
# Load both datasets
# -------------------------
raw_df = pd.read_csv("SP.csv")
clean_df = pd.read_csv("SP_cleaned.csv")

print("\n🚀 Model Training Started...\n")

# Evaluate raw dataset
evaluate(raw_df, "Before Cleaning (SP.csv)")

# Evaluate cleaned dataset
evaluate(clean_df, "After Cleaning (SP_cleaned.csv)")

print("\n✅ Day 5 Completed: Comparison Done!")

