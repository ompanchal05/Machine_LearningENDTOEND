import pandas as pd
import numpy as np
import joblib

# ------------------------
# Load model + scaler + dataset reference
# ------------------------
model = joblib.load("final_model.pkl")
scaler = joblib.load("final_scaler.pkl")

df = pd.read_csv("SP_cleaned.csv")  # Reference structure
TARGET = df.columns[-1]
FEATURES = df.columns[:-1]

print("🚀 Model Loaded Successfully!")
print("\nRequired Inputs:", list(FEATURES))


# ------------------------
# USER INPUT (DYNAMIC)
# ------------------------
user_values = []

print("\n🧾 Enter values for prediction:\n")

for col in FEATURES:
    value = input(f"Enter value for {col}: ")
    
    # Convert to float if possible
    try:
        value = float(value)
    except:
        # If it's text instead of number
        # Convert using stored label encoding logic (optional)
        pass
    
    user_values.append(value)

# Convert to array
user_array = np.array(user_values).reshape(1, -1)

# ------------------------
# Scale the new input
# ------------------------
scaled_input = scaler.transform(user_array)

# ------------------------
# Predict
# ------------------------
prediction = model.predict(scaled_input)[0]

print("\n🎯 Prediction Result:", prediction)

