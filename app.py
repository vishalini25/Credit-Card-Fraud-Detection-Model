import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv('creditcard.csv')

# Define feature names
feature_columns = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
                   "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
                   "V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]

# Prepare training data
X = data[feature_columns]
y = data["Class"]

# Balance dataset
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
legit_sample = legit.sample(n=len(fraud), random_state=2)
balanced_data = pd.concat([legit_sample, fraud], axis=0)

X_balanced = balanced_data[feature_columns]
y_balanced = balanced_data["Class"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=2
)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Streamlit app
st.title("💳 Credit Card Fraud Detection")

st.markdown("Enter **31 values** (30 features + 1 class label), comma-separated:")
st.code(', '.join(feature_columns + ["Class"]), language='markdown')



input_str = st.text_input("Enter your 31 input values:")

if st.button("Predict"):
    try:
        raw_values = input_str.strip().split(',')

        if len(raw_values) != 31:
            st.error(f"❌ Expected 31 values (30 features + 1 class), but got {len(raw_values)}.")
        else:
            feature_values = [float(v) for v in raw_values[:30]]  # Use only first 30
            features = np.array(feature_values).reshape(1, -1)

            prediction = model.predict(features)[0]

            if prediction == 0:
                st.success("✅ This is a **Legitimate** transaction.")
            else:
                st.error("🚨 This is a **Fraudulent** transaction.")

    except ValueError:
        st.error("⚠️ Please ensure all values are valid numbers.")
