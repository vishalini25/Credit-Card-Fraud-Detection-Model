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

# Default example input with 31 values
sample_input = "0,-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62,0"

input_str = st.text_input("Enter your 31 input values:", value=sample_input)

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
