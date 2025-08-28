import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# ------------------- Page Title and Description -------------------
st.set_page_config(page_title="Student Stress Monitor", layout="centered")
st.header('🎓 Student Stress Monitoring Using Machine Learning')

description = '''
To develop a machine learning-based model that analyzes student data to monitor, classify, and predict stress levels, aiming to identify key factors influencing student stress and provide actionable insights to improve mental well-being and academic performance.

**Algorithms Used:**
- Logistic Regression
- Naive Bayes
- Support Vector Machine (Linear)
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- XGBoost
- Artificial Neural Network (1 Hidden Layer, Keras)
'''
st.markdown(description)

# ------------------- Main Image -------------------
st.image('https://i.postimg.cc/sDdnxK38/Screenshot-2025-08-26-222129.png')

# ------------------- Load Model and Dataset -------------------
with open('stud_stress_monitoring.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv("C:\Users\pankh\.cache\kagglehub\datasets\mdsultanulislamovi\student-stress-monitoring-datasets\versions\1.csv")

# ------------------- Sidebar Inputs -------------------
st.sidebar.header('🔍 Select Feature Values')
st.sidebar.image('https://i.postimg.cc/wvDmGXny/Screenshot-2025-08-26-222159.png')

user_inputs = {}

# Loop through dataset columns (skip target)
for col in df.columns:
    if col.lower() in ["stresslevel", "stress_level"]:  # Skip label column
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        min_val, max_val = int(df[col].min()), int(df[col].max())
        default_val = int(df[col].mean())   # use mean as stable default
        val = st.sidebar.slider(f"{col}", min_val, max_val, default_val)
    else:
        options = df[col].unique().tolist()
        val = st.sidebar.selectbox(f"{col}", options)

    user_inputs[col] = val

# ------------------- Make Prediction -------------------
# Ensure column order matches training
input_df = pd.DataFrame([user_inputs])

st.write("📋 Input DataFrame used for prediction:")
st.dataframe(input_df)

prediction = model.predict(input_df)[0]

# ------------------- Simulated Loading -------------------
progress_bar = st.progress(0)
status_text = st.empty()
loading_gif = st.empty()

status_text.subheader('🔄 Predicting Stress Level...')
loading_gif.image('https://img1.picmix.com/output/stamp/normal/7/1/8/1/2331817_aad20.gif', width=200)

for i in range(100):
    time.sleep(0.02)
    progress_bar.progress(i + 1)

status_text.empty()
loading_gif.empty()

# ------------------- Display Prediction -------------------
stress_labels = {
    0: 'No Stress',
    1: 'Mild Stress',
    2: 'Moderate Stress',
    3: 'High Stress'
}

stress_result = stress_labels.get(prediction, "Unknown Stress Level")

if prediction == 0:
    st.success(f"🟢 {stress_result}")
else:
    st.warning(f"🟠 {stress_result}")
