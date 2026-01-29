import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# ------------------ Page Config ------------------
st.set_page_config(page_title="Student Stress Monitor", layout="centered")

# ------------------ Session State ------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ------------------ HOME PAGE ------------------
if st.session_state.page == "home":

    st.title("🏠 Student Stress Monitoring System")

    st.markdown("""
    ### Welcome to Student Stress Monitoring System

    Student Stress Monitoring System is a **data science–based web platform**
    designed to analyze and predict stress levels among students.

    With increasing academic pressure, competition, and lifestyle challenges,
    monitoring mental health has become essential. This system helps in identifying
    stress patterns at an early stage using **machine learning techniques** and
    provides insights to promote a healthier academic life.
    """)

    st.markdown("""
    ### Why This Platform?
    - Stress affects academic performance and mental well-being  
    - Early detection helps prevent serious mental health issues  
    - Data-driven insights help students take proactive steps  
    """)

    st.markdown("""
    ### Get Started
    Enter your details, analyze your stress level, and take the first step
    toward a **balanced and healthy student life**.
    """)

    st.image(
        "https://i.postimg.cc/sDdnxK38/Screenshot-2025-08-26-222129.png",
        use_container_width=True
    )

    if st.button("🚀 Get Started"):
        st.session_state.page = "predict"
        st.rerun()

# ------------------ PREDICTION PAGE ------------------
elif st.session_state.page == "predict":

    st.header("🎓 Student Stress Monitoring Using Machine Learning")

    description = '''
    About the Project

The Student Stress Monitoring System is developed as a data science project with the goal of monitoring, analyzing, and predicting stress levels in students. The system collects academic, behavioral, and lifestyle-related inputs and applies machine learning algorithms to evaluate stress conditions.

This project highlights the importance of mental health awareness and demonstrates how data science can be effectively used to support students in managing stress and improving their academic outcomes.

    '''
    st.markdown(description)

    # ------------------ Load Model & Data ------------------
    with open('stud_stress_monitoring.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv("StressLevelDataset.csv")

    # ------------------ Sidebar Input ------------------
    st.sidebar.header("🔍 Enter Student Details")
    st.sidebar.image(
        "https://i.postimg.cc/wvDmGXny/Screenshot-2025-08-26-222159.png",
        use_container_width=True
    )

    user_inputs = {}

    for col in df.columns:
        if col.lower() in ["stresslevel", "stress_level"]:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            min_val, max_val = int(df[col].min()), int(df[col].max())
            default_val = int(df[col].mean())
            val = st.sidebar.slider(col, min_val, max_val, default_val)
        else:
            options = df[col].unique().tolist()
            val = st.sidebar.selectbox(col, options)

        user_inputs[col] = val

    input_df = pd.DataFrame([user_inputs])

    st.subheader("📋 Input Data")
    st.dataframe(input_df)

    # ------------------ Prediction ------------------
    if st.button("🔍 Predict Stress Level"):

        prediction = model.predict(input_df)[0]

        progress_bar = st.progress(0)
        status_text = st.empty()
        loading_gif = st.empty()

        status_text.subheader("🔄 Predicting Stress Level...")
        loading_gif.image(
            "https://img1.picmix.com/output/stamp/normal/7/1/8/1/2331817_aad20.gif",
            width=200
        )

        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)

        status_text.empty()
        loading_gif.empty()

        stress_labels = {
            0: "No Stress",
            1: "Mild Stress",
            2: "Moderate Stress",
            3: "High Stress"
        }

        stress_result = stress_labels.get(prediction, "Unknown")

        if prediction == 0:
            st.success(f"🟢 Stress Level: {stress_result}")
        elif prediction in [1, 2]:
            st.warning(f"🟠 Stress Level: {stress_result}")
        else:
            st.error(f"🔴 Stress Level: {stress_result}")

    # ------------------ Back Button ------------------
    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"
        st.rerun()
