import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_model.joblib"
DATA_PATH = BASE_DIR / "student_performance_synthetic.csv"

st.title("🎓 Student Performance Prediction Dashboard")
st.write("Predict student final scores using a trained Machine Learning model.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Put student_model.joblib beside app.py.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("Enter Student Details")

study_hours = st.sidebar.slider(
    "Study hours per day", 0.0, 8.0, 3.5, 0.5
)
attendance = st.sidebar.slider(
    "Attendance (%)", 45.0, 100.0, 82.0, 1.0
)
previous_score = st.sidebar.slider(
    "Previous score", 10.0, 100.0, 68.0, 1.0
)
assignments = st.sidebar.slider(
    "Assignments completed (out of 10)", 0, 10, 8
)
sleep_hours = st.sidebar.slider(
    "Sleep hours per day", 4.0, 10.0, 7.0, 0.5
)
extracurricular_hours = st.sidebar.slider(
    "Extracurricular hours per day", 0.0, 8.0, 2.0, 0.5
)
internet_access = st.sidebar.selectbox(
    "Internet access", ["Yes", "No"]
)
parental_support = st.sidebar.selectbox(
    "Parental support", ["High", "Medium", "Low"]
)
study_method = st.sidebar.selectbox(
    "Study method", ["Self-study", "Group study", "Tutoring"]
)

student_data = pd.DataFrame([{
    "study_hours_per_day": study_hours,
    "attendance_percentage": attendance,
    "previous_score": previous_score,
    "assignments_completed_out_of_10": assignments,
    "sleep_hours_per_day": sleep_hours,
    "extracurricular_hours_per_day": extracurricular_hours,
    "internet_access": internet_access,
    "parental_support": parental_support,
    "study_method": study_method
}])

# ---------------- PREDICTION ----------------
st.subheader("Student Prediction")

if st.button("Predict Final Score", type="primary"):
    prediction = float(model.predict(student_data)[0])
    prediction = np.clip(prediction, 0, 100)

    st.metric("Predicted Final Score", f"{prediction:.1f} / 100")

    if prediction >= 75:
        st.success("Predicted category: High score range")
    elif prediction >= 50:
        st.info("Predicted category: Moderate score range")
    else:
        st.warning("Predicted category: Lower score range")

    st.caption(
        "This is an estimate from a demonstration model, not a guaranteed "
        "exam result or a formal assessment of a student."
    )

# ---------------- DASHBOARD ANALYTICS ----------------
st.divider()
st.subheader("Dataset Analytics")

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Average Final Score", f"{df['final_score'].mean():.1f}")
    col3.metric("Average Attendance", f"{df['attendance_percentage'].mean():.1f}%")

    left, right = st.columns(2)

    with left:
        st.write("Final Score Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["final_score"], bins=20)
        ax.set_xlabel("Final Score")
        ax.set_ylabel("Number of Students")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.write("Study Hours vs Final Score")
        fig, ax = plt.subplots()
        ax.scatter(df["study_hours_per_day"], df["final_score"], alpha=0.5)
        ax.set_xlabel("Study Hours per Day")
        ax.set_ylabel("Final Score")
        st.pyplot(fig)
        plt.close(fig)

    st.write("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

else:
    st.warning(
        "Dataset CSV not found. Add student_performance_synthetic.csv "
        "beside app.py to enable analytics."
    )