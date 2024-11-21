import streamlit as st
import pandas as pd
import pickle

st.title("Hiring Probability Predictor")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Data")
    age = st.number_input("Age", min_value=18, max_value=100, value=18)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    distance_from_company = st.number_input("Distance from Company (km)", min_value=0.0, max_value=100.0, value=1.0)

with col2:
    st.subheader("Interview Data")
    interview_score = st.number_input("Interview Score", min_value=0, max_value=100, value=50)
    skill_score = st.number_input("Skill Score", min_value=0, max_value=100, value=50)
    personality_score = st.number_input("Personality Score", min_value=0, max_value=100, value=50)
    recruitment_strategy = st.selectbox("Recruitment Strategy", [1, 2, 3], format_func=lambda x: "Aggressive" if x == 1 else ("Moderate" if x == 2 else "Conservative"))

with col3:
    st.subheader("Background")
    education_level = st.selectbox("Education Level", [1, 2, 3], format_func=lambda x: "Bachelor's" if x == 1 else ("Master's" if x == 2 else "PhD"))
    experience_years = st.number_input("Years of Experience", min_value=0, max_value=50, value=1)
    previous_companies = st.number_input("Previous Companies", min_value=0, max_value=20, value=1)

new_candidate = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'EducationLevel': education_level,
    'ExperienceYears': experience_years,
    'PreviousCompanies': previous_companies,
    'DistanceFromCompany': distance_from_company,
    'InterviewScore': interview_score,
    'SkillScore': skill_score,
    'PersonalityScore': personality_score,
    'RecruitmentStrategy': recruitment_strategy
}])

with open("rf_model.pkl", "rb") as model_file:
    model_rf = pickle.load(model_file)
with open("logreg_model.pkl", "rb") as model_file:
    model_logreg = pickle.load(model_file)
with open("knn_model.pkl", "rb") as model_file:
    model_knn = pickle.load(model_file)

prob_rf = model_rf.predict_proba(new_candidate)[:, 1][0] * 100
prob_logreg = model_logreg.predict_proba(new_candidate)[:, 1][0] * 100
prob_knn = model_knn.predict_proba(new_candidate)[:, 1][0] * 100

results = {
    'Model': ['Random Forest', 'Logistic Regression', 'KNN'],
    'Result': [
        f"Yes ({prob_rf:.2f}%)" if prob_rf >= 70 else f"No ({prob_rf:.2f}%)",
        f"Yes ({prob_logreg:.2f}%)" if prob_logreg >= 70 else f"No ({prob_logreg:.2f}%)",
        f"Yes ({prob_knn:.2f}%)" if prob_knn >= 70 else f"No ({prob_knn:.2f}%)"
    ]
}

df_results = pd.DataFrame(results)

def colorize(val):
    color = 'green' if 'Yes' in val else 'red'
    return f'background-color: {color}; color: white'

def add_bars(val):
    percentage = float(val.split('(')[1].replace('%)', ''))
    if 'Yes' in val:
        return f'background: linear-gradient(90deg, green {percentage}%, lightgray {percentage}%);'
    else:
        return f'background: linear-gradient(90deg, red {percentage}%, lightgray {percentage}%);'

styler = df_results.style.applymap(colorize, subset=['Result']).applymap(add_bars, subset=['Result']).hide(axis="index")

centered_html = f"""
<div style="display: flex; justify-content: center;">
    {styler.to_html()}
</div>
"""

st.write(centered_html, unsafe_allow_html=True)
