import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from experience_extractor import extract_years_of_experience
from resume_parser import extract_text
from skill_extractor import extract_skills
from similarity_model import compute_similarity
from ml_model import train_dummy_model

# Train ML Model
model, tfidf, scaler = train_dummy_model()

# Streamlit UI
st.set_page_config(page_title="Resume Analyzer Pro", layout="wide")
st.title("📄 Resume Analyzer Pro Edition")
st.markdown("**🧠 Smart Resume Screening with AI-powered Skill & JD Matching**")

with st.sidebar:
    st.header("📂 Upload Resumes")
    uploaded_files = st.file_uploader("Choose PDF/DOCX Files", type=["pdf", "docx"], accept_multiple_files=True)

    st.header("📝 Job Description")
    jd_text = st.text_area("Paste Job Description", height=200)

    st.header("⚙️ Scoring Mode")
    sort_option = st.radio("Sort by", ["ML Score", "JD Similarity", "Combined (50-50)"])

if uploaded_files:
    resume_data = []
    st.subheader("🔎 Extracting resumes ...")

    progress_bar = st.progress(0)
    for idx, file in enumerate(uploaded_files):
        text = extract_text(file)
        skills = extract_skills(text)

    # Automatically extract experience from text:
        exp = extract_years_of_experience(text)

        resume_data.append({
        "file_name": file.name,
        "text": text,
        "experience": exp,
        "skills": skills
        })
        progress_bar.progress((idx+1)/len(uploaded_files))
        time.sleep(0.2)


    # ML Prediction
    resume_texts = [r["text"] for r in resume_data]
    resume_vecs = tfidf.transform(resume_texts).toarray()
    exp_scaled = scaler.transform(np.array([r["experience"] for r in resume_data]).reshape(-1, 1))
    features = np.hstack([resume_vecs, exp_scaled])
    ml_scores = model.predict(features)

    # Similarity
    jd_similarity = compute_similarity(jd_text, resume_texts)

    # Combine
    result_df = pd.DataFrame({
        "File Name": [r["file_name"] for r in resume_data],
        "Experience": [r["experience"] for r in resume_data],
        "ML Score": ml_scores,
        "JD Similarity": jd_similarity
    })

    if sort_option == "ML Score":
        result_df = result_df.sort_values("ML Score", ascending=False)
    elif sort_option == "JD Similarity":
        result_df = result_df.sort_values("JD Similarity", ascending=False)
    else:
        result_df["Combined"] = 0.5 * result_df["ML Score"] + 0.5 * result_df["JD Similarity"]
        result_df = result_df.sort_values("Combined", ascending=False)

    st.success("✅ Analysis Complete")

    # Display Table
    st.dataframe(result_df, use_container_width=True)

    # Download Button
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results", data=csv, file_name="results.csv", mime='text/csv')

    # Visualization
    st.subheader("📊 Visual Insights")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.barplot(x="ML Score", y="File Name", data=result_df, palette="viridis", ax=ax)
        ax.set_title("ML Predicted Scores")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.barplot(x="JD Similarity", y="File Name", data=result_df, palette="coolwarm", ax=ax)
        ax.set_title("JD-Resume Similarity")
        st.pyplot(fig)

else:
    st.info("📂 Please upload resumes first.")
