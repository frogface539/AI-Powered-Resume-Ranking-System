import streamlit as st
import os
import tempfile
import pandas as pd
import base64
from parser import load_resumes, load_job_description
from utils import clean_text
from ranker import load_word2vec_model, rank_resumes

st.set_page_config(page_title="Resume Ranking System", layout="wide")
st.title("📄 AI-Powered Resume Ranking System")

st.markdown("""
Upload multiple resumes and a job description to **rank candidates** by textual relevance using semantic similarity.
""")

# Divider
st.markdown("---")

# 🔹 Job Description Section
st.header("📌 Job Description")

jd_input = st.text_area("Paste the Job Description", height=200)
uploaded_jd = st.file_uploader("Or upload a `.txt` JD file", type=['txt'], key="jd_upload")

# 🔹 Resume Upload Section
st.header("📁 Upload Resumes")
resume_files = st.file_uploader("Upload multiple resumes (PDF or TXT)", type=['pdf', 'txt'], accept_multiple_files=True)

# Divider
st.markdown("---")

# 🔍 Main Ranking Logic
if st.button("🔍 Rank Resumes"):
    if not resume_files:
        st.error("Please upload at least one resume.")
    elif not jd_input and not uploaded_jd:
        st.error("Please paste or upload a job description.")
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            resumes = {}
            base64_pdfs = {}

            for file in resume_files:
                file_path = os.path.join(temp_dir, file.name)
                content = file.read()
                with open(file_path, "wb") as f:
                    f.write(content)

                resumes[file.name] = content
                base64_pdfs[file.name] = base64.b64encode(content).decode("utf-8")

            resume_texts = load_resumes(temp_dir)

        # Load and clean JD
        jd_text = jd_input
        if uploaded_jd and not jd_input:
            jd_text = uploaded_jd.read().decode("utf-8")

        cleaned_jd = clean_text(jd_text)
        jd_skills = set(cleaned_jd.split())
        cleaned_resumes = {fn: clean_text(text) for fn, text in resume_texts.items()}

        # Ranking
        with st.spinner("Ranking resumes..."):
            model = load_word2vec_model()
            rankings = rank_resumes(cleaned_resumes, cleaned_jd, model)

        # Match stats
        match_data = []
        for filename, score in rankings:
            resume_words = set(cleaned_resumes[filename].split())
            matched_skills = list(jd_skills & resume_words)
            match_percent = round(len(matched_skills) / len(jd_skills) * 100, 2)
            match_data.append({
                "Resume": filename,
                "Similarity Score": round(score, 2),
                "Match %": match_percent,
                "Matched Skills": ", ".join(sorted(matched_skills)),
            })

        df = pd.DataFrame(match_data)
        st.success("✅ Ranking Complete")
        st.subheader("🏆 Top Ranked Resumes")
        for row in match_data[:3]:
            with st.container():
                st.markdown(f"**📄 {row['Resume']}**")
                col1, col2 = st.columns(2)
                col1.metric("Similarity Score", round(row['Similarity Score'], 2))
                col2.metric("Skill Match %", f"{round(row['Match %'], 2)}%")
                st.markdown(f"**Matched Skills:** `{row['Matched Skills']}`")
                st.markdown("---")

        # Full table
        st.subheader("📊 Full Ranking Table")
        st.dataframe(df, use_container_width=True)

        # CSV download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Ranking Report", csv, "resume_ranking.csv", "text/csv")

        # Optional note
        st.markdown("*Note: Resume preview in Brave is restricted. PDF viewer integration is disabled for now for full compatibility.*")
