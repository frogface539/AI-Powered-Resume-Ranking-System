# 📄 AI-Powered Resume Ranker

This is an NLP-powered resume ranking web app built with **Streamlit**. It ranks uploaded resumes based on their relevance to a given **job description** using **Word2Vec-based semantic similarity** and **skill keyword matching**.

---

## 🚀 Features

- 🔍 Ranks resumes using **semantic similarity** (Word2Vec)
- 🧠 Highlights **matched skills** from the job description
- 📊 Displays **Match %**, **Similarity Score**, and a full ranking table
- 📁 Supports **PDF** and **TXT** resume formats
- 📄 Responsive **Streamlit interface**
- ⬇️ Downloadable ranking report (CSV)

---

## 📦 Tech Stack

- `Python 3.10+`
- `Streamlit`
- `NLTK` for text preprocessing
- `Gensim` for Word2Vec embeddings
- `PyMuPDF` for PDF text extraction
- `scikit-learn` for similarity metrics

---

## 📂 Folder Structure

```text
resume_ranker/
├── app.py                  # Main Streamlit app
├── parser.py               # Resume and JD text extraction
├── ranker.py               # Word2Vec model and ranking logic
├── utils.py                # Text cleaning functions
├── requirements.txt
└── sample_resumes/         # [Optional] Sample PDFs (not included)
````

---

## ⚠️ Word2Vec Model Note

Due to the 1.5 GB size of the original **GoogleNews Word2Vec embeddings**, this project uses:

```python
from gensim.downloader import load
model = load("word2vec-google-news-300")
```

✅ This allows the model to be downloaded **automatically at runtime** (first run only).

---

## 💻 How to Run Locally

1. **Clone the repo**

```bash
git clone https://github.com/your-username/resume-ranker.git
cd resume-ranker
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the app**

```bash
streamlit run app.py
```

5. **Open in browser**

```
http://localhost:8501
```

---

## ✍️ Example Use Case

> Upload 10 resumes and a job description for a "Data Scientist – NLP" role.
> Get a ranked list with relevance scores, matched skills, and a downloadable CSV report.

---

## 📄 License

License © 2025 \[Lakshay Jain]

