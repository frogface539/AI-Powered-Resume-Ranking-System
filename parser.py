import fitz
import os
from utils import clean_text
from ranker import load_word2vec_model, rank_resumes


def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()

        doc.close()
        return text
    
    except Exception as e:
        print(f"[ERROR]Could not read {pdf_path}: {e}")
        return ""
    
def extract_text_from_txt(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
        
    except Exception as e:
        print(f"[ERROR] Could not read {txt_path}: {e}")
        return ""
    
def load_resumes(folder_path):
    resumes = {}
    files = os.listdir(folder_path)
    print("[DEBUG] Files in folder:", files)

    for filename in files:
        print(f"[INFO] Processing: {filename}")
        filepath = os.path.join(folder_path, filename)

        if filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(filepath)
        elif filename.lower().endswith('.txt'):
            text = extract_text_from_txt(filepath)
        else:
            print(f"[SKIPPED] Unsupported file type: {filename}")
            continue

        if text.strip():
            resumes[filename] = text
        else:
            print(f"[WARNING] No text extracted from: {filename}")

    return resumes

    
def load_job_description(jd_path):
    try:
        with open(jd_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"[ERROR] Could not load JD: {e}")
        return ""

## Testing
# from utils import clean_text

# if __name__ == "__main__":
#     resumes = load_resumes("sample_resumes/")
#     jd = load_job_description("job_description.txt")

#     cleaned_resumes = {filename: clean_text(text) for filename, text in resumes.items()}
#     cleaned_jd = clean_text(jd)

#     print("\n=== Cleaned Job Description ===\n")
#     print(cleaned_jd)

#     for filename, cleaned_text in cleaned_resumes.items():
#         print(f"\n=== Cleaned Resume: {filename} ===")
#         print(cleaned_text)

#     model = load_word2vec_model()
#     ranking = rank_resumes(cleaned_resumes, cleaned_jd, model)

#     print("\n=== Resume Ranking ===")
#     for name, score in ranking:
#         print(f"{name}: {score}")

