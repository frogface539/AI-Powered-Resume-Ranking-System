import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_word2vec_model(path="embeddings\GoogleNews-vectors-negative300.bin"):
    print("[INFO] Loading Word2Vec model....")
    model = KeyedVectors.load_word2vec_format(path, binary= True)
    print("[INFO] Model loaded....")
    return model

def get_average_vector(text, model, vector_size=300):
    words = text.split()
    valid_vectors = []

    for word in words:
        if word in model:
            valid_vectors.append(model[word])

    if not valid_vectors:
        return np.zeroes(vector_size)
    
    return np.mean(valid_vectors, axis=0)

## Resume Rankings
def rank_resumes(cleaned_resumes, cleaned_jd, model):
    jd_vec = get_average_vector(cleaned_jd, model)
    scores = []

    for filename, resume_text in cleaned_resumes.items():
        resume_vec = get_average_vector(resume_text, model)
        similarity = cosine_similarity([jd_vec], [resume_vec])[0][0]
        scores.append((filename, round(similarity, 4)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
