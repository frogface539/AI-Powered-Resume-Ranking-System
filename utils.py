import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", ' ', text)
    tokens = word_tokenize(text)

    filtered = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(filtered)