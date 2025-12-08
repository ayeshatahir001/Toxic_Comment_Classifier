import joblib
import re
import os

# --------------------------
# LOAD MODEL + TFIDF SAFELY
# --------------------------

BASE_DIR = os.path.dirname(__file__)

tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "best_toxic_model.pkl"))

# --------------------------
# SIMPLE STOPWORDS (NO NLTK)
# --------------------------
STOPWORDS = set([
    "the","is","and","are","a","an","to","you","your","of","in","for","on",
    "with","that","this","it","be","as","at","by","from","about","but","if","so"
])

# --------------------------
# PREPROCESSING
# --------------------------
def remove_stopwords(text):
    return " ".join([w for w in text.split() if w.lower() not in STOPWORDS])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def stemming(text):
    # simple manual stemming (safe for deployment)
    return " ".join([w.rstrip("ing").rstrip("ed") for w in text.split()])

def preprocess(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = stemming(text)
    return text

# --------------------------
# LABELS
# --------------------------
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# --------------------------
# PREDICT FUNCTION (USED BY app.py)
# --------------------------
def predict_toxicity(text):
    cleaned = preprocess(text)
    vec = tfidf.transform([cleaned])
    preds = model.predict(vec)[0]
    preds = [int(x) for x in preds]
    
    return {label: value for label, value in zip(labels, preds)}
