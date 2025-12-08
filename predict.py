import joblib
import re 
import os
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# --------------------------
# DIRECT LOAD (NO BASE_DIR)
# --------------------------
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("best_toxic_model.pkl")

# --------------------------
# PREPROCESSING
# --------------------------
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stop_words])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def stemming(text):
    return " ".join([stemmer.stem(w) for w in text.split()])

def preprocess(text):
    return stemming(clean_text(remove_stopwords(text)))

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def predict_toxicity(text):
    cleaned = preprocess(text)
    vec = tfidf.transform([cleaned])
    preds = model.predict(vec)[0]
    return {label: int(value) for label, value in zip(labels, preds)}
