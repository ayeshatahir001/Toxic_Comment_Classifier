import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# --------------------------
# LOAD SAVED MODEL & TF-IDF
# --------------------------
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("best_toxic_model.pkl")

# --------------------------
# PREPROCESSING STEPS
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
    text = remove_stopwords(text)
    text = clean_text(text)
    text = stemming(text)
    return text

# --------------------------
# PREDICTION FUNCTION
# --------------------------
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def predict_toxicity(text):
    # Clean text
    cleaned = preprocess(text)

    # Vectorize
    vec = tfidf.transform([cleaned])

    # Predict
    preds = model.predict(vec)[0]

    # Convert np.int64 → int
    return {label: int(value) for label, value in zip(labels, preds)}
