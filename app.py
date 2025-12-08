import streamlit as st
import joblib
from utils import remove_stopwords, clean_text, stemming

# --------------------- CONFIG ---------------------
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🛡️",
    layout="centered"
)

LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

# Load model
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("best_toxic_model.pkl")


# --------------------- SIDEBAR ---------------------
st.sidebar.title("⚙️ Model Info")

st.sidebar.markdown("""
### **🔍 About This App**
This is a **Multi-label Toxic Comment Classifier** that detects:

- 🧪 Toxic  
- 🔥 Severe Toxic  
- 🤬 Obscene  
- ⚠️ Threat  
- 😡 Insult  
- 🎯 Identity Hate  

Enter any sentence and see predictions instantly.

---
### **📦 Model Used**
- TF-IDF Vectorizer  
- Linear SVM (best accuracy)

---
""")


# --------------------- HEADER ---------------------
st.markdown("<h1 style='text-align:center;'>🛡️ Toxic Comment Classifier</h1>", unsafe_allow_html=True)
st.write("Enter a sentence below and click **Predict Toxicity**.")

st.write("---")


# --------------------- TEXT INPUT ---------------------
text = st.text_area("✍️ Write your comment here...", height=150)


# --------------------- PREPROCESS FUNCTION ---------------------
def preprocess(text):
    text = remove_stopwords(text)
    text = clean_text(text)
    text = stemming(text)
    return text


# --------------------- PREDICT BUTTON ---------------------
if st.button("🔎 Predict Toxicity", use_container_width=True):

    if text.strip() == "":
        st.error("❗ Please enter some text!")
    else:
        cleaned = preprocess(text)
        vec = vectorizer.transform([cleaned])
        preds = model.predict(vec)[0]
        preds = [int(x) for x in preds]

        st.success("🎉 **Prediction Complete!**")
        st.write("### Results:")

        # DISPLAY RESULTS AS COLORED TAGS
        for label, value in zip(LABELS, preds):
            color = "#FF4B4B" if value == 1 else "#4CAF50"
            result_text = "Detected" if value == 1 else "Not Detected"
            
            st.markdown(
                f"""
                <div style="
                    padding:10px;
                    margin:5px 0;
                    border-radius:6px;
                    background-color:{color};
                    color:white;
                    font-size:16px;
                    ">
                    <b>{label.upper()}</b> — {result_text}
                </div>
                """,
                unsafe_allow_html=True
            )


# --------------------- FOOTER ---------------------
st.write("---")
st.markdown("<p style='text-align:center; color:grey;'>Built by Ayesha • Streamlit App 🌐</p>", unsafe_allow_html=True)
