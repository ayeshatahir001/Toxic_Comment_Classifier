import streamlit as st
from predict import predict_toxicity

# --------------------- CONFIG ---------------------
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🛡️",
    layout="centered"
)

LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

# --------------------- SIDEBAR ---------------------
st.sidebar.title("Model Info")

st.sidebar.markdown("""
### **About This App**
This is a **Multi-label Toxic Comment Classifier** that detects:

- 🧪 Toxic  
- 🔥 Severe Toxic  
- 🤬 Obscene  
- ⚠️ Threat  
- 😡 Insult  
- 🎯 Identity Hate  

Enter any sentence and see predictions instantly.
""")

# --------------------- HEADER ---------------------
st.markdown("<h1 style='text-align:center;'>🛡️ Toxic Comment Classifier</h1>", unsafe_allow_html=True)
st.write("Enter a sentence below and click **Predict Toxicity**.")
st.write("---")

# --------------------- TEXT INPUT ---------------------
text = st.text_area("Write your comment here...", height=150)

# --------------------- PREDICT BUTTON ---------------------
if st.button("Predict Toxicity", use_container_width=True):

    if text.strip() == "":
        st.error("❗ Please enter some text!")
    else:
        results = predict_toxicity(text)
        st.success("🎉 **Prediction Complete!**")
        st.write("### Results:")

        # --------------------- DISPLAY RESULTS ---------------------
        for label, value in results.items():

            is_detected = (value.lower() == "detected")
            color = "#FF4B4B" if is_detected else "#4CAF50"
            result_text = value

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
