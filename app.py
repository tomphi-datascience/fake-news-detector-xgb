import streamlit as st
import cloudpickle
import re

# Load the TF-IDF vectorizer and trained XGBoost model using cloudpickle
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = cloudpickle.load(f)

with open("fakenews_model.pkl", "rb") as f:
    model = cloudpickle.load(f)

# Set page config
st.set_page_config(page_title="Fake News Classifier", layout="centered")

# Text cleaning function (same as training)
def clean_text_input(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", '', text)  # Remove punctuation and numbers
    text = re.sub(r"\s+", ' ', text).strip()  # Remove extra whitespace
    return text

# Streamlit App UI
st.title("📰 Fake News Classifier")
st.write("Enter a news article and find out whether it's **real** or **fake** using a trained machine learning model.")

# Input text box
user_input = st.text_area("📝 Paste your news article text below:", height=200)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text before clicking Predict.")
    else:
        # Preprocess and vectorize
        cleaned = clean_text_input(user_input)
        vectorized = tfidf.transform([cleaned])

        # Predict
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0][prediction]

        # Output
        label = "REAL 🟢" if prediction == 1 else "FAKE 🔴"
        st.markdown(f"### 🧠 Prediction: **{label}**")
        st.markdown(f"**Confidence:** {probability:.2%}")
