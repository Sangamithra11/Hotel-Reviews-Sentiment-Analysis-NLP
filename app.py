import streamlit as st
import joblib
import re
import string
from transformers import pipeline

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@st.cache_resource
def load_transformer():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=-1  # Force CPU (fixes meta tensor error)
    )

nlp = load_transformer()

ml_label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

transformer_label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

st.set_page_config(
    page_title="Hotel Review Sentiment Analysis",
    layout="centered"
)

st.title("Hotel Review Sentiment Analysis")
st.write(
    "This application analyzes hotel reviews using:\n"
)

user_input = st.text_area(
    "Enter a hotel review:",
    height=160,
    placeholder=""
)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        cleaned_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        ml_pred = model.predict(vectorized_text)[0]
        ml_sentiment = ml_label_map[ml_pred]

        transformer_output = nlp(user_input[:512])[0]
        transformer_sentiment = transformer_label_map[transformer_output["label"]]
        transformer_score = transformer_output["score"]

        st.subheader("Results")

        st.markdown("#### Traditional ML Model")
        st.success(
            f"**Sentiment:** {ml_sentiment}"
        )

        st.markdown("#### Transformer Model (RoBERTa)")
        st.info(
            f"**Sentiment:** {transformer_sentiment}\n\n"
            f"**Confidence:** {transformer_score:.4f}"
        )

    else:
        st.warning("Please enter a review to analyze.")

