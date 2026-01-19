## 🏨 Hotel Reviews Sentiment Analysis using NLP

This project performs sentiment analysis on hotel reviews using both traditional Machine Learning models and a pretrained Transformer (RoBERTa) model.
A Streamlit web application is built to allow users to input hotel reviews and instantly view sentiment predictions.

## 📌 Project Overview

Customer reviews play a crucial role in the hospitality industry.
This project analyzes hotel reviews and classifies them into Positive, Negative, or Neutral sentiments.

We implement and compare:

Traditional NLP + ML models

A pretrained Transformer model (RoBERTa)

An interactive Streamlit UI for real-time prediction

## 🧠 Models Used
# 🔹 Traditional Machine Learning Models

Logistic Regression

Multinomial Naive Bayes

Linear Support Vector Classifier (Linear SVC)

Text Representation:

TF-IDF Vectorization (Unigrams + Bigrams)

# 🔹 Pretrained Transformer Model

Model: cardiffnlp/twitter-roberta-base-sentiment

Framework: Hugging Face Transformers

Output Classes:

Positive

Neutral

Negative

## 🗂 Dataset

Dataset: Hotel Reviews Dataset

Features Used:

Positive Review

Negative Review

Reviewer Score

Target Labels:

Positive (Score ≥ 7)

Neutral (Score 5–6)

Negative (Score ≤ 4)

## 🔧 Text Preprocessing

The following preprocessing steps were applied:

Lowercasing

Punctuation removal

Stopword removal (NLTK)

Stemming (Snowball Stemmer)

TF-IDF feature extraction

## ⚠️ Note:
Pretrained Transformer models are tested using raw (unprocessed) reviews for best performance.

## 📊 Exploratory Data Analysis (EDA)

Sentiment distribution visualization

Review length analysis

Reviewer score vs sentiment boxplots

Word frequency analysis

## 🌐 Streamlit Web Application

The Streamlit app allows users to:

Enter a hotel review

Get sentiment prediction from:

Traditional ML model

Pretrained RoBERTa model

View confidence scores

Compare both model outputs

## 🚀 How to Run the Project Locally
1️⃣ Clone the Repository
git clone https://github.com/your-username/hotel-reviews-sentiment-analysis-nlp.git
cd hotel-reviews-sentiment-analysis-nlp

2️⃣ Create Virtual Environment (Optional but Recommended)
python -m venv myenv
source myenv/bin/activate   # Linux/Mac
myenv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Streamlit App
streamlit run app.py

## 📦 Requirements

Key libraries used:

Python 3.9+

pandas

numpy

scikit-learn

nltk

joblib

transformers

torch

streamlit

matplotlib

seaborn

(Full list available in requirements.txt)

## 📁 Project Structure
hotel-reviews-sentiment-analysis-nlp/
│
├── app.py                       # Streamlit application
├── sentiment_model.pkl          # Trained ML model
├── tfidf_vectorizer.pkl         # TF-IDF vectorizer
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
├── Hotel_Reviews.csv            # Dataset
└── notebooks/
    └── model_training.ipynb     # Model training & EDA

## ✅ Results
Model	Accuracy
Logistic Regression	~High
Naive Bayes	~Moderate
Linear SVC	~High
RoBERTa (Pretrained)	Best Performance

⚡ Transformer model provides superior contextual understanding compared to traditional models.
