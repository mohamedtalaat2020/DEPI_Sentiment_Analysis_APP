import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from helper_functions import *

# Streamlit page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# Title and description with enhanced styling
st.markdown("""
    <div style="background-color: rgba(0, 0, 0, 0.7); padding: 30px; border-radius: 15px; 
                box-shadow: 0px 10px 50px rgba(0, 0, 0, 0.8); text-align: center;">
        <h1 style="color: white; font-size: 50px; margin: 0; text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.8);">
            Sentiment Analysis APP 😊🙁
        </h1>
    </div>
""", unsafe_allow_html=True)
st.markdown("### A User-Friendly Tool for Analyzing Customer Reviews")

# Horizontal radio buttons for model choice
col1, col2 = st.columns(2)
col1.write("Choose Sentiment Analysis Model:")
choose_model = col2.radio("", options=['Logistic Regression', 'LSTM'], horizontal=True)

# Model loading section
logistic_model = joblib.load('Logistic_Regression_model.pkl')
lstm_model = tf.keras.models.load_model('lstm_model.h5')
tokenizer = joblib.load('lstm_tokenizer.pkl')
vectorizer = joblib.load('TFIDF_model.pkl')

# Load preprocessed data
processed_df = pd.read_csv('preprocessed_data.csv', usecols=['Sentiment', 'clean_text'], delimiter=',')
X = processed_df[['clean_text']]
y = processed_df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X_train['clean_text'])

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemming = PorterStemmer()

def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation and word not in stop_words and not word.isdigit()]
    tokens = [stemming.stem(word) for word in tokens]
    return ' '.join(tokens)

# Batch Sentiment Analysis Section
st.write("### Batch Sentiment Analysis Section")
uploaded_file = st.file_uploader("Upload a CSV file for batch sentiment analysis")
st.write("Data must have a text column named 'text'.")

if uploaded_file:
    data = pd.read_csv(uploaded_file, encoding='utf-8')
    expected_columns = ['text']
    
    if set(expected_columns).issubset(data.columns):
        data['clean_text'] = data['text'].astype(str).apply(clean_text)
        text_data_tfidf = tfidf_vectorizer.transform(data['clean_text'])

        # Model selection and prediction
        if choose_model == 'Logistic Regression':
            predictions = logistic_model.predict(text_data_tfidf)
        elif choose_model == 'LSTM':
            tokenizer.fit_on_texts(data['clean_text'])
            sequence = tokenizer.texts_to_sequences(data['clean_text'])
            padded_sequence = pad_sequences(sequence, maxlen=150)
            predictions = lstm_model.predict(padded_sequence)
            predictions = (predictions > 0.5).astype("int32")

        sentiment_mapping = {0: "Negative", 1: "Positive"}
        data['Sentiment'] = predictions
        data['Sentiment'] = data['Sentiment'].map(sentiment_mapping)

        # Display results
        st.write("### Sentiment Analysis Results:")
        st.write(data[['text', 'clean_text', 'Sentiment']])

        # Summary section
        positive_count = data['Sentiment'].value_counts()['Positive']
        negative_count = data['Sentiment'].value_counts()['Negative']
        st.write("### Summary:")
        st.write(f"Total texts analyzed: {len(data)}")
        st.write(f"Positive: {positive_count}")
        st.write(f"Negative: {negative_count}")
    else:
        st.error(f"Incorrect columns. Expected: {expected_columns}, but got: {data.columns.tolist()}")

# Real-time Sentiment Analysis Section
st.write("### Real-time Sentiment Analysis Section")

with st.container():
    st.markdown("""
        <style>
            .stTextArea textarea {
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ccc;
                resize: vertical;
            }
        </style>
    """, unsafe_allow_html=True)
    user_input = st.text_area("Enter text for sentiment analysis:", placeholder="Type sentiment here...")

if st.button("Analyze Sentiment"):
    if user_input:
        user_input_cleaned = clean_text(user_input)
        X_test_tfidf = tfidf_vectorizer.transform([user_input_cleaned])

        if choose_model == 'Logistic Regression':
            prediction = logistic_model.predict(X_test_tfidf)
        elif choose_model == 'LSTM':
            tokenizer.fit_on_texts([user_input_cleaned])
            sequence = tokenizer.texts_to_sequences([user_input_cleaned])
            padded_sequence = pad_sequences(sequence, maxlen=150)
            prediction = lstm_model.predict(padded_sequence)
            prediction = (prediction > 0.5).astype("int32")

        sentiment = "Positive" if (prediction[0] if choose_model != 'LSTM' else prediction[0][0]) == 1 else "Negative"
        st.success(f"Prediction: {sentiment} ✔️") if sentiment == "Positive" else st.error(f"Prediction: {sentiment} ❌")
