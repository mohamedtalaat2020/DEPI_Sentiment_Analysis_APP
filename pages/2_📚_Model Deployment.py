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
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
# Streamlit page configuration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from helper_functions import *  

st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# Title and description
#st.title("Customer Product Reviews Sentiment Analysis App")
# app design
set_bg_hack('Picture1.png')
html_temp = """
  <div style="background-color: rgba(0, 0, 0, 0.7); 
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 10px 50px rgba(0, 0, 0, 0.8);
                text-align: center;">
        <h1 style="color: white; font-size: 50px; margin: 0; 
                   text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.8);">Sentiment Analysis APP 😊🙁</h1>
    </div>	"""
st.markdown(html_temp, unsafe_allow_html=True)
st.markdown("   ")
st.markdown("   ")
st.markdown("   ")



#choose_model = st.radio(
#    "**Choose your model:**",
#    ('RandomForest', 'XGBoost', 'Logistic Regression', 'LSTM'),
#    key="model_choice"  # Optional key for uniqueness
#)

# Define radio button options
model_options = ( 'Logistic Regression', 'LSTM')

# Create a horizontal layout container
col1, col2 = st.columns(2)

# Place the radio button label in the first column
col1.write("Choose Sentiment Analysis Model:")

# Create the radio button in the second column with horizontal alignment
choose_model = col2.radio("", options=model_options, horizontal=True)
#########################################################################################

# Load your trained models
#RF_model = joblib.load('Random_Forest_model.pkl')
#st.title(RF_model)
#XG_model = joblib.load('XGBoost_model.pkl')
logistic_model = joblib.load('Logistic_Regression_model.pkl')
lstm_model = tf.keras.models.load_model('lstm_model.h5')
tokenizer = joblib.load('lstm_tokenizer.pkl')
vectorizer=joblib.load('TFIDF_model.pkl') 

processed_df=pd.read_csv('preprocessed_data.csv', usecols=['Sentiment','clean_text'] ,delimiter=',')
# Split the data into training and testing sets
# Split data
X = processed_df[['clean_text']]
y = processed_df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and fit TF-IDF vectorizer on training data
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X_train['clean_text'])

# Transform training and test data
X_train_tfidf = tfidf_vectorizer.transform(X_train['clean_text'])
X_test_tfidf = tfidf_vectorizer.transform(X_test['clean_text'])

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('stemming', quiet=True)

# Common stop words
stop_words = set(stopwords.words('english'))
stemming = PorterStemmer()

###############################################################################################

def clean_text(text):
    # 1. Convert to lower
    text = text.lower()

    # 2. Split to words
    tokens = word_tokenize(text)

    # 3. Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]

    # 4. Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Remove numbers
    tokens = [word for word in tokens if not word.isdigit()]

    # 6. Apply Stemming
    tokens = [stemming.stem(word) for word in tokens]

    # To return these single words back into one string
    return ' '.join(tokens)

########################################################################################################

#user_input = st.text_area("Enter text for sentiment analysis:", placeholder="Type sentiment here...")
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
        # Preprocess the text using the vectorizer
        user_input = clean_text(user_input)
        X_test_tfidf = tfidf_vectorizer.transform([user_input])
        #user_input=vectorizer.transform([user_input])
        # Make the prediction and choose model
       # if choose_model == 'RandomForest':
        #    prediction = RF_model.predict(X_test_tfidf)
       # if choose_model == 'XGBoost':
        #    prediction = XG_model.predict(X_test_tfidf)
        if choose_model == 'Logistic Regression':
            prediction = logistic_model.predict(X_test_tfidf)
        elif choose_model == 'LSTM':
            # Tokenize and pad the sequence for LSTM (assuming LSTM expects padded sequences)
            #tokenizer = tf.keras.preprocessing.text.Tokenizer()
            tokenizer.fit_on_texts([user_input])  # Fit only for this example
            sequence = tokenizer.texts_to_sequences([user_input])
            padded_sequence = pad_sequences(sequence, maxlen=150)  # Assuming maxlen=100

            prediction = lstm_model.predict(padded_sequence)
            prediction = (prediction > 0.5).astype("int32")  # Convert probabilities to class labels


        # Debugging: Print the prediction to see the output format
    st.write(f"Raw prediction output: {prediction}") 


    	# Convert prediction to sentiment labels
    if choose_model == 'LSTM':
        	# For LSTM, we handle binary class prediction
        sentiment = "Positive" if prediction[0][0] == 1 else "Negative"
    else:
        # For non-LSTM models, prediction is typically a single value array (e.g., [1] or [0])
        sentiment = "Positive" if prediction[0] == 1 else "Negative"

    if sentiment == "Positive":
        st.success(f"Prediction: {sentiment} ✔️")
    else:
        st.error(f"Prediction: {sentiment} ❌")

            