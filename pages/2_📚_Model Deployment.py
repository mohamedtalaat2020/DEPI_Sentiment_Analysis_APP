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
###################################Batch Sentiment Analysis#####################################################        
# Centered, styled header for "Batch Sentiment Analysis Section" with black text and no background
batch_analysis_header = """
    <div style="text-align: center; 
                margin-top: 20px; 
                margin-bottom: 20px;">
        <h2 style="color: black; font-size: 30px; margin: 0;">
            Batch Sentiment Analysis Section
        </h2>
    </div>
"""
st.markdown(batch_analysis_header, unsafe_allow_html=True)

# File upload for batch sentiment analysis
st.write("                                                                                     ")
uploaded_file = st.file_uploader("Upload a CSV file for batch sentiment analysis")
st.write("Data must have a text column with name is 'text' ")

if uploaded_file:
    # Read the file into a DataFrame
    data = pd.read_csv(uploaded_file, encoding='utf-8')    #, encoding='ISO-8859-1'
    # Expected column names
    expected_columns = ['text']
    
    if set(expected_columns).issubset(data.columns):
        st.write("The uploaded file has the correct column!")
        # st.write(data.head())
        
        # Make predictions 
        # Transform the text using the TF-IDF vectorizer
        data['clean_text'] = data['text'].astype(str).apply(lambda x: clean_text(x))
        text_data_tfidf = tfidf_vectorizer.transform(data['clean_text'])

        # Predict sentiments for the text data
        
        if choose_model == 'Logistic Regression':
            predictions = logistic_model.predict(text_data_tfidf)
        elif choose_model == 'LSTM':
            tokenizer.fit_on_texts(data['clean_text'])  # Fit only for this example
            sequence = tokenizer.texts_to_sequences(data['clean_text'])
            padded_sequence = pad_sequences(sequence, maxlen=150)  # Assuming maxlen=100

            predictions = lstm_model.predict(padded_sequence)
            predictions = (predictions > 0.5).astype("int32")

        # Add predictions to the DataFrame
        data['Sentiment'] = predictions 
        
            # Add predictions to the DataFrame
        sentiment_mapping = {
            0: "Negative",
            1: "Positive"
        }
        data['Sentiment'] = data['Sentiment'].map(sentiment_mapping)

        st.write("### Sentiment Analysis Results:")
        st.write(data[['text','clean_text','Sentiment']])

        # Summary Section
        positive_count = data['Sentiment'].value_counts()['Positive']
        negative_count = data['Sentiment'].value_counts()['Negative']

        st.write("### Summary:")
        st.write(f"Total texts analyzed: {len(data)}")
        st.write(f"Positive: {positive_count}")
        st.write(f"Negative: {negative_count}")
    else:
        st.error(f"The uploaded file doesn't have the correct columns. Expected: {expected_columns}, but got: {data.columns.tolist()}")

# User feedback section
#st.write("### Feedback")
# feedback = st.text_input("How accurate was the prediction? (1-5)", placeholder='Rating here...')
#options = [1, 2, 3, 4, 5]
#feedback = st.selectbox("How accurate was the prediction? (1-5)", options, index=None)

#if st.button("Submit Feedback"):
#    st.write("Thank you for your feedback!")

# Footer or final notes
#st.write("App built with Streamlit.")
st.write("__________________________________________________________________________________________")
########################################################################################################
st.write("### Real-time Sentiment Analysis Section")

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


            