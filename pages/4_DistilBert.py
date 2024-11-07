import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load model and tokenizer (replace with your model path)
model_path = './DistilBert_Model'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
    st.write("✅ GPU detected! Using GPU for faster processing.")
else:
    device = torch.device("cpu")  # Fall back to CPU
    st.write("⚠️ No GPU detected. Using CPU for processing.")

# Move model to the selected device
model = model.to(device)

# Function to prepare input data for the model
def prepare_input(review):
    inputs = tokenizer(review, padding=True, truncation=True, max_length=256, return_tensors='pt')
    # Move input tensors to the selected device
    return {key: val.to(device) for key, val in inputs.items()}

# Streamlit app
st.title("Sentiment Analysis with DistilBert")

# Get user input
user_input = st.text_input("Enter a review:")

if user_input:
    # Prepare input data
    encodings = prepare_input(user_input)
    
    # Make prediction using the model
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    # Display result
    sentiment = "positive" if prediction == 1 else "negative"
    st.write(f"Predicted Sentiment: {sentiment}")
