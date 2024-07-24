import streamlit as st
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB

# Load the trained model from the pickle file
model_path = 'email_sentiment_analysis.pkl'
with open(model_path, 'rb') as file:
    nb_model = pickle.load(file)

# Function for text preprocessing
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\W', ' ', text)   # Remove special characters
    text = text.lower()               # Convert to lowercase
    return text

# Streamlit interface
st.title('Email Sentiment Analysis')

st.write("Enter the email content below and click 'Analyze' to predict the sentiment.")

email_input = st.text_area('Email Content', height=200)

if st.button('Analyze'):
    if email_input:
        preprocessed_text = preprocess_text(email_input)
        prediction = nb_model.predict([preprocessed_text])
        predicted_label = 'Positive' if prediction[0] == 1 else 'Negative'
        
        st.write(f'The sentiment of the email is: **{predicted_label}**')
    else:
        st.write('Please enter the email content.')

# Run the app using: streamlit run your_script.py
