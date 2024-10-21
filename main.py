# Importing the necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Helper function
# Encode the review
def preprocess_text(review):
    words = review.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function
def predict_sentiment(review):
    preprocessed_review = preprocess_text(review)
    print('--------------------------------------------------------------------')
    prediction = model.predict(preprocessed_review)
    print(prediction)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


# Streamlit app
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('This is a simple example of using a trained RNN model to predict the sentiment of an IMDB movie review.')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Enter your review here:')

if st.button('Classify'):
    sentiment, score = predict_sentiment(user_input)
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Review: {user_input}')
    st.write(f'Prediction Score: {score}')
else:
    st.write('Please enter a movie review to predict the sentiment.')