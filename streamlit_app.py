import streamlit as st
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Load your trained model
model = load_model('bidirectionalLSTM.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the maximum length for padding
max_length = 100  

# Define a function to make predictions
def predict_proba(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length)
    predictions = model.predict(padded)
    return predictions

# Streamlit UI
st.title("Text Classification with Bidirectional LSTM")
st.write("Enter text to classify")

# Input text from user
input_text = st.text_area("Text")

if st.button('Predict'):
    if input_text:
        prediction = predict_proba([input_text])[0][0]
        label = 'Discriminative' if prediction > 0.5 else 'Not Discriminative'
        st.write(f"Prediction probability: {prediction:.4f}")
        st.write(f"Classification: {label}")
    else:
        st.write("Please enter some text for prediction.")
