from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import json

app = Flask(__name__)

# Load your trained model
model = load_model('bidirectionalLSTM.h5')

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

max_length = 100

# Define a function that predicts probabilities
def predict_proba(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length)
    predictions = model.predict(padded)
    return np.hstack((1 - predictions, predictions))

# Initialize the LIME explainer
class_names = ['Not Discriminative', 'Discriminative']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_instance = [request.form['text']]
    prediction = model.predict(pad_sequences(tokenizer.texts_to_sequences(text_instance), maxlen=max_length))[0][0]
    label = 'Discriminative' if prediction > 0.5 else 'Not Discriminative'
    
    return render_template('result.html', text=text_instance[0], label=label, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
