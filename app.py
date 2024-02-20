from flask import Flask, request, jsonify, render_template
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import tensorflow as tf
import numpy as np
import pickle
import praw
import pandas as pd
import json

MODEL_PATH = 'sentiment_analysis_model.h5'  # Replace with the path to your model
model = load_model(MODEL_PATH)

TOKENIZER_PATH = 'tokenizer.pickle'  # Replace with the path to your tokenizer
with open(TOKENIZER_PATH, 'rb') as file:
    tokenizer = pickle.load(file)

app = Flask(__name__)

reddit = praw.Reddit(client_id='', client_secret='', user_agent='') # Replace with your Reddit API credentials

#Homepage rendering
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['text'] 

    submission = reddit.submission(url=url)

    comments = []
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        comments.append(comment.body)

    # Define constants
    VOCAB_SIZE = 30000
    MAX_LEN = 250
    SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}

    # Analyze the sentiment of each comment
    sentiments = []
    positive_count = 0
    negative_count = 0
    for comment in comments:
        tokens = tokenizer.texts_to_sequences([comment])
        padded_tokens = pad_sequences(tokens, maxlen=MAX_LEN, padding='post', value=VOCAB_SIZE-1)
        prediction = np.argmax(model.predict(padded_tokens))

        sentiment = SENTIMENT_MAP.get(prediction, "Invalid prediction")
        sentiments.append(sentiment)

        if sentiment == 'positive':
            positive_count += 1
        elif sentiment == 'negative':
            negative_count += 1

    comments_and_sentiments = zip(comments, sentiments)

    return render_template('index.html', comments_and_sentiments=comments_and_sentiments, positive_count=positive_count, negative_count=negative_count)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

"""
Single comment analysis
@app.route('/', methods=['GET', 'POST'])
def index():
    VOCAB_SIZE = 10000
    MAX_LEN = 250
    sentiment = None
    if request.method == 'POST':
        text = request.form['text']

        # Preprocess the text
        tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
        tokens = [tokenizer.word_index[word] if word in tokenizer.word_index else 0 for word in tokens]
        padded_tokens = tf.keras.preprocessing.sequence.pad_sequences([tokens], maxlen=MAX_LEN, padding='post', value=VOCAB_SIZE-1)

        # Make a prediction
        prediction = np.argmax(model.predict(padded_tokens))

        # Map the prediction to a sentiment
        SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = SENTIMENT_MAP.get(prediction, "Invalid prediction")

    return render_template('index.html', sentiment=sentiment)
"""


