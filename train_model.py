import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json
import pandas as pd


def load_data(file_path):
    try:
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    return (data.sample(frac=1).reset_index(drop=True))

def train_model(train_data, train_labels, VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, MODEL_PATH):
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes: negative, neutral, positive
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2)
    model.save(MODEL_PATH)


def main():
    with open('config.json') as config_file:
        config = json.load(config_file)

    VOCAB_SIZE = config['VOCAB_SIZE']
    MAX_LEN = config['MAX_LEN']
    EMBEDDING_DIM = config['EMBEDDING_DIM']
    MODEL_PATH = config['MODEL_PATH']
    DATA_PATH = config['DATA_PATH']
    OOV_TOKEN = "<OOV>"

    df_shuffled = load_data(DATA_PATH)

    texts = []
    labels = []

    for _, row in df_shuffled.iterrows():
        texts.append(row[-1])
        label = row[0]
        labels.append(0 if label == 0 else 1 if label == 2 else 2)

    texts = np.array(texts)
    labels = np.array(labels)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE + 1, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, value=VOCAB_SIZE-1, padding='post')

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    train_data = padded_sequences[:-5000]
    test_data = padded_sequences[-5000:]
    train_labels = labels[:-5000]
    test_labels = labels[-5000:]

    model = None

    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        model = load_model(MODEL_PATH)
    else:
        print("Training a new model...")
        train_model(train_data, train_labels, VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, MODEL_PATH)

    loss, accuracy = model.evaluate(test_data, test_labels)

    print(f"Test accuracy: {accuracy * 100:.2f}%")

    def encode_text(text):
        tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
        tokens = [tokenizer.word_index.get(word, tokenizer.word_index[OOV_TOKEN]) for word in tokens]
        return pad_sequences([tokens], maxlen=MAX_LEN, padding='post', value=VOCAB_SIZE-1)

    SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}

    while True:
        try:
            user_input = input("Enter a sentence for sentiment analysis (or 'leave' to quit): ")
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

        if user_input.lower() == 'leave':
            break

        encoded_input = encode_text(user_input)
        prediction = np.argmax(model.predict(encoded_input))

        print(SENTIMENT_MAP.get(prediction, "Invalid prediction"))

if __name__ == "__main__":
    main()