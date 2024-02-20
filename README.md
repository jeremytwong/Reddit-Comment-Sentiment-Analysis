# Reddit Sentiment Analysis

This is a Flask application that uses a pre-trained model to perform sentiment analysis on Reddit comments. It uses the PRAW (Python Reddit API Wrapper) to fetch comments from a given Reddit post and then applies sentiment analysis to each comment.

## Table of Contents

- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Data](#data)
    - [Data Format](#data-format)
    - [Data Usage](#data-usage)
- [Training the Model](#training-the-model)
    - [How it Works](#how-it-works)
    - [Model Usage](#model-usage)

## Features

- Fetches comments from a given Reddit post
- Performs sentiment analysis on each comment
- Displays the sentiment (positive, neutral, negative) of each comment
- Counts the number of positive and negative comments

## Setup

1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Replace the placeholders in the `app.py` file with your Reddit API credentials and the paths to your model and tokenizer.
4. Run the application: `python app.py`

## Usage

1. Open your web browser and navigate to `http://localhost:5000`.
2. Enter the URL of a Reddit post in the form and click 'Submit'.
3. The application will fetch the comments from the post and display their sentiments.

## Data

The model is trained on a dataset from Kaggle, which can be found [here](https://www.kaggle.com/datasets/kazanova/sentiment140). The dataset contains 1.6 million tweets extracted using the Twitter API. The tweets have been annotated (0 = negative, 2 = neutral, 4 = positive) and they can be used to detect sentiment.

### Data Format

The data is in CSV format with the following columns:

- `target`: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
- `ids`: the id of the tweet
- `date`: the date of the tweet
- `flag`: the query (lyx). If there is no query, then this value is 'NO_QUERY'.
- `user`: the user that tweeted
- `text`: the text of the tweet

### Data Usage

To use the data, you need to download it from the link above and place it in the path specified in the `config.json` file. The `train_model.py` script will then load the data from this file.

## Training the Model

The `train_model.py` script is used to train the sentiment analysis model. It loads a dataset from a CSV file, trains a TensorFlow model on the data, and saves the trained model to a file.

### How it Works

1. The script first loads the configuration parameters from a `config.json` file. These parameters include the vocabulary size, maximum sequence length, embedding dimension, and the paths to the model and data files.

2. The script then loads the data from the CSV file specified in the configuration. The data is shuffled and split into texts and labels.

3. A tokenizer is created and fitted on the texts. The texts are then converted to sequences and padded to the maximum sequence length.

4. The data is split into training and testing sets.

5. If a pre-trained model exists at the path specified in the configuration, the script loads that model. Otherwise, it trains a new model on the training data.

6. The script evaluates the model on the testing data and prints the accuracy.

7. Finally, the script enters a loop where it prompts the user to enter a sentence for sentiment analysis. The sentence is encoded and fed into the model, and the predicted sentiment is printed to the console.

### Model Usage

To run the script, use the following command:

```bash
python train_model.py