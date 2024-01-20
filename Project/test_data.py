from keras.models import load_model
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Constants
VOCAB_SIZE = 10000  # Number of words to consider from the dataset
MAX_LEN = 150  # Maximum length of each sequence
EMBEDDING_DIM = 100  # Dimension of the embedding layer


def load_tokenizer(filename):
    with open(filename, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer


def preprocessing(test_data):
    # Load the tokenizer
    tokenizer = load_tokenizer('Models/tokenizer.pkl')

    # Preparing the test data
    test_sequences = tokenizer.texts_to_sequences(
        test_data['review_description'])
    X_test = pad_sequences(test_sequences, maxlen=MAX_LEN)

    return X_test

# Predicting on the test data


def predict_RNN(test_data, X_test):
    print("Predict Test File With RNN......")
    model = load_model('rnn3.h5')
    predictions = model.predict(X_test)
    # Adjusting indices back to -1, 0, 1
    predicted_classes = np.argmax(predictions, axis=1) - 1
    # Preparing the submission file
    submission = pd.DataFrame(
        {'ID': test_data['ID'], 'rating': predicted_classes})
    submission.to_csv('Rnn.csv', index=False)
    print("Submission file created successfully!")


def predict_Transformers(test_data, X_test):
    print("Predict Test File With Transformers......")
    transformer_model = load_model('transformer_model.h5')
    # Predict on the test data
    predictions = transformer_model.predict(X_test)
    # Adjusting indices back to -1, 0, 1
    predicted_classes = np.argmax(predictions, axis=1) - 1
    # Preparing the submission file
    submission = pd.DataFrame(
        {'ID': test_data['ID'], 'rating': predicted_classes})
    submission.to_csv('transformer_predictions.csv', index=False)
    print("Submission file created successfully!")


# Load the test dataset
test_data = pd.read_csv('test _no_label.csv')
X_test = preprocessing(test_data)
predict_RNN(test_data, X_test)
predict_Transformers(test_data, X_test)
