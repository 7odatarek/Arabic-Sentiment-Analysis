from keras.utils import to_categorical
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dense, Dropout, Flatten, Input, LSTM
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import LayerNormalization, MultiHeadAttention
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
import re
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from pyarabic.araby import strip_tashkeel, strip_tatweel, normalize_hamza
from nltk.tokenize import word_tokenize
import pickle

# Constants
VOCAB_SIZE = 10000  # Number of words to consider from the dataset
MAX_LEN = 150  # Maximum length of each sequence
EMBEDDING_DIM = 100  # Dimension of the embedding layer


def save_tokenizer(tokenizer, filename):
    with open(filename, 'wb') as file:
        pickle.dump(tokenizer, file)


def preprocess(text):
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ة', 'ه').replace(
        'ى', 'ي').replace('ؤ', 'ء').replace('ئ', 'ء')
    text = text.replace('ـ', '').lower()
    return text


def preprocessing(train_data):
    train_data['review_description'] = train_data['review_description'].apply(
        preprocess)
    train_data['review_description'] = train_data['review_description'].apply(
        word_tokenize)
    stop_words_arabic = stopwords.words('arabic')
    stop_words_english = stopwords.words('english')
    train_data['review_description'] = train_data['review_description'].apply(
        lambda x: [item for item in x if item not in stop_words_arabic and item not in stop_words_english])

    # Remove Tatweel, Tashkeel, normalize Hamza, and perform stemming
    st = ISRIStemmer()
    train_data['review_description'] = train_data['review_description'].apply(
        lambda x: [st.stem(normalize_hamza(strip_tatweel(strip_tashkeel(item)))) for item in x])

    # Remove Arabic and English punctuation
    arabic_punctuation_pattern = re.compile("[،.؟]")
    english_punctuation_pattern = re.compile("[,?.]")
    train_data['review_description'] = train_data['review_description'].apply(
        lambda x: [arabic_punctuation_pattern.sub("", item) for item in x])
    train_data['review_description'] = train_data['review_description'].apply(
        lambda x: [english_punctuation_pattern.sub("", item) for item in x])

    # Remove exclamation marks
    train_data['review_description'] = train_data['review_description'].apply(
        lambda x: [item.replace('!', '') for item in x])

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    train_data['review_description'] = train_data['review_description'].apply(
        lambda x: [lemmatizer.lemmatize(item) for item in x])
    # Adjusting ratings from -1, 0, 1 to 0, 1, 2 for to_categorical
    train_data['rating'] = train_data['rating'].apply(lambda x: x + 1)

    # Preprocessing the data
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(train_data['review_description'])
    # save tokenizer in Models folder
    save_tokenizer(tokenizer, 'Models/tokenizer.pkl')
    print("---------------------")
    print(train_data['review_description'])
    sequences = tokenizer.texts_to_sequences(train_data['review_description'])
    X_train = pad_sequences(sequences, maxlen=MAX_LEN)

    # Convert labels to categorical
    y_train = to_categorical(train_data['rating'], num_classes=3)

    # Splitting the train data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val


# positional encoding
def positional_encoding(seq_len, d_model):
    pos_enc = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    # Add an extra dimension for broadcasting
    div_term = div_term[np.newaxis, :]
    pos_enc[:, 0::2] = np.sin(position * div_term[:, :d_model // 2])
    pos_enc[:, 1::2] = np.cos(position * div_term[:, :d_model // 2])
    return pos_enc


def build_transformer_model(vocab_size, seq_len, d_model, num_heads, ff_dim, num_classes):
    inputs = Input(shape=(seq_len,), dtype=tf.int32)

    # Input Embedding Layer
    embedding_layer = Embedding(
        input_dim=vocab_size, output_dim=d_model)(inputs)
    embedding_layer = LayerNormalization(epsilon=1e-6)(embedding_layer)
    embedding_layer = Dropout(0.1)(embedding_layer)

    # Positional Encoding Layer
    pos_enc = positional_encoding(seq_len, d_model)
    pos_enc_layer = embedding_layer + pos_enc
    # Multi-Head Attention Layer
    multi_head_attention_layer = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(pos_enc_layer, pos_enc_layer,
                                                                                          pos_enc_layer)

    multi_head_attention_layer = Dropout(0.1)(multi_head_attention_layer)
    multi_head_attention_layer = LayerNormalization(
        epsilon=1e-6)(pos_enc_layer + multi_head_attention_layer)

    # Feed-Forward Layer
    ff_layer = Dense(ff_dim, activation='relu')(multi_head_attention_layer)
    ff_layer = Dropout(0.1)(ff_layer)
    ff_layer = Dense(d_model)(ff_layer)
    ff_layer = Dropout(0.1)(ff_layer)
    ff_layer = LayerNormalization(
        epsilon=1e-6)(multi_head_attention_layer + ff_layer)

    # Flatten and Dense Layers
    flatten_layer = Flatten()(ff_layer)
    output_layer = Dense(num_classes, activation='softmax')(flatten_layer)

    model = Model(inputs=inputs, outputs=output_layer)
    return model


def transformer_model(train_data):
    X_train, X_val, y_train, y_val = preprocessing(train_data)

    vocab_size = VOCAB_SIZE
    seq_len = MAX_LEN
    d_model = EMBEDDING_DIM
    num_heads = 5
    num_classes = 3
    ff_dim = 32

    transformer_model = build_transformer_model(
        vocab_size, seq_len, d_model, num_heads, ff_dim, num_classes)

    # Ensure that the labels have the correct shape for each batch
    batch_size = 32
    num_batches_train = len(y_train) // batch_size

    # Trim the data so that it's a multiple of the batch size
    X_train = X_train[:num_batches_train * batch_size]
    y_train = y_train[:num_batches_train * batch_size]

    print(transformer_model.summary())

    transformer_model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    transformer_model.fit(X_train, y_train, epochs=5,
                          batch_size=64, validation_split=0.2)

    # Save the model
    transformer_model.save('Models/transformer_model.h5')
    # print loss and accuracy
    loss, accuracy = transformer_model.evaluate(X_val, y_val, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))
    print('Loss: %f' % (loss))


def RNN(train_data):
    X_train, X_val, y_train, y_val = preprocessing(train_data)
    # Building the RNN model
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dense(3, activation='softmax'))  # 3 classes: 0, 1, 2

    # Compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    # Training the model
    model.fit(X_train, y_train, validation_data=(
        X_val, y_val), epochs=5, batch_size=64)
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))
    print('Loss: %f' % (loss))
    model.save('Models/rnn.h5')


# Load the train and test datasets
train_data = pd.read_excel('Dataset/train.xlsx')
# LSTM model
RNN(train_data)
# Transformer model
transformer_model(train_data)
