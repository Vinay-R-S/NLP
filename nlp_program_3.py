# Simple LSTM sentiment classification (Custom data)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import numpy as np

def custom_dataset():
    # Sample text dataset (product reviews)
    texts = [
        "product is very good",
        "excellent quality and nice",
        "I love this product",
        "bad product",
        "very poor quality",
        "I hate this item"
    ]

    # Labels (1 for positive sentiment, 0 for negative)
    labels = [1, 1, 1, 0, 0, 0]

    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Padding
    x = pad_sequences(sequences, maxlen=5)
    y = np.array(labels)

    # Model
    model = Sequential()
    model.add(Embedding(input_dim=50, output_dim=8, input_length=5))
    model.add(LSTM(8))
    model.add(Dense(1, activation='sigmoid'))

    # Compile
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x, y, epochs=50, verbose=0)

    # Test with new sentence
    test_text = ['good quality product']
    test_sequences = tokenizer.texts_to_sequences(test_text)
    test_pad = pad_sequences(test_sequences, maxlen=5)

    prediction = model.predict(test_pad)
    print(prediction)

    print("Sentence: ", test_text[0])
    print("Sentiment: ", "Positive" if prediction[0][0] > 0.5 else "Negative")

def IMDB_dataset():
    # Load IMDB dataset
    num_words = 10000
    maxlen = 100
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

    # Padding
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)

    # Model
    model = Sequential()
    model.add(Embedding(input_dim=num_words, output_dim=32, input_length=maxlen))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))

    # Compile and Train
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

    # Test
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"IMDB Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    custom_dataset()
    IMDB_dataset()
