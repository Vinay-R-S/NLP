# 1. Import Libraries
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, Dense, Input, Layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np

# 2. Sample Dataset (Simple)
texts = [
    "cloud security prevents attacks",
    "malware attack detected in system",
    "this product is very good",
    "i am happy with the service",
    "system failure due to unauthorized access",
    "very bad experience and poor support"
]

# Labels: 1 = Positive / Secure, 0 = Negative / Insecure
labels = [1, 0, 1, 1, 0, 0]

# 3. Tokenization & Padding
vocab_size = 5000
max_len = 10
embedding_dim = 100

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(labels)

# 4. Custom Attention Layer
class Attention(Layer):
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        scores = K.softmax(K.sum(inputs, axis=2))
        scores = K.expand_dims(scores, axis=-1)
        context = inputs * scores
        return K.sum(context, axis=1)

# 5. CNN + Attention Model
inputs = Input(shape=(max_len,))

x = Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    input_length=max_len
)(inputs)

x = Conv1D(
    filters=128,
    kernel_size=3,
    activation='relu'
)(x)

x = Attention()(x)

x = Dense(64, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# 6. Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 7. Train Model
model.fit(
    X,
    y,
    epochs=10,
    batch_size=2,
    verbose=1
)

# 8. Test with New Sentence
test_text = ["unauthorized malware attack detected"]

test_seq = tokenizer.texts_to_sequences(test_text)
test_pad = pad_sequences(test_seq, maxlen=max_len)

prediction = model.predict(test_pad)

print("\nPrediction:", prediction)
print(
    "Class:",
    "Positive / Secure" if prediction[0][0] > 0.5 else "Negative / Insecure"
)
