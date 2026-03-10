"""
Original file is located at 
https://colab.research.google.com/drive/1V9V-IxdM9fl9RW_odEdGU6aC441C6okG
"""

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk

text = "Sundar Pichai is the CEO of Google and he lives in California"

# Tokenization
tokens = word_tokenize(text)
print("Tokens: ", tokens)

# Stop word renewal
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
print("Filtered Tokens: ", filtered_tokens)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
print("Lemmatized Tokens: ", lemmatized_tokens)

# Named Entity Recognition (NER)
ner_tokens = ne_chunk(pos_tag(tokens))
print("Named Entities: ", ner_tokens)
