# NLP Comprehensive Study Guide

## Quiz Preparation Material

---

# Table of Contents

1. [What is NLP?](#what-is-nlp)
2. [Program 1: Text Preprocessing & NER](#program-1-text-preprocessing--ner)
3. [Program 2: Word Representation Techniques](#program-2-word-representation-techniques)
4. [Program 3: LSTM Sentiment Analysis](#program-3-lstm-sentiment-analysis)
5. [Key Concepts Summary](#key-concepts-summary)
6. [Practice MCQs (20 Questions)](#practice-mcqs-20-questions)
7. [MCQ Answer Key](#mcq-answer-key)

---

# What is NLP?

## Definition

**Natural Language Processing (NLP)** is a branch of Artificial Intelligence that enables computers to understand, interpret, and generate human language. It bridges the gap between human communication and computer understanding.

## How NLP Works - The Pipeline

```
Raw Text → Preprocessing → Feature Extraction → Model → Output
```

### Step-by-Step NLP Pipeline:

1. **Text Acquisition** - Collecting raw text data
2. **Text Preprocessing** - Cleaning and preparing text
   - Tokenization
   - Stop word removal
   - Lemmatization/Stemming
3. **Feature Extraction** - Converting text to numbers
   - Bag of Words (BoW)
   - TF-IDF
   - Word Embeddings (Word2Vec, GloVe)
4. **Model Training** - Using ML/DL algorithms
5. **Prediction/Output** - Generating results

---

# Program 1: Text Preprocessing & NER

## 1. Tokenization

### What is it?

**Tokenization** is the process of breaking text into smaller units called **tokens** (words, sentences, or subwords).

### Code Example:

```python
from nltk.tokenize import word_tokenize
text = "Sundar Pichai is the CEO of Google"
tokens = word_tokenize(text)
# Output: ['Sundar', 'Pichai', 'is', 'the', 'CEO', 'of', 'Google']
```

### Key Points:

- First step in NLP preprocessing
- Types: Word tokenization, Sentence tokenization
- Punctuation marks are also treated as separate tokens
- Uses `word_tokenize()` from NLTK

---

## 2. Stop Words Removal

### What is it?

**Stop words** are common words that don't carry significant meaning (e.g., "is", "the", "of", "and"). Removing them reduces noise and computational overhead.

### Code Example:

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
# Removes: 'is', 'the', 'of'
```

### Key Points:

- NLTK provides stopwords for multiple languages
- Reduces vocabulary size
- Improves processing efficiency
- Common stop words: a, an, the, is, are, was, were, in, on, at, etc.

---

## 3. Lemmatization

### What is it?

**Lemmatization** reduces words to their **base/root form** (lemma) using vocabulary and morphological analysis.

### Code Example:

```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized = lemmatizer.lemmatize("running")  # Returns: "run"
lemmatized = lemmatizer.lemmatize("better", pos='a')  # Returns: "good"
```

### Lemmatization vs Stemming:

| Feature | Lemmatization     | Stemming                    |
| ------- | ----------------- | --------------------------- |
| Output  | Valid words       | May not be valid words      |
| Method  | Dictionary-based  | Rule-based (suffix removal) |
| Speed   | Slower            | Faster                      |
| Example | "running" → "run" | "running" → "runn"          |

### Key Points:

- Returns dictionary words (actual lemmas)
- Uses WordNet lexical database
- More accurate than stemming
- Considers Part-of-Speech (POS) for better results

---

## 4. Part-of-Speech (POS) Tagging

### What is it?

**POS Tagging** assigns grammatical tags to each word (noun, verb, adjective, etc.).

### Common POS Tags:

| Tag | Meaning     | Example           |
| --- | ----------- | ----------------- |
| NN  | Noun        | "dog", "city"     |
| VB  | Verb        | "run", "eat"      |
| JJ  | Adjective   | "beautiful"       |
| RB  | Adverb      | "quickly"         |
| NNP | Proper Noun | "Google", "India" |

### Code Example:

```python
from nltk import pos_tag
tagged = pos_tag(tokens)
# Output: [('Sundar', 'NNP'), ('Pichai', 'NNP'), ('is', 'VBZ'), ...]
```

---

## 5. Named Entity Recognition (NER)

### What is it?

**NER** identifies and classifies named entities in text into predefined categories like:

- **PERSON** - Names of people
- **ORGANIZATION** - Companies, agencies
- **GPE** (Geo-Political Entity) - Countries, cities
- **DATE** - Dates and times
- **MONEY** - Monetary values

### Code Example:

```python
from nltk import pos_tag, ne_chunk
ner_tokens = ne_chunk(pos_tag(tokens))
# Identifies: Sundar Pichai (PERSON), Google (ORGANIZATION), California (GPE)
```

### Key Points:

- Requires POS tagging first
- Uses IOB notation (Inside, Outside, Beginning)
- Applications: Information extraction, Question answering, Chatbots

---

# Program 2: Word Representation Techniques

> **Why convert words to numbers?**
> Machines can only process numbers. We need to convert text into numerical vectors (embeddings) for ML models.

---

## 1. Bag of Words (BoW)

### What is it?

**BoW** represents text as a collection of word frequencies, ignoring grammar and word order.

### How it works:

1. Create vocabulary of all unique words
2. Represent each document as a vector of word counts

### Code Example:

```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = ["Cyber security prevents attacks",
          "Malware attacks computer systems"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
```

### Example Output:

| Document | attacks | computer | cyber | malware | prevents | security | systems |
| -------- | ------- | -------- | ----- | ------- | -------- | -------- | ------- |
| Doc 1    | 1       | 0        | 1     | 0       | 1        | 1        | 0       |
| Doc 2    | 1       | 1        | 0     | 1       | 0        | 0        | 1       |

### Advantages:

- Simple and easy to implement
- Works well for small datasets

### Disadvantages:

- Ignores word order and context
- High-dimensional sparse vectors
- All words treated equally important

---

## 2. TF-IDF (Term Frequency - Inverse Document Frequency)

### What is it?

**TF-IDF** weighs words based on their importance in a document relative to the entire corpus.

### Formula:

```
TF-IDF = TF × IDF

TF (Term Frequency) = (Number of times term appears in document) / (Total terms in document)

IDF (Inverse Document Frequency) = log(Total documents / Documents containing the term)
```

### Key Insight:

- **High TF-IDF** = Word is frequent in this document but rare in other documents → **Important!**
- **Low TF-IDF** = Word appears in many documents → **Less important**

### Code Example:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
```

### Advantages:

- Considers word importance
- Reduces weight of common words
- Better than BoW for most tasks

### Disadvantages:

- Still ignores word order
- No semantic understanding

---

## 3. Word2Vec

### What is it?

**Word2Vec** is a neural network-based technique that learns **dense vector representations** capturing semantic meaning.

### Key Concept: "Words that appear in similar contexts have similar meanings"

### Two Architectures:

| Architecture                       | Description                       | Best For       |
| ---------------------------------- | --------------------------------- | -------------- |
| **CBOW** (Continuous Bag of Words) | Predicts target word from context | Frequent words |
| **Skip-gram**                      | Predicts context from target word | Rare words     |

### Code Example:

```python
from gensim.models import Word2Vec
model = Word2Vec(sentences=tokenized_corpus,
                 vector_size=50,    # Dimension of vectors
                 window=3,          # Context window size
                 min_count=1,       # Minimum word frequency
                 sg=1)              # 1=Skip-gram, 0=CBOW

# Find similar words
model.wv.most_similar("security")
```

### Key Parameters:

- **vector_size**: Dimension of word vectors (50, 100, 300)
- **window**: Number of context words on each side
- **min_count**: Ignores words with frequency less than this
- **sg**: 0=CBOW, 1=Skip-gram

### Advantages:

- Captures semantic relationships
- Dense vectors (efficient)
- King - Man + Woman = Queen

### Disadvantages:

- Requires large training corpus
- One vector per word (no handling of polysemy)

---

## 4. GloVe (Global Vectors for Word Representation)

### What is it?

**GloVe** combines global statistical information with local context window methods.

### How it works:

- Uses word co-occurrence matrix
- Learns vectors that encode co-occurrence probabilities
- Pre-trained on large corpora (Wikipedia, Common Crawl)

### Code Example:

```python
import gensim.downloader as api
glove_model = api.load("glove-wiki-gigaword-50")
glove_model.most_similar("security")
```

### GloVe vs Word2Vec:

| Feature     | Word2Vec                       | GloVe                          |
| ----------- | ------------------------------ | ------------------------------ |
| Training    | Sliding window (local context) | Global co-occurrence matrix    |
| Computation | Faster for small data          | Better for large data          |
| Pre-trained | Available                      | More commonly used pre-trained |

---

## Comparison of Word Representations

| Method   | Type                 | Captures Meaning? | Vector Type | Order Matters? |
| -------- | -------------------- | ----------------- | ----------- | -------------- |
| BoW      | Count-based          | No                | Sparse      | No             |
| TF-IDF   | Count-based          | No                | Sparse      | No             |
| Word2Vec | Neural               | Yes               | Dense       | No (per word)  |
| GloVe    | Statistical + Neural | Yes               | Dense       | No (per word)  |

---

# Program 3: LSTM Sentiment Analysis

## 1. What is Sentiment Analysis?

**Sentiment Analysis** (Opinion Mining) determines the emotional tone behind text:

- **Positive**: "I love this product!"
- **Negative**: "This is terrible"
- **Neutral**: "The product arrived today"

---

## 2. Deep Learning for NLP

### Why Deep Learning?

- Automatically learns features
- Captures sequential patterns
- Better performance on large datasets

---

## 3. Embedding Layer

### What is it?

Converts integer-encoded words into dense vectors (like Word2Vec but learned during training).

### Code:

```python
from tensorflow.keras.layers import Embedding
model.add(Embedding(input_dim=50,      # Vocabulary size
                    output_dim=8,       # Embedding dimension
                    input_length=5))    # Sequence length
```

### Key Parameters:

- **input_dim**: Size of vocabulary
- **output_dim**: Dimension of embedding vectors
- **input_length**: Length of input sequences

---

## 4. LSTM (Long Short-Term Memory)

### What is it?

**LSTM** is a type of Recurrent Neural Network (RNN) designed to capture long-term dependencies in sequential data.

### Why LSTM over RNN?

- Solves **vanishing gradient problem**
- Remembers information over long sequences
- Uses **gates** to control information flow

### LSTM Gates:

| Gate            | Purpose                               |
| --------------- | ------------------------------------- |
| **Forget Gate** | Decides what to discard from memory   |
| **Input Gate**  | Decides what new information to store |
| **Output Gate** | Decides what to output                |

### Code:

```python
from tensorflow.keras.layers import LSTM
model.add(LSTM(32))  # 32 = number of LSTM units
```

---

## 5. Text Preprocessing for Deep Learning

### a) Tokenization (Keras)

```python
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
```

### b) Padding

Makes all sequences the same length by adding zeros.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
x = pad_sequences(sequences, maxlen=5)
```

**Example:**

```
Before: [1, 2, 3]
After:  [0, 0, 1, 2, 3]  # padded to length 5
```

---

## 6. Complete LSTM Model Architecture

```python
model = Sequential()
model.add(Embedding(input_dim=50, output_dim=8, input_length=5))
model.add(LSTM(8))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

### Layer Breakdown:

1. **Embedding Layer**: Converts words to dense vectors
2. **LSTM Layer**: Processes sequence and captures patterns
3. **Dense Layer**: Output layer with sigmoid for binary classification

### Key Terms:

- **optimizer='adam'**: Adaptive learning rate optimizer
- **loss='binary_crossentropy'**: Loss function for binary classification
- **activation='sigmoid'**: Outputs probability between 0 and 1

---

## 7. IMDB Dataset

### About:

- 50,000 movie reviews (25k train, 25k test)
- Binary labels (positive/negative)
- Pre-tokenized and encoded as integers

### Code:

```python
from tensorflow.keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
```

---

# Key Concepts Summary

## NLP Pipeline Flowchart

```
                    ┌─────────────────┐
                    │    Raw Text     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Tokenization   │
                    └────────┬────────┘
                             │
                    ┌────────▼─────────┐
                    │ Stop Word Removal│
                    └────────┬─────────┘
                             │
                    ┌────────▼────────┐
                    │  Lemmatization  │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐          ┌─────────▼─────────┐
     │   BoW/TF-IDF    │          │  Word Embeddings  │
     │    (Sparse)     │          │     (Dense)       │
     └────────┬────────┘          └─────────┬─────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │   ML/DL Model   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Prediction    │
                    └─────────────────┘
```

---

## Quick Reference Table

| Concept       | Library | Key Function                 |
| ------------- | ------- | ---------------------------- |
| Tokenization  | NLTK    | `word_tokenize()`            |
| Stop Words    | NLTK    | `stopwords.words('english')` |
| Lemmatization | NLTK    | `WordNetLemmatizer()`        |
| POS Tagging   | NLTK    | `pos_tag()`                  |
| NER           | NLTK    | `ne_chunk()`                 |
| Bag of Words  | Sklearn | `CountVectorizer()`          |
| TF-IDF        | Sklearn | `TfidfVectorizer()`          |
| Word2Vec      | Gensim  | `Word2Vec()`                 |
| GloVe         | Gensim  | `api.load()`                 |
| LSTM          | Keras   | `LSTM()`                     |
| Embedding     | Keras   | `Embedding()`                |
| Padding       | Keras   | `pad_sequences()`            |

---

# Practice MCQs (20 Questions)

## Text Preprocessing (Questions 1-5)

**Q1. What is the primary purpose of tokenization in NLP?**

- A) To remove stop words from text
- B) To break text into smaller units like words or sentences
- C) To convert words to their root form
- D) To identify named entities

**Q2. Which of the following is NOT a stop word in English?**

- A) the
- B) is
- C) computer
- D) and

**Q3. What is the difference between lemmatization and stemming?**

- A) Lemmatization is faster than stemming
- B) Stemming always produces valid dictionary words
- C) Lemmatization produces valid dictionary words, stemming may not
- D) There is no difference

**Q4. In NLTK's POS tagging, what does the tag 'NNP' represent?**

- A) Noun
- B) Verb
- C) Proper Noun
- D) Adjective

**Q5. Named Entity Recognition (NER) can identify which of the following?**

- A) Person names only
- B) Organization names only
- C) Person, Organization, Location, Date, and more
- D) Only grammatical structures

---

## Word Representations (Questions 6-12)

**Q6. In Bag of Words (BoW), what does the matrix contain?**

- A) Semantic meanings of words
- B) Word frequency counts
- C) Word vectors
- D) POS tags

**Q7. What does TF-IDF stand for?**

- A) Text Frequency - Inverse Data Frequency
- B) Term Frequency - Inverse Document Frequency
- C) Token Frequency - Inverse Dictionary Frequency
- D) Text Feature - Inverse Document Feature

**Q8. If a word appears in almost every document, its TF-IDF score will be:**

- A) Very high
- B) Very low
- C) Zero
- D) Undefined

**Q9. In Word2Vec, what does 'sg=1' parameter mean?**

- A) Use CBOW architecture
- B) Use Skip-gram architecture
- C) Set vocabulary size to 1
- D) Use only 1 training epoch

**Q10. Which statement about Word2Vec is TRUE?**

- A) It creates sparse vectors
- B) It ignores word context completely
- C) It captures semantic relationships between words
- D) It requires TF-IDF preprocessing

**Q11. GloVe is different from Word2Vec because:**

- A) GloVe only uses local context
- B) GloVe uses global co-occurrence statistics
- C) GloVe creates sparse vectors
- D) GloVe doesn't require training

**Q12. Which word representation method creates dense vectors?**

- A) Bag of Words
- B) TF-IDF
- C) Word2Vec
- D) Both A and B

---

## Deep Learning & LSTM (Questions 13-18)

**Q13. What is the purpose of the Embedding layer in a neural network?**

- A) To reduce overfitting
- B) To convert integer-encoded words to dense vectors
- C) To perform classification
- D) To tokenize text

**Q14. Why is padding used in sequence processing?**

- A) To add noise to the data
- B) To make all sequences the same length
- C) To remove stop words
- D) To increase training speed

**Q15. LSTM solves which problem of traditional RNNs?**

- A) Slow training
- B) Vanishing gradient problem
- C) Large memory usage
- D) Inability to process text

**Q16. In an LSTM cell, the 'forget gate' is used to:**

- A) Add new information to memory
- B) Decide what to output
- C) Decide what information to discard from memory
- D) Initialize weights

**Q17. For binary sentiment classification, which activation function is used in the output layer?**

- A) ReLU
- B) Tanh
- C) Sigmoid
- D) Softmax

**Q18. What loss function is used for binary classification?**

- A) Mean Squared Error
- B) Binary Crossentropy
- C) Categorical Crossentropy
- D) Hinge Loss

---

## General NLP (Questions 19-20)

**Q19. Which Python library is primarily used for traditional NLP preprocessing?**

- A) TensorFlow
- B) NLTK
- C) NumPy
- D) Matplotlib

**Q20. What is sentiment analysis?**

- A) Finding grammatical errors in text
- B) Translating text between languages
- C) Determining the emotional tone of text
- D) Summarizing long documents

---

# MCQ Answer Key

| Question | Answer | Explanation                                                                   |
| -------- | ------ | ----------------------------------------------------------------------------- |
| Q1       | **B**  | Tokenization breaks text into smaller units (words, sentences, subwords)      |
| Q2       | **C**  | "computer" is a content word with meaning; others are function words          |
| Q3       | **C**  | Lemmatization uses dictionary lookup, stemming uses rule-based suffix removal |
| Q4       | **C**  | NNP = Proper Noun (names like "Google", "India")                              |
| Q5       | **C**  | NER identifies multiple entity types: PERSON, ORG, GPE, DATE, MONEY, etc.     |
| Q6       | **B**  | BoW matrix contains word frequency counts                                     |
| Q7       | **B**  | Term Frequency - Inverse Document Frequency                                   |
| Q8       | **B**  | High document frequency means low IDF, resulting in low TF-IDF                |
| Q9       | **B**  | sg=1 enables Skip-gram architecture; sg=0 uses CBOW                           |
| Q10      | **C**  | Word2Vec learns semantic relationships (King - Man + Woman = Queen)           |
| Q11      | **B**  | GloVe combines global co-occurrence statistics with local context             |
| Q12      | **C**  | Word2Vec creates dense vectors; BoW and TF-IDF create sparse vectors          |
| Q13      | **B**  | Embedding layer converts word indices to dense vector representations         |
| Q14      | **B**  | Padding ensures all input sequences have the same length for batch processing |
| Q15      | **B**  | LSTM uses gates to control information flow, solving vanishing gradients      |
| Q16      | **C**  | Forget gate decides what information to remove from cell state                |
| Q17      | **C**  | Sigmoid outputs probability between 0 and 1 for binary classification         |
| Q18      | **B**  | Binary crossentropy measures loss for two-class classification                |
| Q19      | **B**  | NLTK (Natural Language Toolkit) is the standard for NLP preprocessing         |
| Q20      | **C**  | Sentiment analysis determines positive, negative, or neutral sentiment        |

---

# Study Tips for Quiz

1. **Understand the NLP Pipeline**: Know the order of preprocessing steps
2. **Compare Methods**: Be able to distinguish BoW vs TF-IDF vs Word2Vec
3. **Know Key Parameters**: vector_size, window, sg in Word2Vec
4. **LSTM Architecture**: Understand the three gates and their purposes
5. **Code Functions**: Remember which library provides which function
6. **Dense vs Sparse**: Word embeddings = Dense, BoW/TF-IDF = Sparse

---

## Quick Formulas

**TF-IDF Calculation:**

```
TF = (term count in document) / (total terms in document)
IDF = log(total documents / documents containing term)
TF-IDF = TF × IDF
```

**LSTM Intuition:**

```
Forget Gate → What to forget
Input Gate → What to remember
Output Gate → What to output
```

---

> **Good Luck with your Quiz!**
>
> Remember: Understanding concepts is more important than memorizing code!
