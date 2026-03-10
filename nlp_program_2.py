# =====================================================
# WORD REPRESENTATION TECHNIQUES
# BoW, TF-IDF, Word2Vec, GloVe
# =====================================================


# ---------------- 1. Corpus ----------------
corpus = [
    "Cyber security prevents attacks",
    "Malware attacks computer systems",
    "Security protects data"
]

print("CORPUS:")
for doc in corpus:
    print("-", doc)


# ---------------- 2. Bag of Words (BoW) ----------------
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)

print("\n--- BAG OF WORDS ---")
print("Vocabulary:", bow_vectorizer.get_feature_names_out())
print("BoW Matrix:\n", bow.toarray())


# ---------------- 3. TF-IDF ----------------
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)

print("\n--- TF-IDF ---")
print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", tfidf.toarray())


# ---------------- 4. Word2Vec ----------------
# If gensim is not installed, uncomment below line
# NOTE: pip install gensim

from gensim.models import Word2Vec

tokenized_corpus = [sentence.lower().split() for sentence in corpus]

w2v_model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=50,
    window=3,
    min_count=1,
    sg=1   # Skip-gram
)

print("\n--- WORD2VEC ---")
print("Vector for 'security':\n", w2v_model.wv["security"])
print("Similar words to 'security':\n", w2v_model.wv.most_similar("security"))


# ---------------- 5. GloVe ----------------
# NOTE: Internet is required (best run in Google Colab)

import gensim.downloader as api

glove_model = api.load("glove-wiki-gigaword-50")

print("\n--- GLOVE ---")
print("Vector for 'security':\n", glove_model["security"])
print("Similar words to 'security':\n", glove_model.most_similar("security"))
