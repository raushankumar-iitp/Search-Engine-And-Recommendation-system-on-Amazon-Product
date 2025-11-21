# app.py (fixed for Streamlit Cloud)
import os
import string

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# NLP
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- NLTK: ensure required data is present ----------------
def ensure_nltk_downloads():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

ensure_nltk_downloads()

# ---------------- Helpers: preprocessing ----------------
stemmer = SnowballStemmer("english")
try:
    stop_words = set(stopwords.words("english"))
except Exception:
    stop_words = set(["the", "and", "is", "in", "to", "for", "with", "a", "an", "of"])

def tokenize_and_stem_tokens(text):
    """Return list of stemmed tokens (only alphabetic, not stopwords)."""
    if text is None:
        text = ""
    text = str(text).lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # simple tokenization (word_tokenize is optional; using split after cleaning is safe)
    try:
        # prefer nltk tokenizer (works after punkt download)
        tokens = nltk.word_tokenize(text)
    except Exception:
        tokens = text.split()
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

def preprocess_to_string(text):
    """Preprocess text and return a single string of stems (space separated)."""
    return " ".join(tokenize_and_stem_tokens(text))

# ---------------- Load data ----------------
DATA_FILE = "amazon_product.csv"  # ensure this file is in repo root (or use sample file)
if not os.path.exists(DATA_FILE):
    st.error(f"Dataset '{DATA_FILE}' not found. Upload it to the repo root or set DATA_FILE path.")
    st.stop()

data = pd.read_csv(DATA_FILE)

# safe drop if id exists
if "id" in data.columns:
    data = data.drop(columns=["id"])

# ensure Title/Description columns exist
if "Title" not in data.columns:
    data["Title"] = ""
if "Description" not in data.columns:
    data["Description"] = ""

data["Title"] = data["Title"].fillna("").astype(str)
data["Description"] = data["Description"].fillna("").astype(str)

# ---------------- Preprocess corpus (fast) ----------------
# Build combined text column
combined_texts = (data["Title"] + " " + data["Description"]).tolist()
# Preprocess with list comprehension (faster than df.apply)
processed_texts = [preprocess_to_string(t) for t in combined_texts]
data["processed_text"] = processed_texts

# ---------------- TF-IDF: fit once and cache ----------------
@st.cache_resource
def build_tfidf_and_matrix(texts):
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(texts)
    return tfidf, matrix

tfidf_vectorizer, tfidf_matrix = build_tfidf_and_matrix(data["processed_text"].tolist())

# ---------------- Search function ----------------
def search_products(query, top_n=10):
    if not query or str(query).strip() == "":
        return pd.DataFrame(columns=["Title", "Description", "Category", "score"])

    q_processed = preprocess_to_string(query)
    q_vec = tfidf_vectorizer.transform([q_processed])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    data["score"] = sims
    results = data.sort_values(by="score", ascending=False).head(top_n)
    # return needed columns + score
    cols = [c for c in ["Title", "Description", "Category", "score"] if c in results.columns]
    return results[cols]

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Amazon Search & Recommendation", layout="wide")

# Image safely
try:
    img = Image.open("img.png")
    st.image(img, width=600)
except Exception:
    st.write("**(Logo image not found — place `img.png` in repo root to show app image.)**")

st.title("Search Engine and Product Recommendation System")

query = st.text_input("Enter product name or keywords")
if st.button("Search"):
    df_res = search_products(query, top_n=10)
    if df_res.empty:
        st.info("No results found or query was empty.")
    else:
        # show nicer result table with score rounded
        df_show = df_res.copy()
        if "score" in df_show.columns:
            df_show["score"] = df_show["score"].round(4)
        st.dataframe(df_show.reset_index(drop=True))
