import streamlit as st
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from transformers import pipeline

# Download stopwords once
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# Cache Vectorizer
@st.cache_resource
def get_vectorizer():
    return CountVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1,
        max_features=500
    )

# Cache NER model
@st.cache_resource
def load_ner_model():
    return pipeline('ner', grouped_entities=True)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Generate N-grams
def generate_ngrams(text, ngram_range=(1, 2), top_n=10):
    vectorizer = get_vectorizer()
    text = clean_text(text)
    X = vectorizer.fit_transform([text])
    ngram_counts = X.toarray().sum(axis=0)
    ngram_features = vectorizer.get_feature_names_out()
    freq_distribution = dict(zip(ngram_features, ngram_counts))
    sorted_ngrams = sorted(freq_distribution.items(), key=lambda x: x[1], reverse=True)
    return pd.DataFrame(sorted_ngrams, columns=['N-gram', 'Frequency']).head(top_n)

# Perform NER
def perform_ner(text):
    ner_model = load_ner_model()
    ner_results = ner_model(text)
    data = [(entity['entity_group'], entity['word']) for entity in ner_results]
    return pd.DataFrame(data, columns=['Entity', 'Text'])

# Streamlit App
st.title("N-gram and NER Analyzer")
sample_text = st.text_area("Enter your text:", value="Samsung Neo QLED 8K TV is an excellent choice in 2024.")
top_n = st.slider("Top N N-grams:", 5, 20, 10)

# Selection box
option = st.radio("Choose an analysis:", ("N-grams", "Named Entity Recognition (NER)", "Both"))

if st.button("Run Analysis"):
    with st.spinner("Processing..."):
        if option == "N-grams" or option == "Both":
            st.subheader("N-gram Results")
            ngram_results = generate_ngrams(sample_text, top_n=top_n)
            st.dataframe(ngram_results)
        
        if option == "Named Entity Recognition (NER)" or option == "Both":
            st.subheader("NER Results")
            ner_results = perform_ner(sample_text)
            st.dataframe(ner_results)
