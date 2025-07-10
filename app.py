import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords once when app starts
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load('fraud_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

stemmer = PorterStemmer()

# Define clean_text inside a function after stopwords are downloaded
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

st.title("Fake Job Posting Detector")

user_input = st.text_area("Enter Job Description")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a job description.")
    else:
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0][prediction]

        if prediction == 1:
            st.error(f"This job posting is likely **FAKE** with probability {proba:.2f}")
        else:
            st.success(f"This job posting is likely **REAL** with probability {proba:.2f}")
