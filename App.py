import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure required NLTK resources are available
nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load pre-trained vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('SMS Spam Detector')

input_sms = st.text_area('Enter your message')
if st.button('Predict'):
    transformed_sms = transform_text(input_sms)

    # Vectorize the transformed SMS
    vector_input = tfidf.transform([transformed_sms])

    # Predict using the loaded model
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("The Email is Spam")
    else:
        st.header("The Email is Not Spam")
