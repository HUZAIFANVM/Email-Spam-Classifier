import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

# Initialize the PorterStemmer
ps = PorterStemmer()

# Function to preprocess and transform the text
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

# Load the TF-IDF vectorizer and the trained model
tfidf = pickle.load(open('vect.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set the title of the Streamlit app
st.title('Email Spam Classifier')

# Create a text input for the user to enter the email content
email_input = st.text_input('Enter the email')

# Add a button to classify the email
if st.button('Classify Email'):
    # Check if the user has entered an email
    if email_input:
        # Transform the input email using the preprocessing function
        transformed_email = transform_text(email_input)

        # Vectorize the transformed email using the loaded TF-IDF vectorizer
        vect_text = tfidf.transform([transformed_email])

        # Predict the result using the loaded model
        result = model.predict(vect_text)[0]

        # Display the result to the user
        if result == 1:
            st.write('This email is classified as **Spam**.')
        else:
            st.write('This email is classified as **Not Spam**.')
    else:
        st.write('Please enter an email to classify.')

