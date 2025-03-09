import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import re
import string

# Set of stopwords
stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
"you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 
'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', 
"didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', 
"isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

# Define a text preprocessing function without using NLTK
def preprocess_text_re(text):
    """
    Preprocesses the input text by removing non-alphanumeric characters,
    normalizing spaces, converting to lowercase, and removing stopwords.
    """
    # Remove non-alphanumeric characters (retain spaces)
    text = re.sub(r"[^\w\s]", " ", text)
    # Normalize spaces (convert multiple spaces to one)
    text = re.sub(r"\s+", " ", text)
    # Convert to lowercase and strip leading/trailing spaces
    text = text.strip().lower()
    
    # Tokenize the text by splitting on whitespace
    words = text.split()
    
    # Remove stopwords from the tokenized words
    filtered_words = [word for word in words if word not in stop_words]
    
    return " ".join(filtered_words)

# Function to check for unsafe links (http instead of https)
def check_http_links(text):
    """
    Detects URLs in the input text and flags those that use HTTP instead of HTTPS.
    """
    # Regular expression to detect URLs
    url_pattern = r"http[s]?://(?:[a-zA-Z0-9$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    urls = re.findall(url_pattern, text)
    # Filter for URLs starting with "http://"
    unsafe_links = [url for url in urls if url.startswith("http://")]
    return unsafe_links

# Load the pre-trained pipeline using joblib
try:
    pipeline = joblib.load('text_classifier_pipeline.pkl')
except Exception as e:
    st.error(f"Error loading pipeline: {e}")
    st.stop()

# Streamlit App UI
st.title("Email Spam Detector")
st.markdown("""
Welcome! Input text to detect spam.
This app will also flag unsafe links.
""")

# Input text area
user_input = st.text_area("Enter the text for prediction.", "")

if st.button("Predict Text"):
    if user_input.strip():
        # Check for unsafe links
        unsafe_links = check_http_links(user_input)
        if unsafe_links:
            st.warning(f"The message contains unsafe links (HTTP): {', '.join(unsafe_links)}")
            st.write("### Prediction: spam")
            st.write("### This message is 100% spam due to unsafe links.")
        else:
            # Preprocess the input text
            preprocessed_input = preprocess_text_re(user_input)
            
            # Predict using the loaded pipeline
            prediction = pipeline.predict([preprocessed_input])[0]
            probabilities = pipeline.predict_proba([preprocessed_input])[0]
            
            prediction_label = "spam" if prediction == 1 else "legit"
            spam_percentage = round(probabilities[1] * 100, 2) if prediction == 1 else 0
            
            # Display results
            st.markdown(f"### Prediction: {prediction_label}")
            st.write("### Prediction Probabilities:")
            prob_df = pd.DataFrame({'Class': ['legit', 'spam'], 'Probability': probabilities})
            st.table(prob_df)
            
            # Display a bar chart for probability distribution
            st.write("### Probability Stats:")
            fig, ax = plt.subplots()

            # Define categories and corresponding probabilities
            categories = ['legit', 'spam']
            # Create a bar chart with the probabilities
            bars = ax.bar(categories, probabilities, color=['#ff9999','#66b3ff'])

            # Set the y-axis limit from 0 to 1
            ax.set_ylim(0, 1)

            # Add text labels on top of each bar to show percentage
            for bar, prob in zip(bars, probabilities):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{prob*100:.1f}%", 
                        ha='center', fontweight='bold')

            st.pyplot(fig)

    else:
        st.error("Please enter a message!")
