import streamlit as st
from transformers import pipeline

# Step 3: Initialize the Hugging Face pipelines with specified models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Explicit model for summarization
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")  # Explicit model for sentiment analysis

# Step 4: Define the summarization function
def summarize_text_huggingface(text):
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Step 5: Define the sentiment analysis function
def analyze_sentiment_huggingface(text):
    sentiment = sentiment_analyzer(text)
    return sentiment[0]['label']  # 'POSITIVE' or 'NEGATIVE'

# Streamlit App Interface
st.title("Text Summarizer & Sentiment Analyzer")

# Text input from user
text_input = st.text_area("Enter the text you want to analyze:")

if text_input.strip():
    if st.button("Analyze"):
        st.write("**Processing...**")

        # Get summary
        summary = summarize_text_huggingface(text_input)
        st.write("### Summary:")
        st.write(summary)

        # Get sentiment
        sentiment = analyze_sentiment_huggingface(text_input)
        st.write("### Sentiment Analysis:")
        st.write(f"Sentiment: {sentiment}")
else:
    st.write("Please enter some text to analyze.")
