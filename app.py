import streamlit as st
from transformers import pipeline

# Use smaller models to speed up loading
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Summarization function
def summarize_text(text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Sentiment analysis function
def analyze_sentiment(text):
    sentiment = sentiment_analyzer(text)
    return sentiment[0]['label']

# Streamlit app interface
st.title("Text Summarizer & Sentiment Analyzer")

# Input text from the user
text_input = st.text_area("Enter the text you want to analyze:")

if text_input.strip():
    if st.button("Analyze"):
        # Display summary
        summary = summarize_text(text_input)
        st.subheader("Summary:")
        st.write(summary)

        # Display sentiment
        sentiment = analyze_sentiment(text_input)
        st.subheader("Sentiment Analysis:")
        st.write(f"Sentiment: {sentiment}")
else:
    st.write("Please enter some text to analyze.")
