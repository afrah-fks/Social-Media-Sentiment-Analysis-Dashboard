import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import clean_text

# Load model & vectorizer
model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# -----------------------------
# App Title
# -----------------------------
st.title("📊 Social Media Sentiment Analysis Dashboard")

st.write("Analyze user comments and understand public sentiment in real-time.")

# -----------------------------
# SINGLE INPUT
# -----------------------------
st.subheader("🔹 Analyze Single Comment")

user_input = st.text_area("Enter a comment:")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        clean = clean_text(user_input)
        vec = vectorizer.transform([clean])
        result = model.predict(vec)[0]

        st.success(f"Predicted Sentiment: **{result.upper()}**")
    else:
        st.warning("Please enter some text")

# -----------------------------
# MULTIPLE INPUT
# -----------------------------
st.subheader("🔹 Analyze Multiple Comments")

uploaded_file = st.file_uploader("Upload CSV file with 'text' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'text' not in df.columns:
        st.error("CSV must contain a 'text' column")
    else:
        df['clean_text'] = df['text'].apply(clean_text)
        vec = vectorizer.transform(df['clean_text'])
        df['sentiment'] = model.predict(vec)

        st.write("### 📄 Predictions")
        st.dataframe(df[['text', 'sentiment']])

        # -----------------------------
        # VISUALIZATIONS
        # -----------------------------
        st.write("### 📊 Sentiment Distribution")

        sentiment_counts = df['sentiment'].value_counts()

        # Pie Chart
        fig1, ax1 = plt.subplots()
        ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
        ax1.set_title("Sentiment Distribution (Pie Chart)")
        st.pyplot(fig1)

        # Bar Chart
        fig2, ax2 = plt.subplots()
        ax2.bar(sentiment_counts.index, sentiment_counts.values)
        ax2.set_title("Sentiment Count (Bar Chart)")
        ax2.set_xlabel("Sentiment")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

        # -----------------------------
        # INSIGHTS
        # -----------------------------
        st.write("### 📈 Insights")

        total = len(df)
        pos = sentiment_counts.get('positive', 0)
        neg = sentiment_counts.get('negative', 0)
        neu = sentiment_counts.get('neutral', 0)

        st.write(f"✅ Positive: {pos} ({pos/total*100:.2f}%)")
        st.write(f"❌ Negative: {neg} ({neg/total*100:.2f}%)")
        st.write(f"⚪ Neutral: {neu} ({neu/total*100:.2f}%)")