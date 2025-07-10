# app.py

import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Load saved model and vectorizer
model = pickle.load(open("models/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


# Function to fetch YouTube comments
def get_comments(youtube, **kwargs):
    comments = []
    results = youtube.commentThreads().list(**kwargs).execute()

    while results:
        for item in results["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        # check if nextPageToken exists
        if "nextPageToken" in results:
            kwargs["pageToken"] = results["nextPageToken"]
            results = youtube.commentThreads().list(**kwargs).execute()
        else:
            break
    return comments


# Streamlit UI
st.title("🎬 YouTube Comments Sentiment Analysis")

# Input field for YouTube URL
video_url = st.text_input("Enter YouTube Video URL:")

if video_url:
    # Extract video ID from URL
    if "v=" in video_url:
        video_id = video_url.split("v=")[1]
        ampersand_position = video_id.find("&")
        if ampersand_position != -1:
            video_id = video_id[:ampersand_position]
    else:
        st.error("Invalid YouTube URL format.")
        st.stop()

    # Initialize YouTube API client
    api_key = "YOUR_API_KEY"  # 🔴 Replace with your actual API key
    youtube = build("youtube", "v3", developerKey=api_key)

    # Fetch comments
    st.info("Fetching comments...")
    comments = get_comments(
        youtube,
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=100,
    )  # fetches up to 100 comments

    # Show fetched comments
    st.write(f"✅ Fetched {len(comments)} comments.")

    # Create DataFrame
    df = pd.DataFrame(comments, columns=["Comment"])

    # Preprocess and predict
    comments_vec = vectorizer.transform(df["Comment"])
    predictions = model.predict(comments_vec)
    df["Sentiment"] = predictions

    # Display DataFrame
    st.write(df)

    # Plot pie chart
    st.subheader("Sentiment Distribution")
    sentiment_counts = df["Sentiment"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%")
    st.pyplot(fig)
