import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load dataset
df = pd.read_csv("data/100k_youtube_comments.csv")

# Print first few rows to check
print(df.head())
print(df.columns)  # Check column names

# Rename columns if needed (adjust based on printed column names)
# Example below assumes columns are named 'content' and 'sentiment'
# Replace 'content' with actual comment column name if different

df.rename(columns={"content": "comment"}, inplace=True)

# Drop nulls
df.dropna(subset=["comment", "sentiment"], inplace=True)

# Features and labels
X = df["comment"]
y = df["sentiment"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
