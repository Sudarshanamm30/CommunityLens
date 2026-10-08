import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# File locations
INPUT_FILE = "data/processed/sentiment_data.csv"
OUTPUT_FILE = "data/processed/reddit_vader_sentiment.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully!")
print("Total records:", len(df))

# Check required column
if "body" not in df.columns:
    print("ERROR: 'body' column not found.")
    exit()

# Create VADER analyzer
analyzer = SentimentIntensityAnalyzer()


# Function to analyze sentiment
def analyze_sentiment(text):

    if pd.isna(text):
        text = ""

    text = str(text)

    scores = analyzer.polarity_scores(text)

    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return pd.Series([
        sentiment,
        scores["pos"],
        scores["neg"],
        scores["neu"],
        compound
    ])


# Apply VADER to every Reddit comment
print("Analyzing sentiment using VADER...")

df[
    [
        "vader_sentiment",
        "vader_positive",
        "vader_negative",
        "vader_neutral",
        "vader_compound"
    ]
] = df["body"].apply(analyze_sentiment)


# Save processed dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\n--------------------------------")
print("VADER analysis completed!")
print("--------------------------------")

print("Total records:", len(df))

print("\nSentiment Distribution:")
print(df["vader_sentiment"].value_counts())

print("\nSaved file:")
print(OUTPUT_FILE)