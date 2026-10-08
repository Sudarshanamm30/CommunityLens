import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load cleaned dataset
file_path = "data/raw/reddit_comments.csv"
df = pd.read_csv(file_path)

# Remove empty comments
df = df.dropna(subset=["body"])

# Create sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Calculate sentiment score
def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

# Apply sentiment analysis
df["sentiment"] = df["body"].apply(get_sentiment)

# Save results
output_file = "data/processed/sentiment_data.csv"
df.to_csv(output_file, index=False)

print("=" * 50)
print("COMMUNITY LENS - SENTIMENT ANALYSIS")
print("=" * 50)

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

print("\nSample Results:")
print(df[["body", "sentiment"]].head(10).to_string(index=False))

print("\nSaved to:")
print(output_file)

print("\n" + "=" * 50)
print("SENTIMENT ANALYSIS COMPLETED")
print("=" * 50)