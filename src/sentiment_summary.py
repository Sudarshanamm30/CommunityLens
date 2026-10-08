import pandas as pd

# Load the processed sentiment data
df = pd.read_csv("data/processed/sentiment_data.csv")

# Count each sentiment
sentiment_counts = df["sentiment"].value_counts()

print("\n===== SENTIMENT SUMMARY =====")
print(sentiment_counts)

print("\n===== SENTIMENT PERCENTAGE =====")
print((sentiment_counts / len(df) * 100).round(2))