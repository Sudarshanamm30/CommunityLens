import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/processed/sentiment_data.csv")

# Count sentiments
counts = df["sentiment"].value_counts()

# Create bar chart
plt.figure(figsize=(7, 5))
counts.plot(kind="bar")

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Texts")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("data/processed/sentiment_distribution.png")
plt.show()