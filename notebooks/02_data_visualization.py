import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = "data/raw/reddit_comments.csv"
df = pd.read_csv(file_path)

# Convert timestamp
df["created_utc"] = pd.to_datetime(
    df["created_utc"],
    unit="s",
    errors="coerce"
)

print("=" * 50)
print("COMMUNITY LENS - DATA VISUALIZATION")
print("=" * 50)

# 1. Top 10 subreddits
top_subreddits = df["subreddit"].value_counts().head(10)

print("\nTop 10 Subreddits:")
print(top_subreddits)

plt.figure(figsize=(10, 5))
top_subreddits.plot(kind="bar")
plt.title("Top 10 Subreddits by Number of Comments")
plt.xlabel("Subreddit")
plt.ylabel("Number of Comments")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Comments over time
daily_comments = df.groupby(
    df["created_utc"].dt.date
).size()

plt.figure(figsize=(10, 5))
daily_comments.plot()
plt.title("Comments Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Comments")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Replies vs non-replies
reply_counts = [
    df["parent_id"].notna().sum(),
    df["parent_id"].isna().sum()
]

labels = ["Replies", "No Parent"]

plt.figure(figsize=(6, 6))
plt.pie(
    reply_counts,
    labels=labels,
    autopct="%1.1f%%"
)
plt.title("Reply Structure")
plt.show()

print("\n" + "=" * 50)
print("VISUALIZATION COMPLETED")
print("=" * 50)