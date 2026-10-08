import pandas as pd
import os

# ==================================================
# 1. FILE PATHS
# ==================================================

INPUT_FILE = "data/processed/reddit_vader_sentiment.csv"

OUTPUT_DAILY = "data/processed/daily_sentiment.csv"

OUTPUT_COMMUNITY = "data/processed/community_daily_sentiment.csv"


# ==================================================
# 2. LOAD DATASET
# ==================================================

print("Loading VADER sentiment dataset...")

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully!")

print("Rows:", len(df))

print("Columns:")
print(df.columns.tolist())


# ==================================================
# 3. CHECK REQUIRED COLUMNS
# ==================================================

required_columns = [
    "subreddit",
    "body",
    "created_utc",
    "vader_compound",
    "vader_positive",
    "vader_negative",
    "vader_neutral"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    print("\nERROR!")

    print("Missing columns:")
    print(missing_columns)

    print("\nPlease check reddit_vader_sentiment.csv")

    exit()


# ==================================================
# 4. CONVERT TIMESTAMP
# ==================================================

print("\nConverting timestamp...")

df["created_utc"] = pd.to_datetime(
    df["created_utc"],
    unit="s",
    errors="coerce",
    utc=True
)


# Remove invalid timestamps

df = df.dropna(
    subset=["created_utc"]
).copy()


# ==================================================
# 5. CREATE DATE
# ==================================================

df["date"] = df["created_utc"].dt.date


# ==================================================
# 6. CLEAN COMMUNITY
# ==================================================

df["subreddit"] = (
    df["subreddit"]
    .fillna("Unknown")
    .astype(str)
)


# ==================================================
# 7. CLEAN SENTIMENT VALUES
# ==================================================

sentiment_columns = [
    "vader_compound",
    "vader_positive",
    "vader_negative",
    "vader_neutral"
]

for column in sentiment_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna(
    subset=["vader_compound"]
).copy()


# ==================================================
# 8. DAILY SENTIMENT ANALYSIS
# ==================================================

print("\nCreating daily sentiment summary...")


daily_sentiment = (
    df.groupby("date")
      .agg(
          total_comments=("body", "count"),

          average_compound=("vader_compound", "mean"),

          average_positive=("vader_positive", "mean"),

          average_negative=("vader_negative", "mean"),

          average_neutral=("vader_neutral", "mean"),

          active_communities=("subreddit", "nunique")
      )
      .reset_index()
)


# ==================================================
# 9. CREATE SENTIMENT CATEGORY
# ==================================================

def classify_sentiment(score):

    if score >= 0.05:
        return "Positive"

    elif score <= -0.05:
        return "Negative"

    else:
        return "Neutral"


daily_sentiment["overall_sentiment"] = (
    daily_sentiment["average_compound"]
    .apply(classify_sentiment)
)


# ==================================================
# 10. COMMUNITY-WISE DAILY SENTIMENT
# ==================================================

print("\nCreating community-wise sentiment summary...")


community_daily_sentiment = (
    df.groupby(
        ["date", "subreddit"]
    )
    .agg(

        comment_count=("body", "count"),

        average_compound=("vader_compound", "mean"),

        average_positive=("vader_positive", "mean"),

        average_negative=("vader_negative", "mean"),

        average_neutral=("vader_neutral", "mean")

    )
    .reset_index()
)


# ==================================================
# 11. COMMUNITY SENTIMENT CATEGORY
# ==================================================

community_daily_sentiment["sentiment"] = (
    community_daily_sentiment[
        "average_compound"
    ]
    .apply(classify_sentiment)
)


# ==================================================
# 12. CREATE OUTPUT DIRECTORY
# ==================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# ==================================================
# 13. SAVE DAILY SENTIMENT
# ==================================================

daily_sentiment.to_csv(
    OUTPUT_DAILY,
    index=False
)


# ==================================================
# 14. SAVE COMMUNITY SENTIMENT
# ==================================================

community_daily_sentiment.to_csv(
    OUTPUT_COMMUNITY,
    index=False
)


# ==================================================
# 15. DISPLAY RESULTS
# ==================================================

print("\n")
print("=" * 60)

print("PHASE 2 COMPLETED SUCCESSFULLY")

print("=" * 60)


print("\nDaily Sentiment:")
print(
    daily_sentiment.head(10)
)


print("\nCommunity Daily Sentiment:")
print(
    community_daily_sentiment.head(10)
)


print("\nFiles created:")

print(
    "1. data/processed/daily_sentiment.csv"
)

print(
    "2. data/processed/community_daily_sentiment.csv"
)


print("\nDate range:")

print(
    df["created_utc"].min()
)

print(
    "to"
)

print(
    df["created_utc"].max()
)


print("\nNumber of days:")

print(
    df["date"].nunique()
)


print("\nNumber of communities:")

print(
    df["subreddit"].nunique()
)


print("\nPhase 2 completed!")