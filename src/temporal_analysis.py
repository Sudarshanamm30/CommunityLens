import pandas as pd
import os

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

INPUT_FILE = "data/raw/reddit_comments.csv"
OUTPUT_FILE = "data/processed/temporal_reddit_data.csv"


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("Loading Reddit dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())


# --------------------------------------------------
# 3. Convert timestamp
# --------------------------------------------------

print("\nConverting created_utc to datetime...")

df["created_utc"] = pd.to_datetime(
    df["created_utc"],
    unit="s",
    errors="coerce"
)

# Remove rows where timestamp is missing
df = df.dropna(subset=["created_utc"]).copy()


# --------------------------------------------------
# 4. Create temporal features
# --------------------------------------------------

df["date"] = df["created_utc"].dt.date
df["hour"] = df["created_utc"].dt.hour
df["day_of_week"] = df["created_utc"].dt.day_name()
df["week"] = df["created_utc"].dt.isocalendar().week.astype(int)


# --------------------------------------------------
# 5. Clean community and text
# --------------------------------------------------

df["subreddit"] = df["subreddit"].fillna("Unknown").astype(str)

df["body"] = df["body"].fillna("").astype(str)

# Remove empty comments
df = df[df["body"].str.strip() != ""].copy()


# --------------------------------------------------
# 6. Create daily activity summary
# --------------------------------------------------

daily_activity = (
    df.groupby("date")
      .agg(
          total_comments=("id", "count"),
          active_communities=("subreddit", "nunique"),
          active_posts=("submission_id", "nunique")
      )
      .reset_index()
)


# --------------------------------------------------
# 7. Create community-wise temporal summary
# --------------------------------------------------

community_daily = (
    df.groupby(["date", "subreddit"])
      .agg(
          comment_count=("id", "count"),
          post_count=("submission_id", "nunique")
      )
      .reset_index()
)


# --------------------------------------------------
# 8. Create output directory
# --------------------------------------------------

os.makedirs("data/processed", exist_ok=True)


# --------------------------------------------------
# 9. Save processed temporal dataset
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

daily_activity.to_csv(
    "data/processed/daily_activity.csv",
    index=False
)

community_daily.to_csv(
    "data/processed/community_daily_activity.csv",
    index=False
)


# --------------------------------------------------
# 10. Display results
# --------------------------------------------------

print("\n----------------------------------------")
print("TEMPORAL ANALYSIS COMPLETED")
print("----------------------------------------")

print("\nTemporal dataset shape:")
print(df.shape)

print("\nDaily activity:")
print(daily_activity.head())

print("\nCommunity daily activity:")
print(community_daily.head())

print("\nFiles created:")
print("1. data/processed/temporal_reddit_data.csv")
print("2. data/processed/daily_activity.csv")
print("3. data/processed/community_daily_activity.csv")

print("\nPhase 1 completed successfully!")