import pandas as pd

# Load dataset
file_path = "data/raw/reddit_comments.csv"

df = pd.read_csv(file_path)

print("=" * 50)
print("COMMUNITY LENS - DATA EXPLORATION")
print("=" * 50)

# -----------------------------
# 1. Dataset size
# -----------------------------

print("\n1. Dataset Shape")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# -----------------------------
# 2. Column names
# -----------------------------

print("\n2. Columns")
print(df.columns.tolist())

# -----------------------------
# 3. Number of subreddits
# -----------------------------

print("\n3. Number of Subreddits")
print(df["subreddit"].nunique())

# -----------------------------
# 4. Comments per subreddit
# -----------------------------

print("\n4. Top Subreddits")

print(
    df["subreddit"]
    .value_counts()
    .head(10)
)

# -----------------------------
# 5. Missing values
# -----------------------------

print("\n5. Missing Values")

print(df.isnull().sum())

# -----------------------------
# 6. Empty comments
# -----------------------------

print("\n6. Empty Comments")

empty_comments = df["body"].isna().sum()

print("Empty comments:", empty_comments)

# -----------------------------
# 7. Convert timestamp
# -----------------------------

df["created_utc"] = pd.to_datetime(
    df["created_utc"],
    unit="s",
    errors="coerce"
)

print("\n7. Date Range")

print("First date:", df["created_utc"].min())
print("Last date:", df["created_utc"].max())

# -----------------------------
# 8. Reply analysis
# -----------------------------

print("\n8. Interaction Information")

replies = df["parent_id"].notna().sum()

print("Comments with parent:", replies)

print(
    "Comments without parent:",
    df["parent_id"].isna().sum()
)

# -----------------------------
# 9. Sample comments
# -----------------------------

print("\n9. Sample Comments")

print(
    df[
        [
            "subreddit",
            "body",
            "created_utc",
            "parent_id"
        ]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 50)
print("EXPLORATION COMPLETED")
print("=" * 50)