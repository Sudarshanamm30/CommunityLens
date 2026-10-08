import pandas as pd

# -----------------------------
# 1. Load the raw dataset
# -----------------------------

file_path = "../data/raw/reddit_comments.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")

# -----------------------------
# 2. Display basic information
# -----------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

# -----------------------------
# 3. Remove duplicate comments
# -----------------------------

df = df.drop_duplicates(subset=["id"])

# -----------------------------
# 4. Remove comments with no text
# -----------------------------

df = df.dropna(subset=["body"])

# -----------------------------
# 5. Convert timestamp
# -----------------------------

df["created_utc"] = pd.to_datetime(
    df["created_utc"],
    unit="s",
    errors="coerce"
)

# Remove invalid timestamps
df = df.dropna(subset=["created_utc"])

# -----------------------------
# 6. Select useful columns
# -----------------------------

df = df[
    [
        "subreddit",
        "id",
        "submission_id",
        "body",
        "created_utc",
        "parent_id"
    ]
]

# -----------------------------
# 7. Save cleaned dataset
# -----------------------------

output_path = "../data/processed/cleaned_data.csv"

df.to_csv(output_path, index=False)

print("\nCleaning completed!")
print("Cleaned dataset shape:", df.shape)
print("Saved to:", output_path)