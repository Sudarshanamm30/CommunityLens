import pandas as pd
import numpy as np
import os


# ==================================================
# 1. FILE PATHS
# ==================================================

DAILY_SENTIMENT_FILE = (
    "data/processed/daily_sentiment.csv"
)

COMMUNITY_SENTIMENT_FILE = (
    "data/processed/community_daily_sentiment.csv"
)

EDGES_FILE = (
    "data/processed/interaction_edges.csv"
)

OUTPUT_FILE = (
    "data/processed/change_detection.csv"
)


# ==================================================
# 2. LOAD DAILY SENTIMENT
# ==================================================

print("Loading daily sentiment data...")

daily = pd.read_csv(
    DAILY_SENTIMENT_FILE
)

print(
    "Daily sentiment rows:",
    len(daily)
)


# ==================================================
# 3. LOAD COMMUNITY SENTIMENT
# ==================================================

print("\nLoading community sentiment data...")

community_daily = pd.read_csv(
    COMMUNITY_SENTIMENT_FILE
)

print(
    "Community sentiment rows:",
    len(community_daily)
)


# ==================================================
# 4. LOAD INTERACTION EDGES
# ==================================================

print("\nLoading interaction edges...")

edges = pd.read_csv(
    EDGES_FILE,
    low_memory=False
)

print(
    "Interaction edges:",
    len(edges)
)


# ==================================================
# 5. CONVERT DATES
# ==================================================

daily["date"] = pd.to_datetime(
    daily["date"]
)

community_daily["date"] = pd.to_datetime(
    community_daily["date"]
)


# ==================================================
# 6. PREPARE DAILY INTERACTION COUNT
# ==================================================

if len(edges) > 0:

    edges["created_utc"] = pd.to_datetime(
        edges["created_utc"],
        utc=True,
        errors="coerce"
    )

    edges["date"] = (
        edges["created_utc"]
        .dt.date
    )

    edges["date"] = pd.to_datetime(
        edges["date"]
    )

    daily_interactions = (
        edges
        .groupby("date")
        .size()
        .reset_index(
            name="interaction_count"
        )
    )

else:

    daily_interactions = pd.DataFrame(
        columns=[
            "date",
            "interaction_count"
        ]
    )


# ==================================================
# 7. MERGE DAILY INFORMATION
# ==================================================

print("\nCombining daily activity, sentiment and interaction data...")

change_df = daily[
    [
        "date",
        "total_comments",
        "average_compound",
        "active_communities"
    ]
].copy()


change_df = change_df.merge(
    daily_interactions,
    on="date",
    how="left"
)


change_df["interaction_count"] = (
    change_df["interaction_count"]
    .fillna(0)
)


# ==================================================
# 8. CALCULATE DAILY CHANGES
# ==================================================

print("\nCalculating changes...")


# Activity change

change_df["activity_change"] = (
    change_df["total_comments"]
    .pct_change()
)


# Sentiment change

change_df["sentiment_change"] = (
    change_df["average_compound"]
    .diff()
)


# Interaction change

change_df["interaction_change"] = (
    change_df["interaction_count"]
    .pct_change()
)


# Replace infinite values

change_df = change_df.replace(
    [np.inf, -np.inf],
    np.nan
)


# First day's change cannot be calculated

change_df[
    [
        "activity_change",
        "sentiment_change",
        "interaction_change"
    ]
] = change_df[
    [
        "activity_change",
        "sentiment_change",
        "interaction_change"
    ]
].fillna(0)


# ==================================================
# 9. CALCULATE ROLLING BASELINES
# ==================================================

print("\nCalculating rolling baselines...")


# 3-day rolling mean

change_df["activity_baseline"] = (
    change_df["total_comments"]
    .rolling(
        window=3,
        min_periods=2
    )
    .mean()
)


change_df["sentiment_baseline"] = (
    change_df["average_compound"]
    .rolling(
        window=3,
        min_periods=2
    )
    .mean()
)


change_df["interaction_baseline"] = (
    change_df["interaction_count"]
    .rolling(
        window=3,
        min_periods=2
    )
    .mean()
)


# ==================================================
# 10. CALCULATE RELATIVE DEVIATION
# ==================================================

change_df["activity_deviation"] = (
    (
        change_df["total_comments"]
        -
        change_df["activity_baseline"]
    )
    /
    change_df["activity_baseline"]
)


change_df["interaction_deviation"] = (
    (
        change_df["interaction_count"]
        -
        change_df["interaction_baseline"]
    )
    /
    (
        change_df["interaction_baseline"]
        .replace(0, np.nan)
    )
)


change_df["sentiment_deviation"] = (
    (
        change_df["average_compound"]
        -
        change_df["sentiment_baseline"]
    )
    .abs()
)


# Replace invalid values

change_df = change_df.replace(
    [np.inf, -np.inf],
    np.nan
)

change_df = change_df.fillna(0)


# ==================================================
# 11. NORMALIZE CHANGE SIGNALS
# ==================================================

print("\nCreating change signals...")


# Activity signal

change_df["activity_signal"] = (
    change_df["activity_deviation"]
    .abs()
)


# Interaction signal

change_df["interaction_signal"] = (
    change_df["interaction_deviation"]
    .abs()
)


# Sentiment signal

change_df["sentiment_signal"] = (
    change_df["sentiment_deviation"]
)


# ==================================================
# 12. CALCULATE CHANGE SCORE
# ==================================================

change_df["change_score"] = (

    0.40
    *
    change_df["activity_signal"]

    +

    0.30
    *
    change_df["sentiment_signal"]

    +

    0.30
    *
    change_df["interaction_signal"]

)


# ==================================================
# 13. CALCULATE DATA-DRIVEN THRESHOLD
# ==================================================

threshold = (
    change_df["change_score"]
    .mean()
    +
    change_df["change_score"]
    .std()
)


# Prevent threshold from becoming zero

if threshold <= 0:

    threshold = 0.10


print(
    "\nChange detection threshold:",
    round(threshold, 4)
)


# ==================================================
# 14. DETECT SIGNIFICANT CHANGES
# ==================================================

change_df["significant_change"] = (
    change_df["change_score"]
    >= threshold
)


# ==================================================
# 15. CLASSIFY CHANGE TYPE
# ==================================================

def classify_change(row):

    activity = row["activity_signal"]

    sentiment = row["sentiment_signal"]

    interaction = row["interaction_signal"]


    signals = {

        "Activity": activity,

        "Sentiment": sentiment,

        "Interaction": interaction

    }


    strongest = max(
        signals,
        key=signals.get
    )


    # If no meaningful change

    if row["significant_change"] == False:

        return "Normal"


    return strongest


change_df["change_type"] = (
    change_df.apply(
        classify_change,
        axis=1
    )
)


# ==================================================
# 16. CHANGE DIRECTION
# ==================================================

def determine_direction(row):

    if not row["significant_change"]:

        return "Stable"


    if row["change_type"] == "Activity":

        if row["activity_change"] > 0:

            return "Increase"

        else:

            return "Decrease"


    if row["change_type"] == "Sentiment":

        if row["sentiment_change"] > 0:

            return "More Positive"

        else:

            return "More Negative"


    if row["change_type"] == "Interaction":

        if row["interaction_change"] > 0:

            return "Increase"

        else:

            return "Decrease"


    return "Stable"


change_df["change_direction"] = (
    change_df.apply(
        determine_direction,
        axis=1
    )
)


# ==================================================
# 17. SAVE RESULTS
# ==================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


change_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# 18. DISPLAY RESULTS
# ==================================================

print("\n")
print("=" * 60)

print("PHASE 4 — CHANGE DETECTION COMPLETED")

print("=" * 60)


print("\nTotal days analyzed:")

print(
    len(change_df)
)


print("\nSignificant changes detected:")

print(
    change_df[
        "significant_change"
    ].sum()
)


print("\nDetected change events:")

events = change_df[
    change_df["significant_change"]
]


print(
    events[
        [
            "date",
            "total_comments",
            "average_compound",
            "interaction_count",
            "change_score",
            "change_type",
            "change_direction"
        ]
    ]
)


print("\nChange type summary:")

print(
    events[
        "change_type"
    ].value_counts()
)


print("\nOutput file created:")

print(
    OUTPUT_FILE
)


print("\nPhase 4 completed successfully!")