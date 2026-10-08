import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PHASE 9 — DOWNSTREAM SENTIMENT ANOMALY DETECTION
# ============================================================

print("=" * 60)
print("PHASE 9 — DOWNSTREAM SENTIMENT ANOMALY DETECTION")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SENTIMENT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "community_daily_sentiment.csv"
)

CHANGE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "change_detection.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "sentiment_anomalies.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

sentiment_df = pd.read_csv(
    SENTIMENT_FILE
)

change_df = pd.read_csv(
    CHANGE_FILE
)


print("\nSentiment data loaded.")

print(
    "Rows:",
    len(sentiment_df)
)

print(
    "Columns:",
    sentiment_df.columns.tolist()
)


# ============================================================
# 3. PREPARE DATE COLUMNS
# ============================================================

sentiment_df["date"] = pd.to_datetime(
    sentiment_df["date"]
).dt.date

change_df["date"] = pd.to_datetime(
    change_df["date"]
).dt.date


# ============================================================
# 4. DISPLAY SENTIMENT COLUMNS
# ============================================================

print("\nSentiment columns:")

print(
    sentiment_df.columns.tolist()
)


# ============================================================
# 5. SELECT VADER SENTIMENT SCORE
# ============================================================

# Your actual dataset contains:
#
# average_compound
#
# Therefore this is the primary column used.

if "average_compound" in sentiment_df.columns:

    sentiment_column = "average_compound"

elif "avg_compound" in sentiment_df.columns:

    sentiment_column = "avg_compound"

elif "vader_compound" in sentiment_df.columns:

    sentiment_column = "vader_compound"

elif "sentiment_score" in sentiment_df.columns:

    sentiment_column = "sentiment_score"

else:

    raise ValueError(
        "No suitable sentiment score column found."
    )


print(
    "\nUsing sentiment column:",
    sentiment_column
)


# ============================================================
# 6. CHECK FOR REQUIRED COMMUNITY COLUMN
# ============================================================

if "subreddit" not in sentiment_df.columns:

    raise ValueError(
        "The sentiment file does not contain "
        "'subreddit' column."
    )


# ============================================================
# 7. SENTIMENT ANOMALY DETECTION
# ============================================================

results = []


communities = sorted(
    sentiment_df[
        "subreddit"
    ]
    .dropna()
    .unique()
)


print(
    "\nCommunities found:",
    len(communities)
)


for community in communities:

    print(
        "Processing:",
        community
    )


    # --------------------------------------------------------
    # Select one community
    # --------------------------------------------------------

    community_df = sentiment_df[
        sentiment_df["subreddit"] == community
    ].copy()


    # Sort chronologically

    community_df = community_df.sort_values(
        "date"
    )


    # --------------------------------------------------------
    # Convert sentiment to numeric
    # --------------------------------------------------------

    community_df[
        sentiment_column
    ] = pd.to_numeric(
        community_df[
            sentiment_column
        ],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Calculate rolling baseline
    #
    # Previous 3 days are used as the baseline.
    # shift(1) ensures the current day is not included
    # in its own baseline.
    # --------------------------------------------------------

    rolling_mean = (
        community_df[
            sentiment_column
        ]
        .rolling(
            window=3,
            min_periods=3
        )
        .mean()
        .shift(1)
    )


    rolling_std = (
        community_df[
            sentiment_column
        ]
        .rolling(
            window=3,
            min_periods=3
        )
        .std()
        .shift(1)
    )


    # --------------------------------------------------------
    # Avoid division by zero
    # --------------------------------------------------------

    rolling_std = rolling_std.replace(
        0,
        np.nan
    )


    # --------------------------------------------------------
    # Calculate sentiment Z-score
    # --------------------------------------------------------

    community_df[
        "sentiment_zscore"
    ] = (

        (
            community_df[
                sentiment_column
            ]
            - rolling_mean
        )
        / rolling_std

    )


    # --------------------------------------------------------
    # Identify anomalies
    #
    # |Z| >= 2 is considered an anomaly.
    # --------------------------------------------------------

    community_df[
        "sentiment_anomaly"
    ] = (

        community_df[
            "sentiment_zscore"
        ]
        .abs()
        >= 2

    )


    # --------------------------------------------------------
    # Identify anomaly direction
    # --------------------------------------------------------

    community_df[
        "anomaly_direction"
    ] = np.where(

        community_df[
            "sentiment_zscore"
        ] >= 2,

        "Positive Anomaly",

        np.where(

            community_df[
                "sentiment_zscore"
            ] <= -2,

            "Negative Anomaly",

            "Normal"

        )

    )


    # --------------------------------------------------------
    # Add community result
    # --------------------------------------------------------

    results.append(
        community_df
    )


# ============================================================
# 8. COMBINE ALL COMMUNITIES
# ============================================================

if not results:

    raise ValueError(
        "No community sentiment data was processed."
    )


anomaly_df = pd.concat(
    results,
    ignore_index=True
)


# ============================================================
# 9. MATCH WITH DETECTED EVENT DATES
# ============================================================

event_dates = set(

    change_df.loc[
        change_df[
            "significant_change"
        ] == True,
        "date"
    ]

)


print(
    "\nDetected event dates:"
)

for event_date in sorted(event_dates):

    print(
        " -",
        event_date
    )


anomaly_df[
    "event_date"
] = anomaly_df[
    "date"
].isin(
    event_dates
)


# ============================================================
# 10. GET EVENT-DAY SENTIMENT ANOMALIES
# ============================================================

event_day_anomalies = anomaly_df[
    (
        anomaly_df[
            "event_date"
        ]
        == True
    )
    &
    (
        anomaly_df[
            "sentiment_anomaly"
        ]
        == True
    )
].copy()


# ============================================================
# 11. CALCULATE SUMMARY
# ============================================================

total_anomalies = int(
    anomaly_df[
        "sentiment_anomaly"
    ].sum()
)


event_day_anomaly_count = len(
    event_day_anomalies
)


total_communities = (
    anomaly_df[
        "subreddit"
    ]
    .nunique()
)


total_events = len(
    event_dates
)


# ============================================================
# 12. EVENT ANOMALY COVERAGE
# ============================================================

event_dates_with_anomaly = (
    event_day_anomalies[
        "date"
    ]
    .nunique()
)


if total_events > 0:

    event_anomaly_coverage = (

        event_dates_with_anomaly
        / total_events
        * 100

    )

else:

    event_anomaly_coverage = 0


# ============================================================
# 13. PRINT ANOMALY SUMMARY
# ============================================================

print("\n" + "-" * 60)
print("ANOMALY SUMMARY")
print("-" * 60)


print(
    "Communities analyzed:",
    total_communities
)


print(
    "Detected events:",
    total_events
)


print(
    "Total sentiment anomalies:",
    total_anomalies
)


print(
    "Event-day sentiment anomalies:",
    event_day_anomaly_count
)


print(
    "Event dates with sentiment anomaly:",
    event_dates_with_anomaly
)


print(
    "Event anomaly coverage:",
    round(
        event_anomaly_coverage,
        2
    ),
    "%"
)


# ============================================================
# 14. DISPLAY EVENT-DAY ANOMALIES
# ============================================================

print("\n" + "-" * 60)
print("EVENT-DAY SENTIMENT ANOMALIES")
print("-" * 60)


if len(event_day_anomalies) > 0:

    display_columns = [

        "date",

        "subreddit",

        sentiment_column,

        "sentiment_zscore",

        "anomaly_direction"

    ]


    print(
        event_day_anomalies[
            display_columns
        ]
        .sort_values(
            [
                "date",
                "subreddit"
            ]
        )
        .to_string(
            index=False
        )
    )

else:

    print(
        "No sentiment anomalies detected "
        "on event dates."
    )


# ============================================================
# 15. OVERALL ANOMALY BREAKDOWN
# ============================================================

print("\n" + "-" * 60)
print("ANOMALY DIRECTION BREAKDOWN")
print("-" * 60)


direction_counts = (
    anomaly_df[
        anomaly_df[
            "sentiment_anomaly"
        ] == True
    ][
        "anomaly_direction"
    ]
    .value_counts()
)


if len(direction_counts) > 0:

    print(
        direction_counts.to_string()
    )

else:

    print(
        "No anomalies detected."
    )


# ============================================================
# 16. COMMUNITY-WISE ANOMALY COUNT
# ============================================================

print("\n" + "-" * 60)
print("COMMUNITY-WISE ANOMALY COUNT")
print("-" * 60)


community_anomaly_counts = (

    anomaly_df[
        anomaly_df[
            "sentiment_anomaly"
        ] == True
    ]
    .groupby(
        "subreddit"
    )
    .size()
    .sort_values(
        ascending=False
    )

)


if len(
    community_anomaly_counts
) > 0:

    print(
        community_anomaly_counts.to_string()
    )

else:

    print(
        "No community anomalies detected."
    )


# ============================================================
# 17. SAVE RESULTS
# ============================================================

anomaly_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 18. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)

print(
    "SENTIMENT ANOMALY DETECTION COMPLETED"
)

print("=" * 60)


print(
    "Total rows analyzed:",
    len(anomaly_df)
)


print(
    "Total anomalies:",
    total_anomalies
)


print(
    "Event-day anomalies:",
    event_day_anomaly_count
)


print(
    "Event anomaly coverage:",
    round(
        event_anomaly_coverage,
        2
    ),
    "%"
)


print(
    "\nOutput file:"
)

print(
    OUTPUT_FILE
)


print(
    "\nPhase 9 completed successfully!"
)