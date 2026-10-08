import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path


# ============================================================
# PHASE 8 — EVALUATION
# ============================================================

print("=" * 60)
print("PHASE 8 — EVALUATION")
print("=" * 60)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CHANGE_FILE = BASE_DIR / "data" / "processed" / "change_detection.csv"
EDGE_FILE = BASE_DIR / "data" / "processed" / "interaction_edges.csv"
COMPARISON_FILE = BASE_DIR / "data" / "processed" / "fixed_vs_event.csv"

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "evaluation_metrics.csv"


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

change_df = pd.read_csv(CHANGE_FILE)
edges_df = pd.read_csv(EDGE_FILE)
comparison_df = pd.read_csv(COMPARISON_FILE)


# ------------------------------------------------------------
# FIX DATE COLUMNS
# ------------------------------------------------------------

change_df["date"] = pd.to_datetime(
    change_df["date"]
).dt.date


# interaction_edges.csv contains created_utc,
# not date. Therefore create date from created_utc.

edges_df["created_utc"] = pd.to_datetime(
    edges_df["created_utc"]
)

edges_df["date"] = edges_df[
    "created_utc"
].dt.date


comparison_df["date"] = pd.to_datetime(
    comparison_df["date"]
).dt.date


print("\nData loaded successfully.")

print("Interaction edge columns:")
print(edges_df.columns.tolist())


# ============================================================
# 1. DETECTED EVENTS
# ============================================================

event_dates = sorted(
    change_df.loc[
        change_df["significant_change"] == True,
        "date"
    ].unique()
)

print("\nDetected event dates:")

for date in event_dates:
    print(" -", date)

print("\nNumber of detected events:", len(event_dates))


# ============================================================
# 2. UPDATE DATES
# ============================================================

fixed_dates = sorted(
    comparison_df.loc[
        comparison_df["method"] == "Fixed Interval",
        "date"
    ].unique()
)

event_update_dates = sorted(
    comparison_df.loc[
        comparison_df["method"] == "Event Triggered",
        "date"
    ].unique()
)


# ============================================================
# 3. DETECTION DELAY
# ============================================================

print("\n" + "-" * 60)
print("DETECTION DELAY")
print("-" * 60)


delay_results = []


for event_date in event_dates:

    # Event-triggered update
    event_future = [
        d for d in event_update_dates
        if d >= event_date
    ]

    if event_future:

        event_update = min(event_future)

        event_delay = (
            event_update - event_date
        ).days

    else:

        event_update = None
        event_delay = np.nan


    # Fixed-interval update
    fixed_future = [
        d for d in fixed_dates
        if d >= event_date
    ]

    if fixed_future:

        fixed_update = min(fixed_future)

        fixed_delay = (
            fixed_update - event_date
        ).days

    else:

        fixed_update = None
        fixed_delay = np.nan


    delay_results.append({

        "event_date": event_date,

        "fixed_update_date": fixed_update,

        "fixed_detection_delay_days": fixed_delay,

        "event_update_date": event_update,

        "event_detection_delay_days": event_delay

    })


delay_df = pd.DataFrame(
    delay_results
)


print(delay_df.to_string(index=False))


avg_fixed_delay = delay_df[
    "fixed_detection_delay_days"
].mean()


avg_event_delay = delay_df[
    "event_detection_delay_days"
].mean()


print(
    "\nAverage fixed-interval detection delay:",
    round(avg_fixed_delay, 2),
    "days"
)

print(
    "Average event-triggered detection delay:",
    round(avg_event_delay, 2),
    "days"
)


# ============================================================
# 4. COMMUNITY STABILITY
# ============================================================

print("\n" + "-" * 60)
print("COMMUNITY STABILITY")
print("-" * 60)


def build_graph_until(date):

    subset = edges_df[
        edges_df["date"] <= date
    ]

    graph = nx.Graph()

    for _, row in subset.iterrows():

        source = str(row["source"])
        target = str(row["target"])

        if source != target:

            graph.add_edge(
                source,
                target
            )

    return graph


def detect_communities(graph):

    if graph.number_of_nodes() == 0:
        return []

    communities = list(
        nx.community.greedy_modularity_communities(
            graph
        )
    )

    return [
        set(c)
        for c in communities
    ]


def best_jaccard(
    previous_communities,
    current_communities
):

    if (
        not previous_communities
        or not current_communities
    ):
        return 0.0


    scores = []


    for current in current_communities:

        best_score = 0.0


        for previous in previous_communities:

            union = current | previous


            if len(union) == 0:

                score = 0.0

            else:

                score = (
                    len(current & previous)
                    / len(union)
                )


            best_score = max(
                best_score,
                score
            )


        scores.append(best_score)


    if scores:

        return float(
            np.mean(scores)
        )

    return 0.0


# ------------------------------------------------------------
# Event-triggered stability
# ------------------------------------------------------------

previous_communities = None

event_stability_results = []


for date in event_update_dates:

    graph = build_graph_until(date)

    communities = detect_communities(
        graph
    )


    if previous_communities is None:

        stability = np.nan

    else:

        stability = best_jaccard(
            previous_communities,
            communities
        )


    event_stability_results.append({

        "date": date,

        "method": "Event Triggered",

        "communities": len(
            communities
        ),

        "stability": stability

    })


    previous_communities = communities


event_stability_df = pd.DataFrame(
    event_stability_results
)


# ------------------------------------------------------------
# Fixed-interval stability
# ------------------------------------------------------------

previous_communities = None

fixed_stability_results = []


for date in fixed_dates:

    graph = build_graph_until(date)

    communities = detect_communities(
        graph
    )


    if previous_communities is None:

        stability = np.nan

    else:

        stability = best_jaccard(
            previous_communities,
            communities
        )


    fixed_stability_results.append({

        "date": date,

        "method": "Fixed Interval",

        "communities": len(
            communities
        ),

        "stability": stability

    })


    previous_communities = communities


fixed_stability_df = pd.DataFrame(
    fixed_stability_results
)


stability_df = pd.concat(
    [
        fixed_stability_df,
        event_stability_df
    ],
    ignore_index=True
)


print("\nStability results:")

print(
    stability_df.to_string(
        index=False
    )
)


avg_fixed_stability = fixed_stability_df[
    "stability"
].mean()


avg_event_stability = event_stability_df[
    "stability"
].mean()


print(
    "\nAverage Fixed Interval stability:",
    round(avg_fixed_stability, 4)
)

print(
    "Average Event Triggered stability:",
    round(avg_event_stability, 4)
)


# ============================================================
# 5. COMPUTATIONAL COST
# ============================================================

print("\n" + "-" * 60)
print("COMPUTATIONAL COST")
print("-" * 60)


fixed_time = comparison_df.loc[
    comparison_df["method"] == "Fixed Interval",
    "computation_time"
].sum()


event_time = comparison_df.loc[
    comparison_df["method"] == "Event Triggered",
    "computation_time"
].sum()


fixed_updates = len(fixed_dates)

event_updates = len(
    event_update_dates
)


time_saved = (
    fixed_time - event_time
)


if fixed_time > 0:

    percentage_saved = (
        time_saved / fixed_time
    ) * 100

else:

    percentage_saved = 0


update_reduction = (
    (fixed_updates - event_updates)
    / fixed_updates
) * 100


print(
    "Fixed updates:",
    fixed_updates
)

print(
    "Event-triggered updates:",
    event_updates
)

print(
    "Fixed computation time:",
    round(fixed_time, 4),
    "seconds"
)

print(
    "Event-triggered computation time:",
    round(event_time, 4),
    "seconds"
)

print(
    "Computation time saved:",
    round(time_saved, 4),
    "seconds"
)

print(
    "Percentage computation saved:",
    round(percentage_saved, 2),
    "%"
)

print(
    "Community update reduction:",
    round(update_reduction, 2),
    "%"
)


# ============================================================
# 6. EVENT CAPTURE
# ============================================================

print("\n" + "-" * 60)
print("EVENT CAPTURE")
print("-" * 60)


captured_events = 0

event_capture_results = []


for event_date in event_dates:

    captured = (
        event_date
        in event_update_dates
    )


    if captured:

        captured_events += 1


    event_capture_results.append({

        "event_date": event_date,

        "event_triggered_update": captured

    })


capture_df = pd.DataFrame(
    event_capture_results
)


print(
    capture_df.to_string(
        index=False
    )
)


event_capture_rate = (

    captured_events
    / len(event_dates)
    * 100

    if len(event_dates) > 0

    else 0

)


print(
    "\nEvent capture rate:",
    round(event_capture_rate, 2),
    "%"
)


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

summary = pd.DataFrame({

    "metric": [

        "Fixed Interval Updates",

        "Event Triggered Updates",

        "Update Reduction (%)",

        "Fixed Computation Time (seconds)",

        "Event Computation Time (seconds)",

        "Computation Time Saved (seconds)",

        "Computation Time Saved (%)",

        "Average Fixed Detection Delay (days)",

        "Average Event Detection Delay (days)",

        "Average Fixed Community Stability",

        "Average Event Community Stability",

        "Detected Events",

        "Captured Events",

        "Event Capture Rate (%)"

    ],


    "value": [

        fixed_updates,

        event_updates,

        update_reduction,

        fixed_time,

        event_time,

        time_saved,

        percentage_saved,

        avg_fixed_delay,

        avg_event_delay,

        avg_fixed_stability,

        avg_event_stability,

        len(event_dates),

        captured_events,

        event_capture_rate

    ]

})


# ============================================================
# SAVE
# ============================================================

summary.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 60)
print("FINAL EVALUATION SUMMARY")
print("=" * 60)

print(
    summary.to_string(
        index=False
    )
)


print("\nOutput file:")
print(OUTPUT_FILE)

print("\nPhase 8 completed successfully!")