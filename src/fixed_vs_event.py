import pandas as pd
import networkx as nx
import os
import time


# ==================================================
# 1. FILE PATHS
# ==================================================

CHANGE_FILE = (
    "data/processed/change_detection.csv"
)

EDGE_FILE = (
    "data/processed/interaction_edges.csv"
)

OUTPUT_FILE = (
    "data/processed/fixed_vs_event.csv"
)


# ==================================================
# 2. LOAD DATA
# ==================================================

print("Loading change detection data...")

changes = pd.read_csv(
    CHANGE_FILE
)

changes["date"] = pd.to_datetime(
    changes["date"]
)


print("\nLoading interaction edges...")

edges = pd.read_csv(
    EDGE_FILE,
    low_memory=False
)


# ==================================================
# 3. CONVERT EDGE DATE
# ==================================================

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


# ==================================================
# 4. GET ALL DATES
# ==================================================

all_dates = sorted(
    changes["date"].unique()
)


start_date = min(
    all_dates
)

end_date = max(
    all_dates
)


print("\nDataset period:")

print(
    start_date.strftime("%Y-%m-%d"),
    "to",
    end_date.strftime("%Y-%m-%d")
)


# ==================================================
# 5. FIXED INTERVAL
# ==================================================

FIXED_INTERVAL = 3


fixed_dates = []


current_date = start_date


while current_date <= end_date:

    fixed_dates.append(
        current_date
    )

    current_date = (
        current_date
        +
        pd.Timedelta(
            days=FIXED_INTERVAL
        )
    )


# ==================================================
# 6. EVENT-TRIGGERED DATES
# ==================================================

event_dates = sorted(
    changes[
        changes["significant_change"] == True
    ]["date"]
    .tolist()
)


# ==================================================
# 7. PRINT SCHEDULES
# ==================================================

print("\n")
print("=" * 60)

print("FIXED-INTERVAL SCHEDULE")

print("=" * 60)

for date in fixed_dates:

    print(
        date.strftime("%Y-%m-%d")
    )


print("\n")
print("=" * 60)

print("EVENT-TRIGGERED SCHEDULE")

print("=" * 60)

for date in event_dates:

    print(
        date.strftime("%Y-%m-%d")
    )


# ==================================================
# 8. BUILD GRAPH
# ==================================================

def build_graph(
    edge_data,
    target_date
):

    selected = edge_data[
        edge_data["date"] <= target_date
    ]


    G = nx.Graph()


    for _, row in selected.iterrows():

        source = row["source"]

        target = row["target"]


        if source == target:

            continue


        G.add_edge(
            source,
            target
        )


    return G


# ==================================================
# 9. DETECT COMMUNITIES
# ==================================================

def detect_communities(G):

    if G.number_of_nodes() == 0:

        return []


    if G.number_of_edges() == 0:

        return [
            {node}
            for node in G.nodes()
        ]


    communities = (
        nx.community
        .greedy_modularity_communities(
            G
        )
    )


    return [
        set(c)
        for c in communities
    ]


# ==================================================
# 10. RUN FIXED INTERVAL METHOD
# ==================================================

print("\n")
print("=" * 60)

print(
    "RUNNING FIXED-INTERVAL METHOD"
)

print("=" * 60)


fixed_results = []


fixed_total_time = 0


for date in fixed_dates:

    start_time = time.perf_counter()


    G = build_graph(
        edges,
        date
    )


    communities = detect_communities(
        G
    )


    end_time = time.perf_counter()


    elapsed = (
        end_time
        -
        start_time
    )


    fixed_total_time += elapsed


    fixed_results.append({

        "date":
            date,

        "method":
            "Fixed Interval",

        "nodes":
            G.number_of_nodes(),

        "edges":
            G.number_of_edges(),

        "communities":
            len(communities),

        "computation_time":
            elapsed

    })


    print(
        date.strftime("%Y-%m-%d"),
        "| Communities:",
        len(communities),
        "| Time:",
        round(
            elapsed,
            4
        ),
        "seconds"
    )


# ==================================================
# 11. RUN EVENT METHOD
# ==================================================

print("\n")
print("=" * 60)

print(
    "RUNNING EVENT-TRIGGERED METHOD"
)

print("=" * 60)


event_results = []


event_total_time = 0


for date in event_dates:

    start_time = time.perf_counter()


    G = build_graph(
        edges,
        date
    )


    communities = detect_communities(
        G
    )


    end_time = time.perf_counter()


    elapsed = (
        end_time
        -
        start_time
    )


    event_total_time += elapsed


    event_results.append({

        "date":
            date,

        "method":
            "Event Triggered",

        "nodes":
            G.number_of_nodes(),

        "edges":
            G.number_of_edges(),

        "communities":
            len(communities),

        "computation_time":
            elapsed

    })


    print(
        date.strftime("%Y-%m-%d"),
        "| Communities:",
        len(communities),
        "| Time:",
        round(
            elapsed,
            4
        ),
        "seconds"
    )


# ==================================================
# 12. COMBINE RESULTS
# ==================================================

results_df = pd.DataFrame(
    fixed_results
    +
    event_results
)


# ==================================================
# 13. SUMMARY
# ==================================================

fixed_updates = len(
    fixed_results
)

event_updates = len(
    event_results
)


computation_saved = (
    fixed_total_time
    -
    event_total_time
)


if fixed_total_time > 0:

    percentage_saved = (
        computation_saved
        /
        fixed_total_time
    ) * 100

else:

    percentage_saved = 0


# ==================================================
# 14. SAVE RESULTS
# ==================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# 15. DISPLAY SUMMARY
# ==================================================

print("\n")
print("=" * 60)

print(
    "PHASE 7 — FIXED VS EVENT COMPARISON"
)

print("=" * 60)


print(
    "\nFixed-interval updates:",
    fixed_updates
)


print(
    "Event-triggered updates:",
    event_updates
)


print(
    "\nFixed-interval computation time:",
    round(
        fixed_total_time,
        4
    ),
    "seconds"
)


print(
    "Event-triggered computation time:",
    round(
        event_total_time,
        4
    ),
    "seconds"
)


print(
    "\nEstimated computation time saved:",
    round(
        computation_saved,
        4
    ),
    "seconds"
)


print(
    "Estimated percentage saved:",
    round(
        percentage_saved,
        2
    ),
    "%"
)


# ==================================================
# 16. UPDATE REDUCTION
# ==================================================

if fixed_updates > 0:

    update_reduction = (
        (
            fixed_updates
            -
            event_updates
        )
        /
        fixed_updates
    ) * 100

else:

    update_reduction = 0


print(
    "\nCommunity update reduction:",
    round(
        update_reduction,
        2
    ),
    "%"
)


# ==================================================
# 17. RESULT TABLE
# ==================================================

print("\nComparison:")

print(
    results_df
)


print(
    "\nOutput file:"
)

print(
    OUTPUT_FILE
)


print(
    "\nPhase 7 completed successfully!"
)