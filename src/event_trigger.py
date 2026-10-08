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
    "data/processed/event_triggered_communities.csv"
)


# ==================================================
# 2. LOAD CHANGE DETECTION DATA
# ==================================================

print("Loading change detection results...")

changes = pd.read_csv(
    CHANGE_FILE
)

changes["date"] = pd.to_datetime(
    changes["date"]
)

print(
    "Days available:",
    len(changes)
)


# ==================================================
# 3. LOAD INTERACTION EDGES
# ==================================================

print("\nLoading interaction edges...")

edges = pd.read_csv(
    EDGE_FILE,
    low_memory=False
)

print(
    "Interaction edges:",
    len(edges)
)


# ==================================================
# 4. CONVERT EDGE DATE
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


# ==================================================
# 5. GET EVENT DATES
# ==================================================

event_rows = changes[
    changes["significant_change"] == True
].copy()


event_dates = sorted(
    event_rows["date"].tolist()
)


print("\nEvent dates detected:")

for date in event_dates:

    print(
        date.strftime("%Y-%m-%d")
    )


# ==================================================
# 6. FUNCTION TO BUILD GRAPH
# ==================================================

def build_graph_until_date(
    edge_data,
    event_date
):

    # Use interactions up to the event date

    selected_edges = edge_data[
        edge_data["date"] <= event_date
    ].copy()


    G = nx.Graph()


    for _, row in selected_edges.iterrows():

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
# 7. DETECT COMMUNITIES
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
# 8. CALCULATE COMMUNITY STATISTICS
# ==================================================

results = []


print("\n")
print("=" * 60)

print("EVENT-TRIGGERED COMMUNITY DETECTION")

print("=" * 60)


for _, event in event_rows.iterrows():

    event_date = event["date"]


    print(
        "\nProcessing event:",
        event_date.strftime("%Y-%m-%d")
    )


    # ----------------------------------------------
    # Start timing
    # ----------------------------------------------

    start_time = time.perf_counter()


    # ----------------------------------------------
    # Build graph using interactions up to event
    # ----------------------------------------------

    G = build_graph_until_date(
        edges,
        event_date
    )


    # ----------------------------------------------
    # Detect communities
    # ----------------------------------------------

    communities = detect_communities(
        G
    )


    # ----------------------------------------------
    # End timing
    # ----------------------------------------------

    end_time = time.perf_counter()


    computation_time = (
        end_time - start_time
    )


    # ----------------------------------------------
    # Community statistics
    # ----------------------------------------------

    number_of_communities = len(
        communities
    )


    community_sizes = [
        len(c)
        for c in communities
    ]


    if community_sizes:

        largest_community = max(
            community_sizes
        )

        average_community_size = (
            sum(community_sizes)
            /
            len(community_sizes)
        )

    else:

        largest_community = 0

        average_community_size = 0


    # ----------------------------------------------
    # Store result
    # ----------------------------------------------

    results.append({

        "event_date":
            event_date,

        "change_type":
            event["change_type"],

        "change_direction":
            event["change_direction"],

        "change_score":
            event["change_score"],

        "graph_nodes":
            G.number_of_nodes(),

        "graph_edges":
            G.number_of_edges(),

        "communities_detected":
            number_of_communities,

        "largest_community":
            largest_community,

        "average_community_size":
            average_community_size,

        "computation_time_seconds":
            computation_time

    })


    print(
        "Nodes:",
        G.number_of_nodes()
    )

    print(
        "Edges:",
        G.number_of_edges()
    )

    print(
        "Communities:",
        number_of_communities
    )

    print(
        "Computation time:",
        round(
            computation_time,
            4
        ),
        "seconds"
    )


# ==================================================
# 9. CREATE RESULT DATAFRAME
# ==================================================

results_df = pd.DataFrame(
    results
)


# ==================================================
# 10. SAVE RESULTS
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
# 11. FINAL OUTPUT
# ==================================================

print("\n")
print("=" * 60)

print(
    "PHASE 5 — EVENT TRIGGER COMPLETED"
)

print("=" * 60)


print(
    "\nNumber of events:",
    len(results_df)
)


print(
    "\nEvent-triggered community results:"
)


if len(results_df) > 0:

    print(
        results_df[
            [
                "event_date",
                "change_type",
                "change_direction",
                "change_score",
                "graph_nodes",
                "graph_edges",
                "communities_detected",
                "computation_time_seconds"
            ]
        ]
    )

else:

    print(
        "No significant events detected."
    )


print(
    "\nOutput file:"
)

print(
    OUTPUT_FILE
)


print(
    "\nPhase 5 completed successfully!"
)