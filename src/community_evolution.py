import pandas as pd
import networkx as nx
import os


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
    "data/processed/community_evolution.csv"
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
# 4. GET EVENT DATES
# ==================================================

event_dates = sorted(
    changes[
        changes["significant_change"] == True
    ]["date"]
    .tolist()
)


print("\nEvent dates:")

for date in event_dates:

    print(
        date.strftime("%Y-%m-%d")
    )


# ==================================================
# 5. BUILD GRAPH
# ==================================================

def build_graph_until_date(
    edge_data,
    event_date
):

    selected_edges = edge_data[
        edge_data["date"] <= event_date
    ]

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
# 6. DETECT COMMUNITIES
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
# 7. CALCULATE JACCARD SIMILARITY
# ==================================================

def jaccard_similarity(
    community_a,
    community_b
):

    intersection = (
        len(
            community_a
            &
            community_b
        )
    )


    union = (
        len(
            community_a
            |
            community_b
        )
    )


    if union == 0:

        return 0


    return (
        intersection
        /
        union
    )


# ==================================================
# 8. FIND BEST COMMUNITY MATCH
# ==================================================

def best_match(
    old_community,
    new_communities
):

    best_index = None

    best_score = 0


    for index, new_community in enumerate(
        new_communities
    ):

        score = jaccard_similarity(
            old_community,
            new_community
        )


        if score > best_score:

            best_score = score

            best_index = index


    return best_index, best_score


# ==================================================
# 9. DETECT EVOLUTION
# ==================================================

def detect_evolution(
    previous_communities,
    current_communities
):

    events = []


    # ------------------------------------------------
    # No previous communities
    # ------------------------------------------------

    if not previous_communities:

        for index, community in enumerate(
            current_communities
        ):

            events.append({

                "event_type":
                    "Birth",

                "old_community":
                    None,

                "new_community":
                    index,

                "similarity":
                    0,

                "community_size":
                    len(community)

            })


        return events


    # =================================================
    # COMMUNITY BIRTH
    # =================================================

    for new_index, new_community in enumerate(
        current_communities
    ):

        best_index, best_score = best_match(
            new_community,
            previous_communities
        )


        if best_score < 0.20:

            events.append({

                "event_type":
                    "Birth",

                "old_community":
                    None,

                "new_community":
                    new_index,

                "similarity":
                    best_score,

                "community_size":
                    len(new_community)

            })


    # =================================================
    # COMMUNITY DISAPPEARANCE
    # =================================================

    for old_index, old_community in enumerate(
        previous_communities
    ):

        best_index, best_score = best_match(
            old_community,
            current_communities
        )


        if best_score < 0.20:

            events.append({

                "event_type":
                    "Disappearance",

                "old_community":
                    old_index,

                "new_community":
                    None,

                "similarity":
                    best_score,

                "community_size":
                    len(old_community)

            })


    # =================================================
    # SPLIT DETECTION
    # =================================================

    for old_index, old_community in enumerate(
        previous_communities
    ):

        matching_new = []


        for new_index, new_community in enumerate(
            current_communities
        ):

            score = jaccard_similarity(
                old_community,
                new_community
            )


            if score >= 0.20:

                matching_new.append(
                    (
                        new_index,
                        score
                    )
                )


        if len(matching_new) >= 2:

            events.append({

                "event_type":
                    "Split",

                "old_community":
                    old_index,

                "new_community":
                    ",".join(
                        str(x[0])
                        for x in matching_new
                    ),

                "similarity":
                    max(
                        x[1]
                        for x in matching_new
                    ),

                "community_size":
                    len(old_community)

            })


    # =================================================
    # MERGE DETECTION
    # =================================================

    for new_index, new_community in enumerate(
        current_communities
    ):

        matching_old = []


        for old_index, old_community in enumerate(
            previous_communities
        ):

            score = jaccard_similarity(
                old_community,
                new_community
            )


            if score >= 0.20:

                matching_old.append(
                    (
                        old_index,
                        score
                    )
                )


        if len(matching_old) >= 2:

            events.append({

                "event_type":
                    "Merge",

                "old_community":
                    ",".join(
                        str(x[0])
                        for x in matching_old
                    ),

                "new_community":
                    new_index,

                "similarity":
                    max(
                        x[1]
                        for x in matching_old
                    ),

                "community_size":
                    len(new_community)

            })


    # =================================================
    # MEMBERSHIP CHANGE
    # =================================================

    for new_index, new_community in enumerate(
        current_communities
    ):

        old_index, score = best_match(
            new_community,
            previous_communities
        )


        if (
            old_index is not None
            and
            0.20 <= score < 1.0
        ):

            events.append({

                "event_type":
                    "Membership Change",

                "old_community":
                    old_index,

                "new_community":
                    new_index,

                "similarity":
                    score,

                "community_size":
                    len(new_community)

            })


    return events


# ==================================================
# 10. PROCESS EVENTS
# ==================================================

print("\n")
print("=" * 60)

print(
    "COMMUNITY EVOLUTION ANALYSIS"
)

print("=" * 60)


previous_communities = []

results = []


for event_date in event_dates:

    print(
        "\nProcessing:",
        event_date.strftime("%Y-%m-%d")
    )


    # ----------------------------------------------
    # Build graph
    # ----------------------------------------------

    G = build_graph_until_date(
        edges,
        event_date
    )


    # ----------------------------------------------
    # Detect communities
    # ----------------------------------------------

    current_communities = detect_communities(
        G
    )


    print(
        "Communities detected:",
        len(current_communities)
    )


    # ----------------------------------------------
    # Detect evolution
    # ----------------------------------------------

    evolution_events = detect_evolution(
        previous_communities,
        current_communities
    )


    # ----------------------------------------------
    # Store results
    # ----------------------------------------------

    for event in evolution_events:

        results.append({

            "date":
                event_date,

            "event_type":
                event["event_type"],

            "old_community":
                event["old_community"],

            "new_community":
                event["new_community"],

            "similarity":
                event["similarity"],

            "community_size":
                event["community_size"],

            "number_of_communities":
                len(current_communities)

        })


    # ----------------------------------------------
    # Current becomes previous
    # ----------------------------------------------

    previous_communities = (
        current_communities
    )


# ==================================================
# 11. CREATE DATAFRAME
# ==================================================

evolution_df = pd.DataFrame(
    results
)


# ==================================================
# 12. SAVE
# ==================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


evolution_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# 13. DISPLAY RESULTS
# ==================================================

print("\n")
print("=" * 60)

print(
    "PHASE 6 — COMMUNITY EVOLUTION COMPLETED"
)

print("=" * 60)


print(
    "\nTotal evolution events:"
)

print(
    len(evolution_df)
)


if len(evolution_df) > 0:

    print(
        "\nEvolution event types:"
    )

    print(
        evolution_df[
            "event_type"
        ].value_counts()
    )


    print(
        "\nDetected evolution:"
    )

    print(
        evolution_df
    )

else:

    print(
        "\nNo community evolution events detected."
    )


print(
    "\nOutput file:"
)

print(
    OUTPUT_FILE
)


print(
    "\nPhase 6 completed successfully!"
)