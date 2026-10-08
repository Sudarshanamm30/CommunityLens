import pandas as pd
import networkx as nx
import os
import time


# ==================================================
# 1. FILE PATHS
# ==================================================

INPUT_FILE = "data/processed/temporal_reddit_data.csv"

OUTPUT_EDGES = "data/processed/interaction_edges.csv"

OUTPUT_NODES = "data/processed/interaction_nodes.csv"


# ==================================================
# 2. LOAD DATA
# ==================================================

print("Loading temporal Reddit dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("\nDataset loaded successfully!")

print("Rows:", len(df))

print("Columns:")
print(df.columns.tolist())


# ==================================================
# 3. CHECK REQUIRED COLUMNS
# ==================================================

required_columns = [
    "id",
    "parent_id",
    "submission_id",
    "subreddit",
    "created_utc"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR!")

    print("Missing columns:")
    print(missing_columns)

    exit()


# ==================================================
# 4. CLEAN IDENTIFIERS
# ==================================================

df["id"] = (
    df["id"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["parent_id"] = (
    df["parent_id"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["submission_id"] = (
    df["submission_id"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["subreddit"] = (
    df["subreddit"]
    .fillna("Unknown")
    .astype(str)
)


# ==================================================
# 5. REMOVE INVALID COMMENT IDS
# ==================================================

df = df[
    df["id"] != ""
].copy()


print("\nValid comments:", len(df))


# ==================================================
# 6. CREATE COMMENT ID SET
# ==================================================

comment_ids = set(
    df["id"]
)


# ==================================================
# 7. BUILD INTERACTION EDGES
# ==================================================

print("\nBuilding interaction edges...")

edges = []

for _, row in df.iterrows():

    comment_id = row["id"]

    parent_id = row["parent_id"]

    community = row["subreddit"]

    timestamp = row["created_utc"]


    # ----------------------------------------------
    # Parent-child relationship
    # ----------------------------------------------

    if parent_id == "":
        continue


    # Reddit parent IDs can contain prefixes such as
    # t1_ for comments and t3_ for submissions.
    #
    # Extract the actual identifier.

    clean_parent = parent_id

    if clean_parent.startswith("t1_"):
        clean_parent = clean_parent[3:]

    elif clean_parent.startswith("t3_"):
        clean_parent = clean_parent[3:]


    # ----------------------------------------------
    # Only create comment-to-comment edges
    # ----------------------------------------------

    if clean_parent in comment_ids:

        edges.append({

            "source": clean_parent,

            "target": comment_id,

            "subreddit": community,

            "created_utc": timestamp

        })


# ==================================================
# 8. CREATE EDGE DATAFRAME
# ==================================================

edges_df = pd.DataFrame(
    edges
)


print("\nInteraction edges created:", len(edges_df))


# ==================================================
# 9. CREATE NODE DATAFRAME
# ==================================================

nodes_df = df[
    [
        "id",
        "subreddit",
        "created_utc"
    ]
].copy()


nodes_df = nodes_df.rename(
    columns={
        "id": "node"
    }
)


# ==================================================
# 10. CREATE NETWORKX GRAPH
# ==================================================

print("\nCreating NetworkX graph...")

G = nx.DiGraph()


# Add nodes

for _, row in nodes_df.iterrows():

    G.add_node(

        row["node"],

        subreddit=row["subreddit"],

        created_utc=row["created_utc"]

    )


# Add edges

for _, row in edges_df.iterrows():

    G.add_edge(

        row["source"],

        row["target"],

        subreddit=row["subreddit"],

        created_utc=row["created_utc"]

    )


# ==================================================
# 11. GRAPH STATISTICS
# ==================================================

number_of_nodes = G.number_of_nodes()

number_of_edges = G.number_of_edges()

number_of_communities = (
    nodes_df["subreddit"].nunique()
)


print("\n")
print("=" * 60)

print("INTERACTION GRAPH COMPLETED")

print("=" * 60)

print("\nNumber of nodes:", number_of_nodes)

print("Number of edges:", number_of_edges)

print("Number of communities:", number_of_communities)


# ==================================================
# 12. DEGREE STATISTICS
# ==================================================

if number_of_nodes > 0:

    in_degrees = dict(
        G.in_degree()
    )

    out_degrees = dict(
        G.out_degree()
    )

    average_in_degree = (
        sum(in_degrees.values())
        / number_of_nodes
    )

    average_out_degree = (
        sum(out_degrees.values())
        / number_of_nodes
    )

else:

    average_in_degree = 0

    average_out_degree = 0


print(
    "Average in-degree:",
    round(average_in_degree, 3)
)

print(
    "Average out-degree:",
    round(average_out_degree, 3)
)


# ==================================================
# 13. SAVE FILES
# ==================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


edges_df.to_csv(
    OUTPUT_EDGES,
    index=False
)


nodes_df.to_csv(
    OUTPUT_NODES,
    index=False
)


# ==================================================
# 14. SAVE GRAPH
# ==================================================

GRAPH_FILE = (
    "data/processed/interaction_graph.gpickle"
)

try:

    nx.write_gpickle(
        G,
        GRAPH_FILE
    )

    graph_saved = True

except AttributeError:

    # Compatibility with newer NetworkX versions
    import pickle

    with open(
        GRAPH_FILE,
        "wb"
    ) as file:

        pickle.dump(
            G,
            file
        )

    graph_saved = True


# ==================================================
# 15. COMMUNITY EDGE SUMMARY
# ==================================================

if len(edges_df) > 0:

    community_edges = (
        edges_df
        .groupby("subreddit")
        .size()
        .reset_index(
            name="interaction_count"
        )
        .sort_values(
            "interaction_count",
            ascending=False
        )
    )

else:

    community_edges = pd.DataFrame(
        columns=[
            "subreddit",
            "interaction_count"
        ]
    )


community_edges.to_csv(
    "data/processed/community_interactions.csv",
    index=False
)


# ==================================================
# 16. FINAL OUTPUT
# ==================================================

print("\nFiles created:")

print(
    "1. data/processed/interaction_edges.csv"
)

print(
    "2. data/processed/interaction_nodes.csv"
)

print(
    "3. data/processed/interaction_graph.gpickle"
)

print(
    "4. data/processed/community_interactions.csv"
)


print("\nTop communities by interaction count:")

print(
    community_edges.head(10)
)


print("\nPhase 3 completed successfully!")