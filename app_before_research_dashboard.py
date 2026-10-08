import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CommunityLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# INITIALIZATION
# ============================================================

analyzer = SentimentIntensityAnalyzer()

CSV_FILE = "data/processed/reddit_vader_sentiment.csv"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #777777;
        margin-top: 0px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-top: 20px;
    }

    .info-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
        text-align: center;
    }

    .finding-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        background-color: #f8f9fa;
        margin-bottom: 10px;
    }

    .small-text {
        color: #777777;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🔍 CommunityLens</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Community-Aware Sentiment Intelligence Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "CommunityLens analyzes Reddit conversations across communities "
    "using VADER sentiment analysis and transforms them into "
    "interactive community-level insights."
)


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(CSV_FILE):

    st.error(
        "❌ Dataset not found.\n\n"
        "Please make sure the following file exists:\n"
        "`data/processed/reddit_vader_sentiment.csv`"
    )

    st.stop()


df = pd.read_csv(CSV_FILE)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "subreddit",
    "body",
    "vader_sentiment",
    "vader_positive",
    "vader_negative",
    "vader_neutral",
    "vader_compound"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"❌ Missing columns: {missing_columns}"
    )

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

df["subreddit"] = (
    df["subreddit"]
    .fillna("Unknown")
    .astype(str)
)

df["body"] = (
    df["body"]
    .fillna("")
    .astype(str)
)

df = df[
    df["body"].str.strip() != ""
].copy()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Dashboard Controls")

st.sidebar.markdown("---")

st.sidebar.metric(
    "📄 Dataset Records",
    len(df)
)

st.sidebar.metric(
    "👥 Communities",
    df["subreddit"].nunique()
)


st.sidebar.markdown("---")

communities = sorted(
    df["subreddit"].unique()
)


selected_community = st.sidebar.selectbox(
    "🏘️ Select Community",
    ["All Communities"] + communities
)


# ============================================================
# FILTER DATA
# ============================================================

if selected_community == "All Communities":

    filtered_df = df.copy()

else:

    filtered_df = df[
        df["subreddit"] == selected_community
    ].copy()


# ============================================================
# DOWNLOAD DATASET
# ============================================================

st.sidebar.markdown("---")

st.sidebar.download_button(
    label="📥 Download VADER Dataset",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="communitylens_vader_results.csv",
    mime="text/csv"
)


# ============================================================
# SECTION 1 — EXECUTIVE OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📊 Executive Overview</div>',
    unsafe_allow_html=True
)


total_comments = len(filtered_df)

total_communities = (
    filtered_df["subreddit"].nunique()
)


positive_count = (
    filtered_df["vader_sentiment"]
    .eq("Positive")
    .sum()
)

negative_count = (
    filtered_df["vader_sentiment"]
    .eq("Negative")
    .sum()
)

neutral_count = (
    filtered_df["vader_sentiment"]
    .eq("Neutral")
    .sum()
)


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "💬 Comments",
        total_comments
    )


with col2:

    st.metric(
        "🏘️ Communities",
        total_communities
    )


with col3:

    st.metric(
        "😊 Positive",
        positive_count
    )


with col4:

    st.metric(
        "😐 Neutral",
        neutral_count
    )


with col5:

    st.metric(
        "😞 Negative",
        negative_count
    )


# ============================================================
# SECTION 2 — COMMUNITY INTELLIGENCE
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Community Intelligence</div>',
    unsafe_allow_html=True
)


# Create community statistics

community_stats = (
    df.groupby("subreddit")
    .agg(
        Comments=("body", "count"),
        Average_VADER=("vader_compound", "mean")
    )
    .reset_index()
)


# Sentiment percentages

sentiment_counts_all = (
    df.groupby(
        ["subreddit", "vader_sentiment"]
    )
    .size()
    .reset_index(name="Count")
)


sentiment_pivot = (
    sentiment_counts_all
    .pivot(
        index="subreddit",
        columns="vader_sentiment",
        values="Count"
    )
    .fillna(0)
)


for sentiment in [
    "Positive",
    "Negative",
    "Neutral"
]:

    if sentiment not in sentiment_pivot.columns:

        sentiment_pivot[sentiment] = 0


sentiment_pivot["Total"] = (
    sentiment_pivot[
        ["Positive", "Negative", "Neutral"]
    ].sum(axis=1)
)


sentiment_pivot["Positive %"] = (
    sentiment_pivot["Positive"]
    / sentiment_pivot["Total"]
    * 100
)


sentiment_pivot["Negative %"] = (
    sentiment_pivot["Negative"]
    / sentiment_pivot["Total"]
    * 100
)


sentiment_pivot["Neutral %"] = (
    sentiment_pivot["Neutral"]
    / sentiment_pivot["Total"]
    * 100
)


# ============================================================
# COMMUNITY MOOD INDEX
# ============================================================

# Convert VADER compound (-1 to +1)
# into a 0 to 100 scale.

community_stats["Mood_Index"] = (
    (community_stats["Average_VADER"] + 1)
    / 2
    * 100
)


community_stats = community_stats.merge(
    sentiment_pivot[
        [
            "Positive %",
            "Negative %",
            "Neutral %"
        ]
    ].reset_index(),
    on="subreddit",
    how="left"
)


# ============================================================
# FIND KEY COMMUNITIES
# ============================================================

most_active = community_stats.loc[
    community_stats["Comments"].idxmax()
]


most_positive = community_stats.loc[
    community_stats["Mood_Index"].idxmax()
]


most_negative = community_stats.loc[
    community_stats["Mood_Index"].idxmin()
]


# ============================================================
# MOOD INDEX
# ============================================================

if selected_community == "All Communities":

    overall_average = df[
        "vader_compound"
    ].mean()

else:

    overall_average = filtered_df[
        "vader_compound"
    ].mean()


mood_index = (
    (overall_average + 1)
    / 2
    * 100
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🎯 Community Mood Index",
        f"{mood_index:.1f}/100"
    )


with col2:

    st.metric(
        "🔥 Most Positive",
        str(most_positive["subreddit"])
    )


with col3:

    st.metric(
        "⚠️ Most Negative",
        str(most_negative["subreddit"])
    )


with col4:

    st.metric(
        "💬 Most Active",
        str(most_active["subreddit"])
    )


# ============================================================
# MOOD INTERPRETATION
# ============================================================

if mood_index >= 70:

    mood_text = "Generally Positive"

elif mood_index >= 55:

    mood_text = "Slightly Positive"

elif mood_index >= 45:

    mood_text = "Balanced / Neutral"

elif mood_index >= 30:

    mood_text = "Slightly Negative"

else:

    mood_text = "Generally Negative"


st.info(
    f"🎯 **Current Community Mood:** {mood_text} "
    f"with a Mood Index of **{mood_index:.1f}/100**."
)


# ============================================================
# SECTION 3 — OVERALL SENTIMENT
# ============================================================

st.markdown(
    '<div class="section-title">🥧 Overall Sentiment Distribution</div>',
    unsafe_allow_html=True
)


sentiment_counts = (
    filtered_df["vader_sentiment"]
    .value_counts()
    .reset_index()
)


sentiment_counts.columns = [
    "Sentiment",
    "Count"
]


col1, col2 = st.columns(2)


with col1:

    fig_pie = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        hole=0.50,
        title="Overall Sentiment"
    )

    fig_pie.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )


with col2:

    fig_bar = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        text="Count",
        title="Sentiment Count"
    )

    fig_bar.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )


# ============================================================
# SECTION 4 — COMMUNITY-WISE SENTIMENT
# ============================================================

st.markdown(
    '<div class="section-title">🏘️ Community-wise Sentiment Analysis</div>',
    unsafe_allow_html=True
)


community_summary = (
    filtered_df
    .groupby(
        ["subreddit", "vader_sentiment"]
    )
    .size()
    .reset_index(name="Count")
)


fig_community = px.bar(
    community_summary,
    x="subreddit",
    y="Count",
    color="vader_sentiment",
    barmode="stack",
    title="Sentiment Distribution Across Communities"
)


fig_community.update_layout(
    xaxis_title="Community",
    yaxis_title="Number of Comments"
)


st.plotly_chart(
    fig_community,
    use_container_width=True
)


# ============================================================
# SECTION 5 — SENTIMENT PERCENTAGE
# ============================================================

st.markdown(
    '<div class="section-title">📈 Sentiment Percentage by Community</div>',
    unsafe_allow_html=True
)


community_counts = (
    filtered_df
    .groupby("subreddit")
    .size()
    .reset_index(name="Total")
)


sentiment_percentage = (
    filtered_df
    .groupby(
        ["subreddit", "vader_sentiment"]
    )
    .size()
    .reset_index(name="Count")
)


sentiment_percentage = sentiment_percentage.merge(
    community_counts,
    on="subreddit"
)


sentiment_percentage["Percentage"] = (
    sentiment_percentage["Count"]
    / sentiment_percentage["Total"]
    * 100
)


percentage_table = (
    sentiment_percentage
    .pivot(
        index="subreddit",
        columns="vader_sentiment",
        values="Percentage"
    )
    .fillna(0)
)


for sentiment in [
    "Positive",
    "Negative",
    "Neutral"
]:

    if sentiment not in percentage_table.columns:

        percentage_table[sentiment] = 0


percentage_table = percentage_table[
    ["Positive", "Negative", "Neutral"]
].round(2)


st.dataframe(
    percentage_table,
    use_container_width=True
)


# ============================================================
# SECTION 6 — HEATMAP
# ============================================================

st.markdown(
    '<div class="section-title">🔥 Sentiment Percentage Heatmap</div>',
    unsafe_allow_html=True
)


fig_heatmap = px.imshow(
    percentage_table,
    text_auto=".1f",
    aspect="auto",
    title="Community Sentiment Percentage Heatmap",
    labels={
        "x": "Sentiment",
        "y": "Community",
        "color": "Percentage"
    }
)


st.plotly_chart(
    fig_heatmap,
    use_container_width=True
)


# ============================================================
# SECTION 7 — AVERAGE VADER SCORE
# ============================================================

st.markdown(
    '<div class="section-title">📊 Average VADER Compound Score</div>',
    unsafe_allow_html=True
)


avg_scores = (
    filtered_df
    .groupby("subreddit")["vader_compound"]
    .mean()
    .reset_index()
)


avg_scores.columns = [
    "Community",
    "Average VADER Score"
]


avg_scores = avg_scores.sort_values(
    "Average VADER Score",
    ascending=False
)


fig_avg = px.bar(
    avg_scores,
    x="Community",
    y="Average VADER Score",
    text_auto=".2f",
    title="Average Sentiment Score by Community"
)


st.plotly_chart(
    fig_avg,
    use_container_width=True
)


# ============================================================
# SECTION 8 — COMMUNITY MOOD MAP
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Community Mood Map</div>',
    unsafe_allow_html=True
)


mood_map = community_stats[
    [
        "subreddit",
        "Mood_Index",
        "Comments",
        "Positive %",
        "Negative %",
        "Neutral %"
    ]
].copy()


mood_map.columns = [
    "Community",
    "Mood Index",
    "Comments",
    "Positive %",
    "Negative %",
    "Neutral %"
]


mood_map = mood_map.sort_values(
    "Mood Index",
    ascending=False
)


fig_mood = px.scatter(
    mood_map,
    x="Comments",
    y="Mood Index",
    size="Comments",
    hover_name="Community",
    hover_data=[
        "Positive %",
        "Negative %",
        "Neutral %"
    ],
    title="Community Activity vs Mood"
)


fig_mood.add_hline(
    y=50,
    line_dash="dash",
    annotation_text="Neutral midpoint"
)


st.plotly_chart(
    fig_mood,
    use_container_width=True
)


# ============================================================
# SECTION 9 — COMMUNITY INTELLIGENCE TABLE
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Community Intelligence Table</div>',
    unsafe_allow_html=True
)


intelligence_table = community_stats.copy()


intelligence_table = intelligence_table[
    [
        "subreddit",
        "Comments",
        "Mood_Index",
        "Positive %",
        "Negative %",
        "Neutral %",
        "Average_VADER"
    ]
]


intelligence_table.columns = [
    "Community",
    "Comments",
    "Mood Index",
    "Positive %",
    "Negative %",
    "Neutral %",
    "Average VADER"
]


intelligence_table["Mood Index"] = (
    intelligence_table["Mood Index"]
    .round(1)
)


intelligence_table["Positive %"] = (
    intelligence_table["Positive %"]
    .round(1)
)


intelligence_table["Negative %"] = (
    intelligence_table["Negative %"]
    .round(1)
)


intelligence_table["Neutral %"] = (
    intelligence_table["Neutral %"]
    .round(1)
)


intelligence_table["Average VADER"] = (
    intelligence_table["Average VADER"]
    .round(3)
)


st.dataframe(
    intelligence_table,
    use_container_width=True
)


# ============================================================
# SECTION 10 — COMMUNITY COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">⚖️ Community Comparison</div>',
    unsafe_allow_html=True
)


available_communities = sorted(
    df["subreddit"].unique()
)


if len(available_communities) >= 2:

    col1, col2 = st.columns(2)


    with col1:

        community_a = st.selectbox(
            "Community A",
            available_communities,
            index=0
        )


    with col2:

        community_b = st.selectbox(
            "Community B",
            available_communities,
            index=1
        )


    df_a = df[
        df["subreddit"] == community_a
    ]


    df_b = df[
        df["subreddit"] == community_b
    ]


    def sentiment_percentages(data):

        counts = (
            data["vader_sentiment"]
            .value_counts()
        )

        total = len(data)

        if total == 0:

            return {
                "Positive": 0,
                "Negative": 0,
                "Neutral": 0
            }

        return {

            "Positive":
                counts.get("Positive", 0)
                / total * 100,

            "Negative":
                counts.get("Negative", 0)
                / total * 100,

            "Neutral":
                counts.get("Neutral", 0)
                / total * 100
        }


    percent_a = sentiment_percentages(df_a)

    percent_b = sentiment_percentages(df_b)


    comparison_df = pd.DataFrame({

        "Sentiment": [
            "Positive",
            "Negative",
            "Neutral"
        ],

        community_a: [
            percent_a["Positive"],
            percent_a["Negative"],
            percent_a["Neutral"]
        ],

        community_b: [
            percent_b["Positive"],
            percent_b["Negative"],
            percent_b["Neutral"]
        ]

    })


    comparison_df = comparison_df.round(2)


    st.dataframe(
        comparison_df,
        use_container_width=True
    )


    fig_compare = px.bar(
        comparison_df,
        x="Sentiment",
        y=[
            community_a,
            community_b
        ],
        barmode="group",
        title=f"{community_a} vs {community_b}"
    )


    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )


    avg_a = df_a[
        "vader_compound"
    ].mean()


    avg_b = df_b[
        "vader_compound"
    ].mean()


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            f"{community_a} Average Score",
            f"{avg_a:.3f}"
        )


    with col2:

        st.metric(
            f"{community_b} Average Score",
            f"{avg_b:.3f}"
        )


# ============================================================
# SECTION 11 — AUTOMATIC COMMUNITY INSIGHTS
# ============================================================

st.markdown(
    '<div class="section-title">💡 Automatic Community Insights</div>',
    unsafe_allow_html=True
)


insight_data = []


for community, group in df.groupby("subreddit"):

    total = len(group)

    counts = (
        group["vader_sentiment"]
        .value_counts()
    )


    dominant_sentiment = (
        counts.idxmax()
    )


    dominant_percentage = (
        counts.max()
        / total
        * 100
    )


    positive_percentage = (
        counts.get("Positive", 0)
        / total
        * 100
    )


    negative_percentage = (
        counts.get("Negative", 0)
        / total
        * 100
    )


    neutral_percentage = (
        counts.get("Neutral", 0)
        / total
        * 100
    )


    average_score = (
        group["vader_compound"]
        .mean()
    )


    mood = (
        (average_score + 1)
        / 2
        * 100
    )


    insight_data.append({

        "Community": community,

        "Total Comments": total,

        "Dominant Sentiment":
            dominant_sentiment,

        "Dominant %":
            round(dominant_percentage, 2),

        "Positive %":
            round(positive_percentage, 2),

        "Negative %":
            round(negative_percentage, 2),

        "Neutral %":
            round(neutral_percentage, 2),

        "Average VADER Score":
            round(average_score, 3),

        "Mood Index":
            round(mood, 1)

    })


insight_df = pd.DataFrame(
    insight_data
)


st.dataframe(
    insight_df,
    use_container_width=True
)


# ============================================================
# SECTION 12 — SELECTED COMMUNITY INSIGHT
# ============================================================

if selected_community != "All Communities":

    selected_row = insight_df[
        insight_df["Community"]
        == selected_community
    ]


    if not selected_row.empty:

        row = selected_row.iloc[0]


        st.info(
            f"""
### 🔍 Community Insight: r/{selected_community}

**💬 Total comments:** {row["Total Comments"]}

**🎯 Dominant sentiment:** {row["Dominant Sentiment"]}

**😊 Positive:** {row["Positive %"]}%

**😐 Neutral:** {row["Neutral %"]}%

**😞 Negative:** {row["Negative %"]}%

**📊 Average VADER score:** {row["Average VADER Score"]}

**🎯 Community Mood Index:** {row["Mood Index"]}/100
"""
        )


# ============================================================
# SECTION 13 — KEY FINDINGS
# ============================================================

st.markdown(
    '<div class="section-title">📌 Key Findings</div>',
    unsafe_allow_html=True
)


finding1 = (
    f"🔥 **{most_positive['subreddit']}** "
    f"has the highest measured Mood Index "
    f"({most_positive['Mood_Index']:.1f}/100)."
)


finding2 = (
    f"⚠️ **{most_negative['subreddit']}** "
    f"has the lowest measured Mood Index "
    f"({most_negative['Mood_Index']:.1f}/100)."
)


finding3 = (
    f"💬 **{most_active['subreddit']}** "
    f"contains the largest number of analyzed comments "
    f"({int(most_active['Comments'])})."
)


st.markdown(
    f"""
    <div class="finding-card">
    {finding1}
    </div>

    <div class="finding-card">
    {finding2}
    </div>

    <div class="finding-card">
    {finding3}
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SECTION 14 — AUTOMATIC COMMUNITY SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Community Summary</div>',
    unsafe_allow_html=True
)


if selected_community == "All Communities":

    st.write(
        f"""
        CommunityLens analyzed **{len(df):,} comments**
        across **{df['subreddit'].nunique():,} communities**.

        The overall Community Mood Index is
        **{mood_index:.1f}/100**, indicating a
        **{mood_text.lower()}** sentiment environment
        based on the VADER scores in the current dataset.

        The community with the highest measured mood score is
        **{most_positive['subreddit']}**, while the community
        with the lowest measured score is
        **{most_negative['subreddit']}**.

        The most active community by comment volume is
        **{most_active['subreddit']}**.
        """
    )

else:

    selected_group = df[
        df["subreddit"] == selected_community
    ]


    selected_avg = (
        selected_group["vader_compound"]
        .mean()
    )


    selected_mood = (
        (selected_avg + 1)
        / 2
        * 100
    )


    selected_counts = (
        selected_group["vader_sentiment"]
        .value_counts()
    )


    selected_dominant = (
        selected_counts.idxmax()
    )


    st.write(
        f"""
        **r/{selected_community}** contains
        **{len(selected_group):,} analyzed comments**.

        Its current Community Mood Index is
        **{selected_mood:.1f}/100**.

        The dominant sentiment is
        **{selected_dominant}**.

        The average VADER compound score is
        **{selected_avg:.3f}**.
        """
    )


# ============================================================
# SECTION 15 — NEW TEXT SENTIMENT ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">📝 Analyze New Reddit Text</div>',
    unsafe_allow_html=True
)


st.write(
    "Enter a new Reddit comment and CommunityLens "
    "will analyze its sentiment using VADER."
)


analysis_community = st.selectbox(
    "🏘️ Select Community for New Text",
    available_communities,
    key="analysis_community"
)


user_text = st.text_area(
    "💬 Enter Reddit Comment",
    placeholder=(
        "Example: I really enjoyed this community. "
        "Everyone was helpful!"
    )
)


if st.button(
    "🔍 Analyze Sentiment",
    key="analyze_button"
):

    if user_text.strip() == "":

        st.warning(
            "⚠️ Please enter some text."
        )

    else:

        scores = analyzer.polarity_scores(
            user_text
        )


        compound = scores["compound"]


        if compound >= 0.05:

            sentiment = "Positive"

        elif compound <= -0.05:

            sentiment = "Negative"

        else:

            sentiment = "Neutral"


        st.subheader(
            "📋 Analysis Result"
        )


        result_col1, result_col2, result_col3, result_col4 = st.columns(4)


        with result_col1:

            st.metric(
                "Sentiment",
                sentiment
            )


        with result_col2:

            st.metric(
                "Positive",
                f"{scores['pos']:.2f}"
            )


        with result_col3:

            st.metric(
                "Negative",
                f"{scores['neg']:.2f}"
            )


        with result_col4:

            st.metric(
                "Compound",
                f"{compound:.2f}"
            )


        st.write(
            "**Analyzed Text:**"
        )

        st.write(
            user_text
        )


        new_row = {
            "subreddit":
                analysis_community,

            "body":
                user_text,

            "vader_sentiment":
                sentiment,

            "vader_positive":
                scores["pos"],

            "vader_negative":
                scores["neg"],

            "vader_neutral":
                scores["neu"],

            "vader_compound":
                scores["compound"]
        }


        if st.button(
            "💾 Save Analysis",
            key="save_button"
        ):

            updated_df = pd.concat(
                [
                    df,
                    pd.DataFrame([new_row])
                ],
                ignore_index=True
            )


            updated_df.to_csv(
                CSV_FILE,
                index=False
            )


            st.success(
                "✅ New analysis saved successfully!"
            )


            st.rerun()


# ============================================================
# SECTION 16 — RECENT COMMENTS
# ============================================================

st.markdown(
    '<div class="section-title">💬 Recent Reddit Comments</div>',
    unsafe_allow_html=True
)


recent_df = filtered_df[
    [
        "subreddit",
        "body",
        "vader_sentiment",
        "vader_compound"
    ]
].tail(10)


st.dataframe(
    recent_df,
    use_container_width=True
)


# ============================================================
# SECTION 17 — PROJECT PIPELINE
# ============================================================

st.markdown(
    '<div class="section-title">🔄 CommunityLens Pipeline</div>',
    unsafe_allow_html=True
)


pipeline_col1, pipeline_col2, pipeline_col3, pipeline_col4 = st.columns(4)


with pipeline_col1:

    st.info(
        "📥\n\n"
        "**1. Reddit Dataset**\n\n"
        "Collect community conversations"
    )


with pipeline_col2:

    st.info(
        "🧹\n\n"
        "**2. Data Processing**\n\n"
        "Prepare Reddit text and community data"
    )


with pipeline_col3:

    st.info(
        "🧠\n\n"
        "**3. VADER Analysis**\n\n"
        "Calculate sentiment scores"
    )


with pipeline_col4:

    st.info(
        "📊\n\n"
        "**4. Community Insights**\n\n"
        "Visualize and compare communities"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🔍 CommunityLens | Community-Aware Sentiment Intelligence | "
    "VADER + Python + Streamlit"
)