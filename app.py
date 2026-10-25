
import os
import html
import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ============================================================
# COMMUNITYLENS — INTERACTIVE COMMUNITY INTELLIGENCE DASHBOARD
# ============================================================

st.set_page_config(
    page_title="CommunityLens | Intelligence Dashboard",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROCESSED = "data/processed"

DATA_CANDIDATES = [
    os.path.join(PROCESSED, "reddit_vader_sentiment.csv"),
    os.path.join(PROCESSED, "sentiment_data.csv"),
    os.path.join(PROCESSED, "temporal_reddit_data.csv"),
    os.path.join("data", "raw", "reddit_comments.csv"),
]

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
COLOR_MAP = {
    "Positive": "#10b981",
    "Neutral": "#94a3b8",
    "Negative": "#f43f5e",
}

analyzer = SentimentIntensityAnalyzer()

# -------------------- PAGE STYLING ---------------------------

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, rgba(37,99,235,.045), transparent 300px);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101b32, #1b355b);
}
[data-testid="stSidebar"] * { color: #f3f6fc; }

.cl-brand {
    font-size: 1.8rem;
    font-weight: 850;
    letter-spacing: -.7px;
    color: #ffffff;
}
.cl-tag {
    font-size: .72rem;
    letter-spacing: 1.6px;
    color: #bfdbfe;
    font-weight: 750;
    margin-bottom: 10px;
}
.cl-hero {
    background:
        linear-gradient(100deg,
            rgba(16,27,50,.96),
            rgba(37,99,235,.79),
            rgba(15,118,110,.72)),
        url("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1600&q=80")
        center/cover;
    border-radius: 20px;
    padding: 28px;
    min-height: 155px;
    margin: 8px 0 20px 0;
    box-shadow: 0 10px 28px rgba(20,33,61,.14);
}
.cl-hero .eyebrow {
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    color: #bfdbfe;
    font-weight: 800;
}
.cl-hero h1 {
    color: white;
    font-size: 2.1rem;
    margin: 8px 0;
}
.cl-hero p {
    color: #edf4ff;
    margin: 0;
    font-size: 1rem;
}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,145,170,.28);
    padding: 12px 15px;
    border-radius: 13px;
    background: var(--secondary-background-color);
}
div[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(128,145,170,.18);
    border-radius: 13px;
    padding: 5px;
}
.stButton button, .stDownloadButton button {
    border-radius: 10px;
    font-weight: 650;
}
hr { opacity: .25; }

/* Top-level navigation buttons */
div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    padding: 10px;
    border: 1px solid rgba(128,145,170,.25);
    border-radius: 15px;
    background: var(--secondary-background-color);
}
div[data-testid="stRadio"] div[role="radiogroup"] label {
    border: 1px solid rgba(128,145,170,.35);
    border-radius: 10px;
    padding: 7px 12px;
    cursor: pointer;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    border-color: #2563eb;
    background: rgba(37,99,235,.08);
}
div[data-testid="stRadio"] input {
    accent-color: #2563eb;
}
</style>
""", unsafe_allow_html=True)


# -------------------- HELPER FUNCTIONS ----------------------

def show_hero(eyebrow, title, subtitle):
    st.markdown(
        f'<div class="cl-hero">'
        f'<div class="eyebrow">{html.escape(str(eyebrow))}</div>'
        f'<h1>{html.escape(str(title))}</h1>'
        f'<p>{html.escape(str(subtitle))}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def load_research(filename):
    path = os.path.join(PROCESSED, filename)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def sentiment_counts(data):
    return data["vader_sentiment"].value_counts().reindex(
        SENTIMENT_ORDER, fill_value=0
    )


def sentiment_percentages(data):
    if data.empty:
        return {s: 0.0 for s in SENTIMENT_ORDER}
    counts = sentiment_counts(data)
    return {s: counts[s] / len(data) * 100 for s in SENTIMENT_ORDER}


def mood_index(data):
    if data.empty:
        return 50.0
    return float((data["vader_compound"].mean() + 1) * 50)


def download_csv(data, filename, label="⬇️ Download CSV"):
    st.download_button(
        label,
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )


# -------------------- LOAD THE DATASET ----------------------

dataset_path = next(
    (p for p in DATA_CANDIDATES if os.path.exists(p)),
    None,
)

if dataset_path is None:
    st.error(
        "Could not find a usable dataset. Check that your CommunityLens "
        "project contains the original dataset or a processed sentiment CSV."
    )
    st.info(
        "Expected files include data/processed/sentiment_data.csv or "
        "data/processed/temporal_reddit_data.csv."
    )
    st.stop()

try:
    df = pd.read_csv(dataset_path)
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

if "subreddit" not in df.columns:
    st.error("The dataset must contain a 'subreddit' column.")
    st.stop()

if "body" not in df.columns:
    text_column = next(
        (c for c in ["text", "comment", "content"] if c in df.columns),
        None,
    )
    if text_column:
        df["body"] = df[text_column]
    else:
        st.error("No comment text column was found.")
        st.stop()

df["subreddit"] = df["subreddit"].fillna("Unknown").astype(str)
df["body"] = df["body"].fillna("").astype(str)
df = df[df["body"].str.strip().ne("")].copy()

if df.empty:
    st.error("The loaded dataset contains no non-empty comments.")
    st.stop()

# Calculate VADER scores if required columns are missing.
if "vader_compound" not in df.columns:
    if "compound" in df.columns:
        df["vader_compound"] = pd.to_numeric(
            df["compound"], errors="coerce"
        )
    else:
        scores = df["body"].map(analyzer.polarity_scores)
        df["vader_compound"] = scores.map(lambda x: x["compound"])

df["vader_compound"] = pd.to_numeric(
    df["vader_compound"], errors="coerce"
).fillna(0).clip(-1, 1)

for target, short in [
    ("vader_positive", "pos"),
    ("vader_negative", "neg"),
    ("vader_neutral", "neu"),
]:
    if target not in df.columns:
        if short in df.columns:
            df[target] = pd.to_numeric(
                df[short], errors="coerce"
            ).fillna(0)
        else:
            scores = df["body"].map(analyzer.polarity_scores)
            df[target] = scores.map(lambda x: x[short])

if "vader_sentiment" not in df.columns:
    df["vader_sentiment"] = df["vader_compound"].apply(
        lambda x: (
            "Positive" if x >= .05
            else "Negative" if x <= -.05
            else "Neutral"
        )
    )

df["vader_sentiment"] = (
    df["vader_sentiment"].fillna("Neutral").astype(str).str.title()
)
df.loc[
    ~df["vader_sentiment"].isin(SENTIMENT_ORDER),
    "vader_sentiment",
] = "Neutral"

communities = sorted(df["subreddit"].unique().tolist())

community_stats = df.groupby("subreddit").agg(
    Comments=("body", "size"),
    Average_VADER=("vader_compound", "mean"),
).reset_index()

community_stats["Mood_Index"] = (
    (community_stats["Average_VADER"] + 1) * 50
).clip(0, 100)

for sentiment in SENTIMENT_ORDER:
    counts = df.groupby("subreddit")["vader_sentiment"].apply(
        lambda x: (x == sentiment).sum()
    )
    totals = df.groupby("subreddit").size()
    community_stats[f"{sentiment} %"] = (
        community_stats["subreddit"].map(counts)
        / community_stats["subreddit"].map(totals)
        * 100
    )


# -------------------- SIDEBAR FILTERS -----------------------

with st.sidebar:
    st.markdown(
        '<div class="cl-brand">🔎 CommunityLens</div>'
        '<div class="cl-tag">COMMUNITY INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Dataset status")
    st.metric("Comments", f"{len(df):,}")
    st.metric("Communities", f"{df['subreddit'].nunique():,}")
    st.caption(f"Loaded: {os.path.basename(dataset_path)}")
    st.markdown("---")
    sidebar_community = st.selectbox(
        "Filter overview by community",
        ["All Communities"] + communities,
    )
    st.download_button(
        "⬇️ Download loaded dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="communitylens_vader_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(
        "Community mood is a rescaled average VADER compound score, "
        "not a validated measure of wellbeing."
    )

global_df = (
    df.copy()
    if sidebar_community == "All Communities"
    else df[df["subreddit"] == sidebar_community].copy()
)


# -------------------- TOP NAVIGATION ------------------------

st.markdown("### 🔎 CommunityLens")
st.caption("COMMUNITY-AWARE SENTIMENT AND RESEARCH INTELLIGENCE")

page = st.radio(
    "Main navigation",
    [
        "🏠 Overview",
        "😊 Sentiment Explorer",
        "🏘️ Community Insights",
        "⚖️ Compare Communities",
        "📝 Text Analyzer",
        "💬 Recent Comments",
        "🔬 Research Dashboard",
        "ℹ️ About Project",
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation",
)

st.markdown("---")


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":
    show_hero(
        "COMMUNITY PULSE / OVERVIEW",
        "Understand the conversation.",
        "Explore sentiment patterns, community mood and discussion activity.",
    )

    pct = sentiment_percentages(global_df)
    most_active = community_stats.loc[community_stats["Comments"].idxmax()]
    highest_mood = community_stats.loc[community_stats["Mood_Index"].idxmax()]
    lowest_mood = community_stats.loc[community_stats["Mood_Index"].idxmin()]

    a, b, c, d = st.columns(4)
    a.metric("Comments analysed", f"{len(global_df):,}")
    b.metric("Communities", f"{global_df['subreddit'].nunique():,}")
    c.metric("Positive comments", f"{pct['Positive']:.1f}%")
    d.metric("Community Mood Index", f"{mood_index(global_df):.1f}/100")

    st.markdown("### 📊 Sentiment at a glance")
    left, right = st.columns([1, 1.4])

    counts = sentiment_counts(global_df).rename_axis(
        "Sentiment"
    ).reset_index(name="Comments")

    with left:
        fig = px.pie(
            counts,
            names="Sentiment",
            values="Comments",
            hole=.6,
            color="Sentiment",
            color_discrete_map=COLOR_MAP,
            title="Sentiment distribution",
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        ranked = global_df.groupby("subreddit").agg(
            Comments=("body", "size"),
            Mood=("vader_compound", "mean"),
        ).reset_index().nlargest(12, "Comments").sort_values("Comments")

        fig = px.bar(
            ranked,
            x="Comments",
            y="subreddit",
            orientation="h",
            color="Mood",
            color_continuous_scale="RdYlGn",
            range_color=[-1, 1],
            title="Most active communities",
        )
        fig.update_layout(yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ✨ Quick insights")
    a, b, c = st.columns(3)
    a.info(
        f"**Most active community**\n\n"
        f"r/{most_active['subreddit']}\n\n"
        f"{int(most_active['Comments']):,} comments"
    )
    b.success(
        f"**Highest measured mood**\n\n"
        f"r/{highest_mood['subreddit']}\n\n"
        f"{highest_mood['Mood_Index']:.1f}/100"
    )
    c.warning(
        f"**Lowest measured mood**\n\n"
        f"r/{lowest_mood['subreddit']}\n\n"
        f"{lowest_mood['Mood_Index']:.1f}/100"
    )

    st.markdown("### 🧭 How CommunityLens works")
    steps = st.columns(4)
    workflow = [
        ("01", "Reddit data", "Load and prepare comment records"),
        ("02", "VADER", "Calculate sentiment scores"),
        ("03", "Community analysis", "Compare groups and interactions"),
        ("04", "Visual insights", "Explore results through charts"),
    ]
    for col, (number, title, description) in zip(steps, workflow):
        with col:
            st.info(f"**{number} · {title}**\n\n{description}")


# ============================================================
# PAGE 2 — SENTIMENT EXPLORER
# ============================================================

elif page == "😊 Sentiment Explorer":
    show_hero(
        "EXPLORE / SENTIMENT",
        "Sentiment Explorer",
        "Filter comments, inspect sentiment scores and export results.",
    )

    a, b, c = st.columns([1, 1, 1.4])
    with a:
        community_filter = st.selectbox(
            "Community", ["All Communities"] + communities
        )
    with b:
        selected_sentiments = st.multiselect(
            "Sentiment",
            SENTIMENT_ORDER,
            default=SENTIMENT_ORDER,
        )
    with c:
        keyword = st.text_input(
            "Search comment text", placeholder="Enter a keyword..."
        )

    view = df.copy()
    if community_filter != "All Communities":
        view = view[view["subreddit"] == community_filter]
    view = view[view["vader_sentiment"].isin(selected_sentiments)]

    if keyword.strip():
        view = view[
            view["body"].str.contains(
                keyword.strip(), case=False, na=False, regex=False
            )
        ]

    a, b, c = st.columns(3)
    a.metric("Matching comments", f"{len(view):,}")
    b.metric(
        "Average compound",
        f"{view['vader_compound'].mean():.3f}" if not view.empty else "—",
    )
    c.metric(
        "Positive share",
        f"{(view['vader_sentiment'].eq('Positive').mean() * 100):.1f}%"
        if not view.empty else "—",
    )

    if not view.empty:
        counts = sentiment_counts(view).rename_axis(
            "Sentiment"
        ).reset_index(name="Comments")

        fig = px.bar(
            counts,
            x="Sentiment",
            y="Comments",
            color="Sentiment",
            color_discrete_map=COLOR_MAP,
            title="Filtered sentiment distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            view[
                [
                    "subreddit", "body", "vader_sentiment",
                    "vader_positive", "vader_neutral",
                    "vader_negative", "vader_compound",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        download_csv(view, "communitylens_filtered_comments.csv")
    else:
        st.info("No comments match these filters. Change your search criteria.")


# ============================================================
# PAGE 3 — COMMUNITY INSIGHTS
# ============================================================

elif page == "🏘️ Community Insights":
    show_hero(
        "EXPLORE / COMMUNITIES",
        "Community Insights",
        "Explore activity, sentiment percentages and measured community mood.",
    )

    top_n = st.slider(
        "Communities to display",
        min_value=1,
        max_value=max(1, min(30, len(community_stats))),
        value=max(1, min(12, len(community_stats))),
    )
    sort_by = st.selectbox(
        "Rank communities by",
        ["Comments", "Mood_Index", "Average_VADER"],
    )
    ranked = community_stats.sort_values(
        sort_by, ascending=False
    ).head(top_n)

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            ranked.sort_values("Comments"),
            x="Comments",
            y="subreddit",
            orientation="h",
            title="Community activity",
        )
        fig.update_layout(yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.bar(
            ranked.sort_values("Mood_Index"),
            x="Mood_Index",
            y="subreddit",
            orientation="h",
            color="Mood_Index",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            title="Community Mood Index",
        )
        fig.add_vline(x=50, line_dash="dash")
        fig.update_layout(yaxis_title="", xaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        community_stats,
        x="Comments",
        y="Mood_Index",
        size="Comments",
        color="Average_VADER",
        color_continuous_scale="RdYlGn",
        range_color=[-1, 1],
        hover_name="subreddit",
        title="Community activity versus mood",
    )
    fig.add_hline(y=50, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Community sentiment summary")
    summary = community_stats[
        [
            "subreddit", "Comments", "Positive %", "Neutral %",
            "Negative %", "Average_VADER", "Mood_Index",
        ]
    ].round(3)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    download_csv(summary, "communitylens_community_summary.csv")


# ============================================================
# PAGE 4 — COMPARE COMMUNITIES
# ============================================================

elif page == "⚖️ Compare Communities":
    show_hero(
        "EXPLORE / COMPARISON",
        "Compare Communities",
        "Compare two subreddits by activity, sentiment mix and compound scores.",
    )

    if len(communities) < 2:
        st.warning("At least two communities are required.")
    else:
        a, b = st.columns(2)
        community_a = a.selectbox("Community A", communities, index=0)
        community_b = b.selectbox("Community B", communities, index=1)

        if community_a == community_b:
            st.warning("Choose two different communities.")
        else:
            data_a = df[df["subreddit"] == community_a]
            data_b = df[df["subreddit"] == community_b]
            p_a = sentiment_percentages(data_a)
            p_b = sentiment_percentages(data_b)

            a, b, c, d = st.columns(4)
            a.metric(f"{community_a} comments", f"{len(data_a):,}")
            b.metric(f"{community_b} comments", f"{len(data_b):,}")
            c.metric(f"{community_a} mood", f"{mood_index(data_a):.1f}/100")
            d.metric(f"{community_b} mood", f"{mood_index(data_b):.1f}/100")

            comparison = pd.DataFrame({
                "Sentiment": SENTIMENT_ORDER,
                community_a: [p_a[s] for s in SENTIMENT_ORDER],
                community_b: [p_b[s] for s in SENTIMENT_ORDER],
            })

            fig = px.bar(
                comparison,
                x="Sentiment",
                y=[community_a, community_b],
                barmode="group",
                title="Sentiment percentage comparison",
                color_discrete_sequence=["#2563eb", "#0f766e"],
            )
            fig.update_layout(yaxis_title="Percentage")
            st.plotly_chart(fig, use_container_width=True)

            compound = pd.DataFrame({
                "Community": [community_a, community_b],
                "Average compound": [
                    data_a["vader_compound"].mean(),
                    data_b["vader_compound"].mean(),
                ],
            })
            fig = px.bar(
                compound,
                x="Community",
                y="Average compound",
                color="Average compound",
                color_continuous_scale="RdYlGn",
                range_color=[-1, 1],
                title="Average VADER compound score",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(comparison.round(2), hide_index=True)
            download_csv(comparison, "communitylens_comparison.csv")


# ============================================================
# PAGE 5 — LIVE TEXT ANALYZER
# ============================================================

elif page == "📝 Text Analyzer":
    show_hero(
        "TOOLS / LIVE ANALYSIS",
        "Text Sentiment Analyzer",
        "Enter a comment to calculate sentiment using VADER.",
    )

    text_input = st.text_area(
        "Comment text",
        placeholder="Example: The discussion was helpful and informative!",
        height=150,
    )

    if st.button("🔍 Analyse sentiment", type="primary"):
        if not text_input.strip():
            st.warning("Enter some text first.")
        else:
            scores = analyzer.polarity_scores(text_input.strip())
            compound = scores["compound"]
            label = (
                "Positive" if compound >= .05
                else "Negative" if compound <= -.05
                else "Neutral"
            )
            st.session_state["last_analysis"] = {
                "text": text_input.strip(),
                "scores": scores,
                "label": label,
            }

    result = st.session_state.get("last_analysis")

    if result:
        scores = result["scores"]
        st.markdown("### Analysis result")
        a, b, c, d = st.columns(4)
        a.metric("Predicted sentiment", result["label"])
        b.metric("Positive score", f"{scores['pos']:.3f}")
        c.metric("Neutral score", f"{scores['neu']:.3f}")
        d.metric("Compound score", f"{scores['compound']:.3f}")

        score_df = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Score": [scores["pos"], scores["neu"], scores["neg"]],
        })
        fig = px.bar(
            score_df,
            x="Sentiment",
            y="Score",
            color="Sentiment",
            color_discrete_map=COLOR_MAP,
            range_y=[0, 1],
            title="VADER score breakdown",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Analysed text**")
        st.write(result["text"])

        analysis_csv = pd.DataFrame([{
            "body": result["text"],
            "vader_sentiment": result["label"],
            "vader_positive": scores["pos"],
            "vader_neutral": scores["neu"],
            "vader_negative": scores["neg"],
            "vader_compound": scores["compound"],
        }])
        download_csv(analysis_csv, "communitylens_text_analysis.csv")


# ============================================================
# PAGE 6 — RECENT COMMENTS
# ============================================================

elif page == "💬 Recent Comments":
    show_hero(
        "EXPLORE / COMMENTS",
        "Comment Browser",
        "Search, filter, inspect and export comments from your dataset.",
    )

    a, b = st.columns([1, 2])
    chosen_sentiments = a.multiselect(
        "Sentiment type",
        SENTIMENT_ORDER,
        default=SENTIMENT_ORDER,
    )
    search_term = b.text_input(
        "Search comment content",
        placeholder="Search for a keyword...",
    )

    comments_view = global_df[
        global_df["vader_sentiment"].isin(chosen_sentiments)
    ].copy()

    if search_term.strip():
        comments_view = comments_view[
            comments_view["body"].str.contains(
                search_term.strip(), case=False, na=False, regex=False
            )
        ]

    st.caption(f"{len(comments_view):,} matching records")
    st.dataframe(
        comments_view[
            ["subreddit", "body", "vader_sentiment", "vader_compound"]
        ].tail(200),
        use_container_width=True,
        hide_index=True,
    )
    download_csv(comments_view, "communitylens_comments.csv")


# ============================================================
# PAGE 7 — RESEARCH DASHBOARD
# ============================================================

elif page == "🔬 Research Dashboard":
    show_hero(
        "RESEARCH / EVENT-TRIGGERED DYNAMIC COMMUNITIES",
        "Research Dashboard",
        "Explore temporal patterns, detected changes, interaction structure and evaluation.",
    )

    research_page = st.radio(
        "Research sections",
        [
            "Research Overview",
            "Temporal Analysis",
            "Event Detection",
            "Interaction Graph",
            "Community Evolution",
            "Fixed vs Event-Triggered",
            "Evaluation Metrics",
            "Sentiment Anomalies",
        ],
        horizontal=True,
        key="research_navigation",
    )
    st.markdown("---")

    change_df = load_research("change_detection.csv")
    event_df = load_research("event_triggered_communities.csv")
    evolution_df = load_research("community_evolution.csv")
    comparison_df = load_research("fixed_vs_event.csv")
    evaluation_df = load_research("evaluation_metrics.csv")
    anomaly_df = load_research("sentiment_anomalies.csv")
    daily_activity_df = load_research("daily_activity.csv")
    daily_sentiment_df = load_research("daily_sentiment.csv")
    interaction_edges_df = load_research("interaction_edges.csv")
    interaction_nodes_df = load_research("interaction_nodes.csv")

    # Research overview
    if research_page == "Research Overview":
        st.markdown("### Research objective")
        st.write(
            "Identify significant changes in community activity, sentiment "
            "and interaction structure, then examine selective updates."
        )

        a, b, c, d = st.columns(4)
        a.metric(
            "Detected changes",
            len(change_df) if change_df is not None else "Unavailable",
        )
        b.metric("Fixed-interval updates", "11")
        c.metric("Event-triggered updates", "4")
        d.metric("Update reduction", "63.64%")

        a, b, c = st.columns(3)
        a.metric("Fixed computation", "1.2743 sec")
        b.metric("Event computation", "0.4453 sec")
        c.metric("Computation saved", "65.06%")

        st.markdown("### Research workflow")
        workflow = [
            "Reddit data", "Temporal analysis", "VADER sentiment",
            "Interaction graph", "Change detection", "Event trigger",
            "Community update", "Community evolution",
            "Fixed vs event", "Evaluation",
        ]
        for start in range(0, len(workflow), 5):
            cols = st.columns(5)
            for col, item in zip(cols, workflow[start:start + 5]):
                col.info(item)

        if change_df is not None:
            st.markdown("### Detected changes")
            st.dataframe(change_df, use_container_width=True, hide_index=True)

        st.caption(
            "The summary values reflect the currently recorded experiment. "
            "The event-capture result has not been independently ground-truth validated."
        )

    # Temporal analysis
    elif research_page == "Temporal Analysis":
        st.markdown("### Daily activity")
        if daily_activity_df is not None:
            if "date" in daily_activity_df.columns:
                numeric = daily_activity_df.select_dtypes(
                    include="number"
                ).columns.tolist()
                if numeric:
                    metric = st.selectbox("Activity metric", numeric)
                    fig = px.line(
                        daily_activity_df,
                        x="date",
                        y=metric,
                        markers=True,
                        title=f"Daily {metric}",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                daily_activity_df, use_container_width=True, hide_index=True
            )
        else:
            st.warning("daily_activity.csv was not found.")

        st.markdown("### Daily sentiment")
        if daily_sentiment_df is not None:
            if "date" in daily_sentiment_df.columns:
                numeric = daily_sentiment_df.select_dtypes(
                    include="number"
                ).columns.tolist()
                if numeric:
                    metric = st.selectbox(
                        "Sentiment metric", numeric, key="daily_sentiment_metric"
                    )
                    fig = px.line(
                        daily_sentiment_df,
                        x="date",
                        y=metric,
                        markers=True,
                        title=f"Daily {metric}",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                daily_sentiment_df, use_container_width=True, hide_index=True
            )
        else:
            st.warning("daily_sentiment.csv was not found.")

    # Event detection
    elif research_page == "Event Detection":
        st.write(
            "Events are flagged when the measured change crosses the "
            "threshold used by the current detection procedure."
        )
        if change_df is not None:
            st.metric("Detected event records", len(change_df))
            if "date" in change_df.columns and "score" in change_df.columns:
                color_col = (
                    "event_type" if "event_type" in change_df.columns else None
                )
                fig = px.scatter(
                    change_df,
                    x="date",
                    y="score",
                    color=color_col,
                    title="Change detection timeline",
                )
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(change_df, use_container_width=True, hide_index=True)
            download_csv(change_df, "communitylens_events.csv")
        else:
            st.warning("change_detection.csv was not found.")

    # Interaction graph
    elif research_page == "Interaction Graph":
        a, b = st.columns(2)
        a.metric(
            "Edge records",
            len(interaction_edges_df)
            if interaction_edges_df is not None else "Unavailable",
        )
        b.metric(
            "Node records",
            len(interaction_nodes_df)
            if interaction_nodes_df is not None else "Unavailable",
        )

        if interaction_edges_df is not None:
            st.markdown("### Interaction edges")
            st.dataframe(
                interaction_edges_df.head(200),
                use_container_width=True,
                hide_index=True,
            )
            download_csv(interaction_edges_df, "interaction_edges.csv")
        else:
            st.warning("interaction_edges.csv was not found.")

        if interaction_nodes_df is not None:
            st.markdown("### Interaction nodes")
            st.dataframe(
                interaction_nodes_df.head(200),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "The current graph uses identifiable comment relationships; "
            "it should not be interpreted as a complete user-to-user network."
        )

    # Community evolution
    elif research_page == "Community Evolution":
        if evolution_df is not None:
            st.metric("Evolution records", len(evolution_df))
            type_col = next(
                (
                    c for c in [
                        "event_type", "event", "change_type", "type"
                    ] if c in evolution_df.columns
                ),
                None,
            )
            if type_col:
                counts = evolution_df[type_col].value_counts().rename_axis(
                    "Evolution type"
                ).reset_index(name="Records")
                fig = px.bar(
                    counts,
                    x="Evolution type",
                    y="Records",
                    title="Community evolution records",
                )
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(evolution_df, use_container_width=True, hide_index=True)
            download_csv(evolution_df, "communitylens_evolution.csv")
        else:
            st.warning("community_evolution.csv was not found.")

    # Fixed versus event-triggered
    elif research_page == "Fixed vs Event-Triggered":
        a, b, c = st.columns(3)
        a.metric("Fixed updates", "11")
        b.metric("Event-triggered updates", "4")
        c.metric("Update reduction", "63.64%")

        a, b, c = st.columns(3)
        a.metric("Fixed runtime", "1.2743 sec")
        b.metric("Event runtime", "0.4453 sec")
        c.metric("Runtime reduction", "65.06%")

        if comparison_df is not None:
            st.dataframe(
                comparison_df, use_container_width=True, hide_index=True
            )
            download_csv(comparison_df, "fixed_vs_event.csv")
        else:
            st.info(
                "fixed_vs_event.csv was not found. Summary values above "
                "reflect the currently recorded experiment."
            )

        st.caption(
            "Execution-time results depend on the dataset, implementation "
            "and execution environment."
        )

    # Evaluation
    elif research_page == "Evaluation Metrics":
        if evaluation_df is not None:
            st.markdown("### Saved evaluation output")
            st.dataframe(
                evaluation_df, use_container_width=True, hide_index=True
            )
            download_csv(evaluation_df, "evaluation_metrics.csv")
        else:
            st.warning("evaluation_metrics.csv was not found.")

        a, b, c = st.columns(3)
        a.metric("Average community stability", "0.676233")
        b.metric("Events detected / captured", "4 / 4")
        c.metric("Recorded event capture", "100%")

        st.info(
            "Interpret the 100% event-capture result cautiously: the current "
            "evaluation reuses the same detection workflow and is not "
            "independent ground-truth validation."
        )

    # Sentiment anomalies
    elif research_page == "Sentiment Anomalies":
        if anomaly_df is not None:
            st.metric("Anomaly records", len(anomaly_df))
            if "date" in anomaly_df.columns:
                numeric = anomaly_df.select_dtypes(
                    include="number"
                ).columns.tolist()
                if numeric:
                    metric = st.selectbox("Numeric field", numeric)
                    fig = px.scatter(
                        anomaly_df,
                        x="date",
                        y=metric,
                        title=f"Anomaly view: {metric}",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
            download_csv(anomaly_df, "sentiment_anomalies.csv")
        else:
            st.warning("sentiment_anomalies.csv was not found.")


# ============================================================
# PAGE 8 — ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":
    show_hero(
        "ABOUT / COMMUNITYLENS",
        "Turning comments into community insights.",
        "An interactive research and visualization project for Reddit discussions.",
    )

    st.markdown("### What this dashboard does")
    st.write(
        "CommunityLens applies VADER sentiment analysis to Reddit comments, "
        "aggregates sentiment by subreddit, and presents activity, interaction "
        "and saved event-triggered research results through an interactive interface."
    )

    st.markdown("### Main features")
    features = {
        "Sentiment analysis": (
            "Positive, neutral, negative and compound sentiment scores."
        ),
        "Community mood": (
            "A 0–100 rescaling of the mean VADER compound score; "
            "not a validated wellbeing measure."
        ),
        "Community comparison": (
            "Compare sentiment proportions and compound scores between communities."
        ),
        "Research outputs": (
            "Explore temporal, event-detection, graph, evolution, comparison "
            "and anomaly CSV files."
        ),
        "Exports": (
            "Download filtered comments and analysis tables as CSV files."
        ),
    }
    for title, description in features.items():
        with st.expander(title):
            st.write(description)

    st.markdown("### Method limitations")
    st.write(
        "VADER may misinterpret sarcasm, context, slang and domain-specific "
        "language. Detected changes do not establish the cause of an event. "
        "Independent validation is needed before interpreting the results "
        "as confirmed real-world events."
    )

    st.markdown("### Technology stack")
    st.write(
        "Python · Pandas · VADER · Plotly · Streamlit · CSV data"
    )

st.markdown("---")
st.caption(
    "CommunityLens · Community-Aware Sentiment Intelligence · "
    "Interactive research dashboard"
)
