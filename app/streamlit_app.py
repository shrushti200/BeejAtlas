
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ==================================================
# 1. APP CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="BeejAtlas | Seed Research Explorer",

    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "dry_bean_clean.csv"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"


# ==================================================
# 2. DATA LOADING
# ==================================================

@st.cache_data
def load_csv(file_path):

    path = Path(file_path)

    if not path.exists():
        return None

    return pd.read_csv(path)

@st.cache_data(show_spinner="Loading the UCI Dry Bean dataset...")
def load_dataset(file_path):

    path = Path(file_path)

    # Use the existing local cleaned dataset when available.
    if path.exists():
        return pd.read_csv(path)

    # Otherwise retrieve the public UCI dataset.
    try:
        from ucimlrepo import fetch_ucirepo

        st.info(
            "Retrieving the UCI Dry Bean benchmark dataset..."
        )

        dry_bean = fetch_ucirepo(id=602)

        features = dry_bean.data.features.copy()
        targets = dry_bean.data.targets

        dataset = features.copy()

        dataset["Class"] = targets.iloc[:, 0].to_numpy()

        dataset = (
            dataset
            .drop_duplicates(keep="first")
            .reset_index(drop=True)
        )

        return dataset

    except Exception as exc:

        st.error(
            "Unable to retrieve the UCI Dry Bean dataset. "
            "Please check the data source connection."
        )

        st.caption(str(exc))

        return None


data = load_dataset(DATA_FILE)

model_results = load_csv(
    TABLES / "model_comparison.csv"
)

cv_results = load_csv(
    TABLES / "cross_validation_results.csv"
)

repeated_cv_results = load_csv(
    TABLES / "repeated_cv_summary.csv"
)

permutation_results = load_csv(
    TABLES / "permutation_importance.csv"
)

grouped_results = load_csv(
    TABLES / "grouped_permutation_importance.csv"
)

feature_groups = load_csv(
    TABLES / "correlated_feature_groups.csv"
)

builtin_importance = load_csv(
    TABLES / "random_forest_feature_importance.csv"
)


# ==================================================
# 3. STYLING
# ==================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 750;
        color: #276749;
        margin-bottom: 0;
    }

    .subtitle {
        color: #6B7280;
        font-size: 16px;
        margin-top: 4px;
    }

    .section-title {
        color: #276749;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# 3A. BEEJATLAS VISUAL DESIGN SYSTEM
# ==================================================

st.markdown(
    """
    <style>

    :root {
        --ba-forest: #18392F;
        --ba-green: #315C4B;
        --ba-sage: #8DAA91;
        --ba-sage-light: #E8EFE8;
        --ba-cream: #F6F7F3;
        --ba-white: #FFFFFF;
        --ba-ink: #26352F;
        --ba-muted: #6D7972;
        --ba-border: #E1E7E0;
    }

    /* Main application background */

    .stApp,
    [data-testid="stAppViewContainer"] {
        background-color: var(--ba-cream);
        color: var(--ba-ink);
    }

    [data-testid="stHeader"] {
        background-color: rgba(246, 247, 243, 0.9);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1320px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Typography */

    .main-title {
        font-size: clamp(2.4rem, 4vw, 3.2rem);
        font-weight: 750;
        letter-spacing: -1.8px;
        color: var(--ba-forest);
        margin-bottom: 0.2rem;
        line-height: 1.15;
    }

    .subtitle {
        color: var(--ba-muted);
        font-size: 1rem;
        line-height: 1.7;
        margin-top: 0.3rem;
    }

    h1, h2, h3 {
        color: var(--ba-forest) !important;
        letter-spacing: -0.5px;
    }

    h2 {
        font-weight: 700 !important;
    }

    h3 {
        font-weight: 650 !important;
    }

    p, li {
        color: var(--ba-ink);
        line-height: 1.7;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background-color: #EDF2EC;
        border-right: 1px solid var(--ba-border);
    }

    section[data-testid="stSidebar"] h1 {
        color: var(--ba-forest) !important;
        font-size: 1.7rem;
        letter-spacing: -0.8px;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--ba-muted);
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 10px;
        padding: 0.35rem 0.55rem;
        transition: background-color 0.2s ease;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background-color: #DCE8DE;
        border: 1px solid #C9D9CB;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #E3EBE2;
    }

    /* Dataset metric cards */

    div[data-testid="stMetric"] {
        background-color: var(--ba-white);
        border: 1px solid var(--ba-border);
        border-radius: 15px;
        padding: 18px 20px;
        box-shadow: 0 3px 12px rgba(24, 57, 47, 0.045);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--ba-muted);
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: var(--ba-forest);
        font-weight: 750;
    }

    /* Charts */

    div[data-testid="stPlotlyChart"] {
        background-color: var(--ba-white);
        border: 1px solid var(--ba-border);
        border-radius: 15px;
        padding: 8px;
        box-shadow: 0 3px 12px rgba(24, 57, 47, 0.035);
    }

    /* Data tables */

    [data-testid="stDataFrame"] {
        border: 1px solid var(--ba-border);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Buttons */

    .stButton > button,
    [data-testid="stDownloadButton"] button,
    .stLinkButton > a {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #C9D8CB;
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    [data-testid="stDownloadButton"] button:hover {
        border-color: var(--ba-green);
        color: var(--ba-green);
    }

    /* Alerts and information panels */

    [data-testid="stAlert"] {
        border-radius: 13px;
        border: 1px solid var(--ba-border);
    }

    /* Inputs */

    input, textarea {
        border-radius: 9px !important;
    }

    /* Dividers */

    hr {
        border-color: var(--ba-border) !important;
    }

    /* Responsive layout */

    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.1rem 1rem 2rem;
        }

        .main-title {
            font-size: 2.3rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Consistent colours and styling for every Plotly Express chart.

px.defaults.template = "plotly_white"

px.defaults.color_discrete_sequence = [
    "#315C4B",
    "#8DAA91",
    "#C5A77C",
    "#71899A",
    "#B87967",
    "#9A8BAA",
    "#8A957A"
]

px.defaults.color_continuous_scale = [
    "#EDF3ED",
    "#A7C1A8",
    "#315C4B"
]


# ==================================================
# 4. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.markdown("#  BeejAtlas")

st.sidebar.caption(
    "Exploring seed diversity through Data Science"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "Seed Data Explorer",
        "Machine Learning Lab",
        "Feature Importance",
        "Research & Limitations",
        "Indian Seed Conservation"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Research project inspired by seed conservation "
    "and traditional agricultural biodiversity."
)


# ==================================================
# 5. HEADER
# ==================================================

st.markdown(
    '<p class="main-title">BeejAtlas</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'An interactive exploration of seed characteristics, '
    'data analysis and machine learning.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# ==================================================
# 6. HELPER FUNCTIONS
# ==================================================

def show_existing_image(filename, caption=None):

    image_path = FIGURES / filename

    if image_path.exists():
        st.image(
            str(image_path),
            caption=caption,
            use_container_width=True
        )
    else:
        st.info(f"Chart not available yet: {filename}")

def show_csv_report(filename):

    report = load_csv(TABLES / filename)

    if report is not None:

        if "Unnamed: 0" in report.columns:
            report = report.rename(
                columns={"Unnamed: 0": "Category"}
            )

        st.dataframe(
            report,
            hide_index=True,
            use_container_width=True
        )

    else:
        st.info(
            f"Report not available yet: {filename}"
        )

# ==================================================
# 7. PROJECT OVERVIEW
# ==================================================

if page == "Project Overview":

    # HERO SECTION

    with st.container(border=True):

        st.caption("INDEPENDENT RESEARCH PROJECT")

        st.title(
            "Understanding seed diversity through data science."
        )

        st.write(
            "An exploration of agricultural biodiversity "
            "through data analysis, statistical evaluation "
            "and machine learning."
        )

        st.caption(
            "DATA EXPLORATION  |  MACHINE LEARNING  |  RESEARCH"
        )


    st.write("")

    # DATASET METRICS

    if data is not None:

        st.markdown("### Dataset overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Observations",
            f"{len(data):,}"
        )

        c2.metric(
            "Numerical features",
            len(data.select_dtypes(include="number").columns)
        )

        c3.metric(
            "Bean categories",
            data["Class"].nunique()
        )

        c4.metric(
            "Exact duplicates",
            data.duplicated().sum()
        )

        st.write("")

        # CATEGORY DISTRIBUTION

        left, right = st.columns([1.45, 0.85])

        with left:

            st.markdown("### Category distribution")

            counts = (
                data["Class"]
                .value_counts()
                .rename_axis("Category")
                .reset_index(name="Observations")
                .sort_values("Category")
            )

            fig = px.bar(
                counts,
                x="Category",
                y="Observations",
                color="Category",
                color_discrete_sequence=[
                    "#315C4B",
                    "#78917C",
                    "#C5A77C",
                    "#8297A1",
                    "#B77968",
                    "#9684A3",
                    "#9DAA83"
                ]
            )

            fig.update_layout(
                template="plotly_white",
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=15, b=10),
                font=dict(
                    family="Arial",
                    color="#26352F"
                ),
                xaxis_title=None,
                yaxis_title="Observations"
            )

            fig.update_xaxes(showgrid=False)

            fig.update_yaxes(
                gridcolor="#E5EAE5",
                zeroline=False
            )

            fig.update_traces(
                marker_line_width=0
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False}
            )

        # RESEARCH SNAPSHOT

        with right:

            st.markdown("### Research snapshot")
            st.markdown("**UCI Dry Bean**")

            st.write(
                "Seven registered dry-bean categories represented through sixteen numerical measurements."
            )

            st.divider()

            st.markdown("**Classification models**")
            st.write("Dummy Classifier · Logistic Regression · Random Forest")

            st.divider()

            st.caption(
                "Evaluation includes stratified cross-validation, confusion matrices and permutation-based feature importance."
            )

    else:

        st.warning(
            "The dataset is currently unavailable. "
            "Please check the data source connection."
        )

    st.divider()

    # RESEARCH FOCUS

    st.markdown("### Research focus")

    st.caption(
        "The project combines three complementary areas "
        "of data science."
    )

    focus1, focus2, focus3 = st.columns(3)

    focus_cards = [
        (
            focus1,
            "01 / EXPLORATION",
            "Data Exploration",
            "Study physical measurements, category distributions "
            "and relationships between numerical features."
        ),
        (
            focus2,
            "02 / EVALUATION",
            "Machine Learning",
            "Compare classification algorithms using accuracy, "
            "Macro F1 and balanced accuracy."
        ),
        (
            focus3,
            "03 / INTERPRETATION",
            "Model Insights",
            "Investigate confusion patterns and understand the "
            "contribution of individual and grouped features."
        )
    ]


    for column, label, heading, description in focus_cards:

        with column:
            st.caption(label)
            st.subheader(heading)
            st.write(description)

    st.divider()

    # RESEARCH SCOPE

    st.markdown("### Research scope")

    st.info(
        """
        **Important distinction:** The current machine-learning
        experiment uses the UCI Dry Bean benchmark. It does not
        identify Indian traditional seed varieties or represent
        India's community seed-bank diversity.

        The Indian seed-conservation component is a separate
        planned direction, subject to appropriate data-reuse
        permission from the relevant organisations.
        """
    )

    st.caption(
        "BeejAtlas is an independent student research project."
    )

# ==================================================
# 8. SEED DATA EXPLORER
# ==================================================

elif page == "Seed Data Explorer":

    st.subheader("Dry Bean Data Explorer")

    st.write(
        """
        Explore the physical characteristics of seven dry-bean
        categories using the UCI benchmark dataset.
        """
    )

    if data is None:

        st.error("Cleaned dataset was not found.")

    else:

        categories = sorted(data["Class"].unique())

        selected_categories = st.multiselect(
            "Choose bean categories",
            categories,
            default=categories
        )

        filtered_data = data[
            data["Class"].isin(selected_categories)
        ]

        numeric_features = list(
            data.select_dtypes(include="number").columns
        )

        st.markdown("### Explore feature relationships")

        col1, col2 = st.columns(2)

        with col1:
            x_feature = st.selectbox(
                "X-axis feature",
                numeric_features,
                index=numeric_features.index("Area")
                if "Area" in numeric_features else 0
            )

        with col2:
            y_feature = st.selectbox(
                "Y-axis feature",
                numeric_features,
                index=numeric_features.index("Perimeter")
                if "Perimeter" in numeric_features else 1
            )

        scatter = px.scatter(
            filtered_data,
            x=x_feature,
            y=y_feature,
            color="Class",
            opacity=0.65,
            title=f"{x_feature} vs {y_feature}",
            hover_data=["Class"]
        )

        st.plotly_chart(
            scatter,
            use_container_width=True
        )

        st.markdown("### Feature distribution")

        selected_feature = st.selectbox(
            "Select a feature to examine",
            numeric_features,
            key="histogram_feature"
        )

        histogram = px.histogram(
            filtered_data,
            x=selected_feature,
            color="Class",
            barmode="overlay",
            opacity=0.65,
            title=f"Distribution of {selected_feature}"
        )

        st.plotly_chart(
            histogram,
            use_container_width=True
        )

        st.markdown("### Data preview")

        st.dataframe(
            filtered_data.head(100),
            use_container_width=True
        )

        st.download_button(
            "Download filtered dataset",
            filtered_data.to_csv(index=False),
            file_name="beejatlas_filtered_data.csv",
            mime="text/csv"
        )

        st.caption(
            f"Showing {len(filtered_data):,} observations."
        )


# ==================================================
# 9. MACHINE LEARNING LAB
# ==================================================

elif page == "Machine Learning Lab":

    st.subheader("Machine Learning Experiment")

    st.write(
        """
        Comparison of Dummy Classifier, Logistic Regression
        and Random Forest on the cleaned Dry Bean dataset.
        """
    )

    if model_results is None:

        st.warning("Model comparison results are not available.")

    else:

        st.markdown("### Test-set performance")

        display_results = model_results.copy()

        metric_options = {
            "Accuracy": "Test Accuracy",
            "Balanced Accuracy": "Test Balanced Accuracy",
            "Macro F1": "Test Macro F1"
        }

        selected_metric = st.selectbox(
            "Choose evaluation metric",
            list(metric_options.keys())
        )

        metric_column = metric_options[selected_metric]

        chart = px.bar(
            display_results,
            x="Model",
            y=metric_column,
            color="Model",
            title=f"Model Comparison — {selected_metric}"
        )

        chart.update_yaxes(
            range=[
                0,
                min(1, display_results[metric_column].max() + 0.10)
            ]
        )

        st.plotly_chart(
            chart,
            use_container_width=True
        )

        st.markdown("### Exact test results")

        st.dataframe(
            model_results.round(4),
            use_container_width=True
        )

    if cv_results is not None:

        st.markdown("### Five-fold cross-validation")

        st.dataframe(
            cv_results.round(4),
            use_container_width=True
        )

    st.divider()
    
    
    if repeated_cv_results is not None:

        st.markdown("### Repeated Stratified 5-Fold Cross-Validation")

        st.caption(
            "Three repetitions of five-fold cross-validation. "
            "Results show mean and standard deviation across "
            "15 fold scores per model."
        )

        display_cv = repeated_cv_results.copy()

        st.dataframe(
            display_cv.style.format({
                "Accuracy Mean": "{:.2%}",
                "Accuracy Std": "{:.2%}",
                "Macro F1 Mean": "{:.2%}",
                "Macro F1 Std": "{:.2%}",
                "Balanced Accuracy Mean": "{:.2%}",
                "Balanced Accuracy Std": "{:.2%}"
            }),
            hide_index=True,
            use_container_width=True
        )

        st.markdown("#### Macro F1 comparison")

        cv_chart = px.bar(
            repeated_cv_results,
            x="Model",
            y="Macro F1 Mean",
            color="Model",
            error_y="Macro F1 Std",
            title="Repeated Cross-Validation — Macro F1"
        )

        cv_chart.update_yaxes(
            tickformat=".0%",
            range=[0, 1.05]
        )

        cv_chart.update_layout(
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(
            cv_chart,
            use_container_width=True
        )

    else:
        st.info(
            "Repeated cross-validation results are not available."
        )


    st.markdown("### Classification reports")

    model_choice = st.selectbox(
        "Select a model",
        [
            "Dummy Classifier",
            "Logistic Regression",
            "Random Forest"
        ],
        index=2,
        key="ml_model_selector_v2"
    )

    report_filename = (
        model_choice.lower().replace(" ", "_")
        + "_classification_report.csv"
    )

    show_csv_report(report_filename)

    st.markdown("### Confusion matrix")

    confusion_filename = (
        model_choice.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    show_existing_image(
        confusion_filename,
        caption=f"{model_choice} confusion matrix"
    )


# ==================================================
# 10. FEATURE IMPORTANCE
# ==================================================

elif page == "Feature Importance":

    st.subheader("Understanding Model Behaviour")

    st.write(
        """
        Explore how individual physical measurements and groups
        of correlated features influence model predictions.
        """
    )

    if permutation_results is not None:

        st.markdown("### Individual permutation importance")

        selected_model = st.selectbox(
            "Choose a model",
            sorted(permutation_results["Model"].unique())
        )

        selected_results = permutation_results[
            permutation_results["Model"] == selected_model
        ].sort_values(
            "Importance Mean",
            ascending=True
        )

        fig = px.bar(
            selected_results,
            x="Importance Mean",
            y="Feature",
            error_x="Importance Std",
            orientation="h",
            title=f"{selected_model}: Permutation Importance"
        )

        fig.add_vline(
            x=0,
            line_dash="dash"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            selected_results.round(5),
            use_container_width=True
        )

    else:

        st.info("Individual permutation results are not available.")

    st.divider()

    if grouped_results is not None:

        st.markdown("### Grouped permutation importance")

        grouped_chart = px.bar(
            grouped_results,
            x="Importance Mean",
            y="Group",
            color="Model",
            barmode="group",
            orientation="h",
            error_x="Importance Std",
            hover_data=["Features"]
        )

        grouped_chart.add_vline(x=0, line_dash="dash")

        st.plotly_chart(
            grouped_chart,
            use_container_width=True
        )

        st.dataframe(
            grouped_results.round(5),
            use_container_width=True
        )

    if builtin_importance is not None:

        st.divider()

        st.markdown(
            "### Random Forest built-in feature importance"
        )

        built_in = builtin_importance.sort_values(
            "Importance",
            ascending=True
        )

        fig = px.bar(
            built_in,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ==================================================
# 11. RESEARCH AND LIMITATIONS
# ==================================================

elif page == "Research & Limitations":

    st.subheader("Research Methodology")

    st.markdown("### Data preparation")

    st.write(
        """
        The original UCI dataset contained 13,611 observations.
        After removing 68 exact duplicate records, 13,543
        observations remained.
        """
    )

    st.markdown("### Experimental design")

    st.write(
        """
        - 80:20 stratified train-test split.
        - Random state: 42.
        - Five-fold stratified cross-validation.
        - Dummy Classifier baseline.
        - Logistic Regression.
        - Random Forest.
        - Macro F1 and balanced accuracy evaluation.
        - Individual and grouped permutation importance.
        """
    )

    st.markdown("### Important limitations")

    limitations = [
        "The UCI dataset is a dry-bean classification benchmark, not an Indian traditional-seed dataset.",
        "The benchmark does not establish real-world identification accuracy for traditional Indian seed varieties.",
        "The class distribution is uneven.",
        "Physical features may be correlated and can share predictive information.",
        "Model performance has only been evaluated using the available benchmark dataset.",
        "Repeated use of the same test set for interpretation makes the reported results exploratory rather than independent final validation.",
        "The benchmark does not establish seed quality, germination, nutritional value or climate resilience."
    ]

    for limitation in limitations:
        st.markdown(f"- {limitation}")

    st.markdown("### Dataset attribution")

    st.write(
        """
        UCI Machine Learning Repository:
        Dry Bean Dataset.
        DOI: 10.24432/C50S4B.
        """

    )

    st.link_button(
        "Open UCI Dry Bean Dataset",
        "https://archive.ics.uci.edu/dataset/602/dry+bean"
    )


# ==================================================
# 12. INDIAN SEED CONSERVATION
# ==================================================

elif page == "Indian Seed Conservation":

    st.subheader("Indian Seed Conservation ")

    st.write(
        """
        This section represents the original inspiration behind
        BeejAtlas: traditional seed diversity, community seed banks,
        local agricultural knowledge and conservation.
        """
    )

    st.info(
        """
        **Research component under development**

        Permission to reuse the Centre for Science and Environment's
        Community Seed Banks dataset has been requested.

        Until permission is obtained, its underlying data is not
        included, copied or redistributed in this application.
        """
    )

    st.markdown("### Project inspiration")

    st.write(
        """
        BeejAtlas is inspired by work surrounding seed sovereignty,
        traditional agricultural knowledge and community-led
        conservation initiatives.
        """

    )

    st.link_button(
        "Explore CSE's Community Seed Banks resource",
        "https://www.cseindia.org/page/seedbank"
    )

    st.markdown(
        """
        Our future goal is to responsibly explore documented
        seed-conservation efforts and make permitted information
        easier to understand through data visualisation.
        """
    )


# ==================================================
# 13. FOOTER
# ==================================================

st.divider()

st.caption(
    "BeejAtlas | Independent Data Science Research Project | "
    "UCI Dry Bean Benchmark"
)

st.caption(
    "Built with Python, Streamlit, Plotly, pandas and scikit-learn."
)
