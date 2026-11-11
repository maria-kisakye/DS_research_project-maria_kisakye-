
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Explainable Credit Risk Assessment",
    page_icon="📊",
    layout="wide",
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "graphsage"
FIGURES_DIR = RESULTS_DIR / "figures"

# --------------------------------------------------
# LOAD SAVED EXPERIMENTAL RESULTS
# --------------------------------------------------
@st.cache_data
def load_results():
    comparison = pd.read_csv(
        RESULTS_DIR / "final_test_model_comparison.csv"
    )
    predictions = pd.read_csv(
        RESULTS_DIR / "final_test_predictions.csv"
    )
    return comparison, predictions


comparison_df, predictions_df = load_results()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("Research Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Project Overview",
        "Model Comparison",
        "Credit Risk Predictions",
        "Explainability",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Uganda Christian University\n"
    "BSc Data Science Research Project"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("Explainable Credit Risk Assessment")
st.caption(
    "GraphSAGE and Logistic Regression | "
    "Historical Prosper Lending Network"
)

# --------------------------------------------------
# PROJECT OVERVIEW
# --------------------------------------------------
if page == "Project Overview":

    st.header("Research Project Overview")

    st.write(
        "This research investigates whether Graph Neural Networks "
        "can improve credit default prediction by incorporating "
        "historical relationships between lenders and borrowing events."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Lending Records", "3.39M")
    col2.metric("Lender Nodes", "59,046")
    col3.metric("Borrowing Events", "43,572")

    st.subheader("Research Methodology")

    st.markdown(
        """
        1. Construct a heterogeneous lender-event network.
        2. Select historical lending relationships using timestamps.
        3. Train a two-layer GraphSAGE model.
        4. Compare performance against Logistic Regression.
        5. Explain predictions using SHAP and GNNExplainer.
        """
    )

    st.info(
        "The experiments use historical Prosper data. "
        "The relevance to Uganda's fintech sector is discussed "
        "as a potential future application, not as a validated deployment."
    )

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------
elif page == "Model Comparison":

    st.header("Held-Out Test Performance")

    st.write(
        "Both models were evaluated on 5,545 held-out "
        "borrowing events."
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("ROC Curves")

    roc_path = FIGURES_DIR / "final_test_roc_curves.png"

    if roc_path.exists():
        st.image(str(roc_path), use_container_width=True)

    st.subheader("Precision-Recall Curves")

    pr_path = (
        FIGURES_DIR / "final_test_precision_recall_curves.png"
    )

    if pr_path.exists():
        st.image(str(pr_path), use_container_width=True)

    st.subheader("Confusion Matrices")

    confusion_path = (
        FIGURES_DIR / "final_test_confusion_matrices.png"
    )

    if confusion_path.exists():
        st.image(
            str(confusion_path),
            use_container_width=True,
        )

# --------------------------------------------------
# CREDIT RISK PREDICTIONS
# --------------------------------------------------

elif page == "Credit Risk Predictions":

    st.header("Interactive Credit Risk Prediction Explorer")

    st.write(
        "Select a historical borrowing event to compare "
        "the default predictions made by Logistic Regression "
        "and GraphSAGE on the held-out test dataset."
    )

    # Select an actual event from the saved test predictions
    event_ids = predictions_df["event_id"].astype(int).tolist()

    selected_event = st.selectbox(
        "Select a borrowing event",
        options=event_ids,
        index=0,
    )

    # Retrieve the selected event's saved results
    event = predictions_df.loc[
        predictions_df["event_id"] == selected_event
    ].iloc[0]

    actual_label = int(event["true_label"])

    lr_probability = float(
        event["lr_default_probability"]
    )

    gnn_probability = float(
        event["gnn_default_probability"]
    )

    lr_prediction = int(event["lr_prediction"])
    gnn_prediction = int(event["gnn_prediction"])

    st.divider()

    st.subheader(f"Borrowing Event {selected_event}")

    actual_outcome = (
        "Default"
        if actual_label == 1
        else "Non-default"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Actual Historical Outcome",
        actual_outcome,
    )

    col2.metric(
        "Logistic Regression",
        f"{lr_probability:.1%}",
    )

    col3.metric(
        "GraphSAGE",
        f"{gnn_probability:.1%}",
    )

    st.subheader("Predicted Default Probabilities")

    probability_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "GraphSAGE",
        ],
        "Default Probability": [
            lr_probability,
            gnn_probability,
        ],
    }).set_index("Model")

    st.bar_chart(
        probability_df,
        y="Default Probability",
    )

    st.subheader("Model Decisions")

    decision_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "GraphSAGE",
        ],
        "Default Probability": [
            f"{lr_probability:.2%}",
            f"{gnn_probability:.2%}",
        ],
        "Decision Threshold": [
            "20%",
            "22%",
        ],
        "Predicted Outcome": [
            "Default" if lr_prediction == 1 else "Non-default",
            "Default" if gnn_prediction == 1 else "Non-default",
        ],
        "Correct Prediction": [
            "Yes" if lr_prediction == actual_label else "No",
            "Yes" if gnn_prediction == actual_label else "No",
        ],
    })

    st.dataframe(
        decision_df,
        use_container_width=True,
        hide_index=True,
    )

    if lr_prediction != gnn_prediction:
        st.info(
            "The two models disagree on this borrowing event. "
            "This illustrates how their assessments can differ."
        )
    else:
        st.success(
            "Both models predict the same outcome "
            "for this borrowing event."
        )

    st.divider()

    st.caption(
        "These are previously generated predictions for "
        "historical Prosper borrowing events. "
        "Thresholds were selected using validation data. "
        "This prototype does not score new applicants "
        "or provide lending decisions."
    )


# --------------------------------------------------
# EXPLAINABILITY
# --------------------------------------------------
elif page == "Explainability":

    st.header("Understanding Model Predictions")

    st.subheader("SHAP Feature Importance")

    shap_path = (
        FIGURES_DIR / "logistic_regression_shap_importance.png"
    )

    if shap_path.exists():
        st.image(str(shap_path), use_container_width=True)

    st.subheader("Historical Relationship Sensitivity")

    sensitivity_path = (
        FIGURES_DIR / "graphsage_relational_sensitivity.png"
    )

    if sensitivity_path.exists():
        st.image(
            str(sensitivity_path),
            use_container_width=True,
        )

    st.subheader("GNNExplainer Relationship Importance")

    gnn_path = (
        FIGURES_DIR
        / "gnnexplainer_event_17448_edge_directions.png"
    )

    if gnn_path.exists():
        st.image(str(gnn_path), use_container_width=True)

    st.warning(
        "Graph explanations were evaluated on a limited "
        "validation sample and should not be interpreted "
        "as causal explanations."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Research by Kisakye Maria Sengendo | "
    "Uganda Christian University | 2026"
)

