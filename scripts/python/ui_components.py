"""
User Interface components for the Sanctions False Positive Detection application.

This module contains all Streamlit UI components including visualizations,
controls, and display functions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from config import (
    ALL_FEATURES, TARGET_COLUMN, DEFAULT_TEST_SIZE, DEFAULT_THRESHOLD,
    DEFAULT_NAME_SIMILARITY, DEFAULT_PRIOR_FP_RATE
)


def setup_page_config() -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Sanctions False-Positive Reducer",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def render_header() -> None:
    """Render the main application header."""
    st.title("🛡️ Sanctions False-Positive Reducer (Demo)")
    st.write("A machine learning demo to **prioritize likely false positives** and **assist analysts** in sanctions screening.")


def render_sidebar_controls() -> Tuple[bool, st.runtime.uploaded_file_manager.UploadedFile, float, float]:
    """
    Render sidebar controls for data source and model parameters.
    
    Returns:
        Tuple of (use_sample, uploaded_file, test_size, threshold)
    """
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    use_sample = st.sidebar.checkbox("Use built-in synthetic dataset", value=True)
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV",
        type=["csv"],
        help="Upload a CSV file with the required features"
    )
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    test_size = st.sidebar.slider(
        "Test Set Size",
        min_value=0.1,
        max_value=0.4,
        value=DEFAULT_TEST_SIZE,
        step=0.05,
        help="Proportion of data used for testing"
    )
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.2,
        max_value=0.8,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="Probability threshold for classifying as true match"
    )
    
    return use_sample, uploaded_file, test_size, threshold


def render_dataset_overview(dataset: pd.DataFrame) -> None:
    """
    Display dataset overview with key statistics.
    
    Args:
        dataset: The loaded dataset
    """
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(dataset))
    with col2:
        st.metric("Features", len(ALL_FEATURES))
    with col3:
        true_matches = dataset[TARGET_COLUMN].sum()
        st.metric("True Matches", f"{true_matches} ({true_matches/len(dataset):.1%})")
    
    st.dataframe(dataset.head(5), use_container_width=True)


def render_model_results(metrics_df: pd.DataFrame, confusion_matrix_data: np.ndarray) -> None:
    """
    Display model training results including metrics and confusion matrix.
    
    Args:
        metrics_df: DataFrame containing performance metrics
        confusion_matrix_data: Confusion matrix array
    """
    st.subheader("🤖 Model Training & Evaluation")
    
    metrics_column, matrix_column = st.columns(2)
    with metrics_column:
        st.markdown("#### Performance Metrics")
        st.dataframe(metrics_df, use_container_width=True)
    with matrix_column:
        st.markdown("#### Confusion Matrix")
        render_confusion_matrix(confusion_matrix_data)


def render_confusion_matrix(confusion_matrix_data: np.ndarray) -> None:
    """
    Plot confusion matrix with meaningful colors for each category.
    
    Args:
        confusion_matrix_data: 2x2 numpy array containing confusion matrix values
        
    Color scheme:
    - True Negatives (Correct FP): Light Green (Good prediction)
    - False Positives (Incorrect True): Light Red (Bad prediction)
    - False Negatives (Incorrect FP): Orange (Missed true match)
    - True Positives (Correct True): Dark Green (Good prediction)
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Define meaningful colors for each cell
    # [0,0] = True Negative (Correctly predicted FP) - Light Green
    # [0,1] = False Positive (Incorrectly predicted True) - Light Red
    # [1,0] = False Negative (Incorrectly predicted FP) - Orange
    # [1,1] = True Positive (Correctly predicted True) - Dark Green
    colors = [
        ['#90EE90', '#FFB6C1'],  # Row 0: Light Green, Light Red
        ['#FFA500', '#228B22']   # Row 1: Orange, Dark Green
    ]
    
    # Create colored background for each cell
    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                               facecolor=colors[i][j], alpha=0.7)
            ax.add_patch(rect)
    
    # Set labels and title
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted FP", "Predicted True"], fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual FP", "Actual True"], fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold', pad=20)
    
    # Add value annotations with black text
    for i in range(2):
        for j in range(2):
            value = int(confusion_matrix_data[i, j])
            ax.text(j, i, str(value),
                   ha="center", va="center",
                   color="black",
                   fontsize=18, fontweight='bold')
    
    # Add category labels below each value
    labels = [
        ['True Negative\n(Correct FP)', 'False Positive\n(Wrong True)'],
        ['False Negative\n(Missed True)', 'True Positive\n(Correct True)']
    ]
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i-0.25, labels[i][j],
                   ha="center", va="center",
                   color="black",
                   fontsize=9, style='italic')
    
    # Set limits and remove ticks
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    
    # Add grid lines
    ax.axhline(y=0.5, color='black', linewidth=2)
    ax.axvline(x=0.5, color='black', linewidth=2)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#90EE90', alpha=0.7, label='Correct FP Prediction'),
        plt.Rectangle((0,0),1,1, facecolor='#228B22', alpha=0.7, label='Correct True Prediction'),
        plt.Rectangle((0,0),1,1, facecolor='#FFB6C1', alpha=0.7, label='Incorrect Prediction'),
        plt.Rectangle((0,0),1,1, facecolor='#FFA500', alpha=0.7, label='Missed True Match')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    st.pyplot(fig)


def render_model_interpretability(coefficients_table: pd.DataFrame) -> None:
    """
    Display model interpretability section with coefficients table.
    
    Args:
        coefficients_table: DataFrame containing model coefficients
    """
    st.markdown("---")
    st.subheader("🔍 Model Interpretability")
    
    if not coefficients_table.empty:
        st.dataframe(coefficients_table, use_container_width=True)

        coefficients_csv = coefficients_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Coefficients (CSV)",
            data=coefficients_csv,
            file_name="model_coefficients.csv",
            mime="text/csv",
        )

        st.caption(
            "**Interpretation Guide:** Positive coefficients increase true match probability; "
            "negative coefficients decrease it. Odds ratios > 1 increase odds; < 1 decrease odds. "
            "Categorical features are one-hot encoded."
        )


def render_single_alert_controls() -> pd.DataFrame:
    """
    Render Streamlit controls for testing individual alerts.
    
    Returns:
        Single-row DataFrame containing user-selected feature values
    """
    st.markdown("---")
    st.subheader("🧪 Test Individual Alert")
    st.write("**Configure Alert Features:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name_similarity = st.slider(
            "Name Similarity",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_NAME_SIMILARITY,
            step=0.01,
            help="Similarity score between customer name and watchlist name"
        )
        
        dob_match = st.selectbox(
            "Date of Birth Match",
            options=[0, 1],
            format_func=lambda x: "No Match" if x == 0 else "Match",
            help="Whether customer DOB matches watchlist entry"
        )
        
        country_risk = st.selectbox(
            "Country Risk Level",
            options=[1, 2, 3],
            format_func=lambda x: f"Level {x} ({'Low' if x==1 else 'Medium' if x==2 else 'High'})",
            help="Risk level of customer's country"
        )
        
        pep_flag = st.selectbox(
            "PEP Flag",
            options=[0, 1],
            format_func=lambda x: "Not PEP" if x == 0 else "PEP",
            help="Politically Exposed Person indicator"
        )
    
    with col2:
        prior_fp_rate = st.slider(
            "Prior False Positive Rate",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_PRIOR_FP_RATE,
            step=0.01,
            help="Historical false positive rate for this name"
        )
        
        list_type = st.selectbox(
            "List Type",
            options=["WORLDCHECK", "IWL"],
            help="Source of the watchlist entry"
        )
        
        alert_type = st.selectbox(
            "Alert Type",
            options=["name", "dob", "address"],
            help="Type of matching that triggered the alert"
        )

    alert_features = {
        "name_similarity": name_similarity,
        "dob_match": dob_match,
        "country_risk": country_risk,
        "pep_flag": pep_flag,
        "prior_name_fp_rate": prior_fp_rate,
        "list_type": list_type,
        "alert_type": alert_type,
    }
    
    return pd.DataFrame([alert_features])


def render_prediction_results(
    true_match_probability: float, 
    prediction: int, 
    threshold: float, 
    explanation: str
) -> None:
    """
    Display prediction results with color coding and explanation.
    
    Args:
        true_match_probability: Predicted probability of true match
        prediction: Binary prediction (0 or 1)
        threshold: Classification threshold used
        explanation: Human-readable explanation of the prediction
    """
    # Display prediction with color coding
    prediction_color = "🔴" if prediction == 0 else "🟢"
    prediction_text = "False Positive" if prediction == 0 else "True Match"
    
    st.markdown(f"""
    ### {prediction_color} Prediction Results
    **True-Match Probability:** {true_match_probability:.1%}  
    **Classification:** {prediction_text} (threshold: {threshold:.2f})
    """)
    
    st.info(explanation)


def render_test_results(scored_dataset: pd.DataFrame) -> None:
    """
    Display test set results with summary statistics and download option.
    
    Args:
        scored_dataset: DataFrame containing test results with predictions
    """
    st.markdown("---")
    st.subheader("📋 Test Set Results")
    
    # Summary statistics
    accuracy = (scored_dataset['actual_labels'] == scored_dataset['predicted_label']).mean()
    st.metric("Test Set Accuracy", f"{accuracy:.1%}")
    
    st.dataframe(scored_dataset.head(12), use_container_width=True)

    scored_csv = scored_dataset.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Test Results (CSV)",
        data=scored_csv,
        file_name="scored_test_set.csv",
        mime="text/csv",
    )


def render_footer() -> None:
    """Render the application footer."""
    st.markdown("---")
    st.caption("⚠️ **Demo Application** - For educational and demonstration purposes only.")


def show_error(error_message: str) -> None:
    """
    Display error message with helpful information.
    
    Args:
        error_message: The error message to display
    """
    st.error(f"An error occurred: {error_message}")
    st.info("Please check your data format and try again.")