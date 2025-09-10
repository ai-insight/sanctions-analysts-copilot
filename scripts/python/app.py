import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from data_gen import generate_synthetic_dataset


# =========================
# Config & Constants
# =========================
# Threshold constants for feature interpretation
NAME_SIMILARITY_HIGH_THRESHOLD = 0.85
NAME_SIMILARITY_MODERATE_THRESHOLD = 0.6
PRIOR_FP_RATE_HIGH_THRESHOLD = 0.7

# Risk level constants
COUNTRY_RISK_HIGH = 3
COUNTRY_RISK_MEDIUM = 2
DOB_MATCH_TRUE = 1
PEP_FLAG_PRESENT = 1

# Model configuration
DEFAULT_RANDOM_STATE = 7
MAX_ITERATIONS = 200

# Feature definitions
NUM_FEATURES = ["name_similarity", "dob_match", "country_risk", "pep_flag", "prior_name_fp_rate"]
CAT_FEATURES = ["list_type", "alert_type"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

# Target column name
TARGET_COLUMN = "label_true_match"


# =========================
# Data utilities
# =========================
def load_data(use_sample: bool, uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile]) -> pd.DataFrame:
    """
    Load dataset from either synthetic generator or uploaded CSV file.
    
    Args:
        use_sample: If True, use synthetic dataset instead of uploaded file
        uploaded_file: Streamlit uploaded file object containing CSV data
        
    Returns:
        DataFrame containing the sanctions screening dataset
        
    Raises:
        Exception: If uploaded file cannot be read as CSV
    """
    if use_sample or uploaded_file is None:
        return generate_synthetic_dataset()
    
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading uploaded file: {str(e)}")
        st.info("Using synthetic dataset instead.")
        return generate_synthetic_dataset()


# =========================
# Modeling utilities
# =========================
def build_model() -> Pipeline:
    """
    Create a machine learning pipeline with preprocessing and classification.
    
    The pipeline includes:
    - Numerical features: passed through without transformation
    - Categorical features: one-hot encoded with unknown value handling
    - Classifier: Logistic regression with balanced class weights
    
    Returns:
        Configured sklearn Pipeline ready for training
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )
    classifier = LogisticRegression(
        max_iter=MAX_ITERATIONS,
        class_weight="balanced",
        solver="liblinear",
        random_state=DEFAULT_RANDOM_STATE
    )
    return Pipeline([("prep", preprocessor), ("clf", classifier)])


def train_and_evaluate(
    dataset: pd.DataFrame,
    test_size: float,
    threshold: float,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[Pipeline, pd.DataFrame, pd.Series, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Train machine learning pipeline and evaluate performance on test set.
    
    Args:
        dataset: Input dataset containing features and target labels
        test_size: Proportion of dataset to use for testing (0.0 to 1.0)
        threshold: Classification threshold for converting probabilities to predictions
        random_state: Random seed for reproducible train/test splits
        
    Returns:
        Tuple containing:
        - Trained pipeline model
        - Test features DataFrame
        - Test labels Series
        - Prediction probabilities array
        - Binary predictions array
        - Confusion matrix array
        - Metrics DataFrame with precision, recall, F1
        
    Raises:
        KeyError: If required columns are missing from dataset
        ValueError: If test_size is not between 0 and 1
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    if TARGET_COLUMN not in dataset.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset")
    
    missing_features = [f for f in ALL_FEATURES if f not in dataset.columns]
    if missing_features:
        raise KeyError(f"Missing required features: {missing_features}")

    features = dataset[ALL_FEATURES]
    target_labels = dataset[TARGET_COLUMN]

    features_train, features_test, labels_train, labels_test = train_test_split(
        features, target_labels, test_size=test_size, stratify=target_labels, random_state=random_state
    )

    model = build_model()
    model.fit(features_train, labels_train)

    probabilities = model.predict_proba(features_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(labels_test, predictions, average="binary")
    confusion_matrix_result = confusion_matrix(labels_test, predictions)

    metrics_df = pd.DataFrame(
        {"Metric": ["Precision", "Recall", "F1"], "Score": [precision, recall, f1]}
    )

    return model, features_test, labels_test, probabilities, predictions, confusion_matrix_result, metrics_df


def get_coeff_table(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extract model coefficients and odds ratios for interpretability.
    
    Args:
        pipeline: Trained sklearn Pipeline containing preprocessor and classifier
        
    Returns:
        DataFrame with feature names, coefficients, and odds ratios sorted by coefficient magnitude
        
    Raises:
        AttributeError: If pipeline doesn't contain expected components
    """
    try:
        classifier = pipeline.named_steps["clf"]
        encoder = pipeline.named_steps["prep"].named_transformers_["cat"]
        categorical_feature_names = encoder.get_feature_names_out(CAT_FEATURES)
        all_feature_names = np.concatenate([np.array(NUM_FEATURES), categorical_feature_names])

        coefficients = classifier.coef_[0]
        coefficients_table = pd.DataFrame(
            {
                "Feature": all_feature_names,
                "Coefficient": coefficients,
                "Odds Ratio (exp(coef))": np.exp(coefficients),
            }
        ).sort_values(by="Coefficient", ascending=False)
        return coefficients_table
    except (KeyError, AttributeError) as e:
        st.error(f"Error extracting coefficients: {str(e)}")
        return pd.DataFrame()


def build_scored_test_df(
    features_test: pd.DataFrame,
    labels_test: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray
) -> pd.DataFrame:
    """
    Combine test features with model predictions for analysis.
    
    Args:
        features_test: Test set features DataFrame
        labels_test: True labels for test set
        probabilities: Model prediction probabilities
        predictions: Binary predictions from model
        
    Returns:
        DataFrame combining features, true labels, probabilities, and predictions
    """
    scored_dataset = features_test.copy()
    scored_dataset["actual_labels"] = labels_test.values
    scored_dataset["probability_true_match"] = probabilities
    scored_dataset["predicted_label"] = predictions
    return scored_dataset.reset_index(drop=True)


def explain_alert(alert_data: pd.Series, true_match_probability: float) -> str:
    """
    Generate human-readable explanation of alert features for analysts.
    
    Args:
        alert_data: Series containing feature values for a single alert
        true_match_probability: Model's predicted probability of true match
        
    Returns:
        Formatted string with probability and key signal descriptions
    """
    signal_descriptions = []
    
    # Name similarity assessment
    if alert_data["name_similarity"] >= NAME_SIMILARITY_HIGH_THRESHOLD:
        signal_descriptions.append("very high name similarity")
    elif alert_data["name_similarity"] >= NAME_SIMILARITY_MODERATE_THRESHOLD:
        signal_descriptions.append("moderate name similarity")
    else:
        signal_descriptions.append("low name similarity")

    # Date of birth match
    if int(alert_data["dob_match"]) == DOB_MATCH_TRUE:
        signal_descriptions.append("DOB matches")

    # Country risk assessment
    if int(alert_data["country_risk"]) == COUNTRY_RISK_HIGH:
        signal_descriptions.append("high-risk country")
    elif int(alert_data["country_risk"]) == COUNTRY_RISK_MEDIUM:
        signal_descriptions.append("medium-risk country")

    # PEP (Politically Exposed Person) flag
    if int(alert_data["pep_flag"]) == PEP_FLAG_PRESENT:
        signal_descriptions.append("PEP flag present")

    # Historical false positive rate
    if alert_data["prior_name_fp_rate"] >= PRIOR_FP_RATE_HIGH_THRESHOLD:
        signal_descriptions.append("name has high false-positive history")

    # List source
    if alert_data["list_type"] == "IWL":
        signal_descriptions.append("from Internal Watch List")
    else:
        signal_descriptions.append("from World-Check")

    return f"Analyst note: Probability={true_match_probability:.0%}. Signals: " + ", ".join(signal_descriptions)


# =========================
# UI utilities
# =========================
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


def render_single_alert_controls() -> pd.DataFrame:
    """
    Render Streamlit controls for testing individual alerts.
    
    Returns:
        Single-row DataFrame containing user-selected feature values
    """
    st.write("**Configure Alert Features:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name_similarity = st.slider(
            "Name Similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
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
            value=0.7,
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


# =========================
# Main app
# =========================
def main() -> None:
    """
    Main Streamlit application for sanctions false positive detection.
    
    This application provides:
    - Dataset loading and preview
    - Model training and evaluation
    - Interactive single alert testing
    - Model interpretability through coefficients
    - Downloadable results
    """
    # Page configuration
    st.set_page_config(
        page_title="Sanctions False-Positive Reducer",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🛡️ Sanctions False-Positive Reducer (Demo)")
    st.write("A machine learning demo to **prioritize likely false positives** and **assist analysts** in sanctions screening.")
    
    # Sidebar controls
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
        value=0.25,
        step=0.05,
        help="Proportion of data used for testing"
    )
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.2,
        max_value=0.8,
        value=0.5,
        step=0.05,
        help="Probability threshold for classifying as true match"
    )

    try:
        # Load and display data
        st.subheader("📊 Dataset Overview")
        dataset = load_data(use_sample, uploaded_file)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(dataset))
        with col2:
            st.metric("Features", len(ALL_FEATURES))
        with col3:
            true_matches = dataset[TARGET_COLUMN].sum()
            st.metric("True Matches", f"{true_matches} ({true_matches/len(dataset):.1%})")
        
        st.dataframe(dataset.head(5), width='stretch')

        # Train and evaluate model
        st.subheader("🤖 Model Training & Evaluation")
        with st.spinner("Training model..."):
            pipeline, features_test, labels_test, probabilities, predictions, confusion_matrix_result, metrics_df = train_and_evaluate(
                dataset=dataset, test_size=test_size, threshold=threshold
            )

        # Display results
        metrics_column, matrix_column = st.columns(2)
        with metrics_column:
            st.markdown("#### Performance Metrics")
            st.dataframe(metrics_df, width='stretch')
        with matrix_column:
            st.markdown("#### Confusion Matrix")
            render_confusion_matrix(confusion_matrix_result)

        # Model interpretability
        st.markdown("---")
        st.subheader("🔍 Model Interpretability")
        coefficients_table = get_coeff_table(pipeline)
        
        if not coefficients_table.empty:
            st.dataframe(coefficients_table, width='stretch')

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

        # Interactive alert testing
        st.markdown("---")
        st.subheader("🧪 Test Individual Alert")
        single_alert = render_single_alert_controls()

        if st.button("🔍 Analyze Alert", type="primary"):
            true_match_probability = pipeline.predict_proba(single_alert)[0, 1]
            prediction = int(true_match_probability >= threshold)
            
            # Display prediction with color coding
            prediction_color = "🔴" if prediction == 0 else "🟢"
            prediction_text = "False Positive" if prediction == 0 else "True Match"
            
            st.markdown(f"""
            ### {prediction_color} Prediction Results
            **True-Match Probability:** {true_match_probability:.1%}
            **Classification:** {prediction_text} (threshold: {threshold:.2f})
            """)
            
            st.info(explain_alert(single_alert.iloc[0], true_match_probability))

        # Test set results
        st.markdown("---")
        st.subheader("📋 Test Set Results")
        scored_dataset = build_scored_test_df(features_test, labels_test, probabilities, predictions)
        
        # Summary statistics
        accuracy = (scored_dataset['actual_labels'] == scored_dataset['predicted_label']).mean()
        st.metric("Test Set Accuracy", f"{accuracy:.1%}")
        
        st.dataframe(scored_dataset.head(12), width='stretch')

        scored_csv = scored_dataset.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Test Results (CSV)",
            data=scored_csv,
            file_name="scored_test_set.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your data format and try again.")
    
    # Footer
    st.markdown("---")
    st.caption("⚠️ **Demo Application** - For educational and demonstration purposes only.")


if __name__ == "__main__":
    main()
