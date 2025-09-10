"""
Machine Learning model functionality for Sanctions False Positive Detection.

This module contains all ML-related functions including data loading, model building,
training, evaluation, and prediction utilities.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from data_gen import generate_synthetic_dataset
from config import (
    NUM_FEATURES, CAT_FEATURES, ALL_FEATURES, TARGET_COLUMN,
    DEFAULT_RANDOM_STATE, MAX_ITERATIONS,
    NAME_SIMILARITY_HIGH_THRESHOLD, NAME_SIMILARITY_MODERATE_THRESHOLD,
    PRIOR_FP_RATE_HIGH_THRESHOLD, COUNTRY_RISK_HIGH, COUNTRY_RISK_MEDIUM,
    DOB_MATCH_TRUE, PEP_FLAG_PRESENT
)


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


def predict_single_alert(pipeline: Pipeline, alert_data: pd.DataFrame, threshold: float) -> Tuple[float, int, str]:
    """
    Make prediction for a single alert and generate explanation.
    
    Args:
        pipeline: Trained ML pipeline
        alert_data: Single-row DataFrame containing alert features
        threshold: Classification threshold
        
    Returns:
        Tuple of (probability, prediction, explanation)
    """
    true_match_probability = pipeline.predict_proba(alert_data)[0, 1]
    prediction = int(true_match_probability >= threshold)
    explanation = explain_alert(alert_data.iloc[0], true_match_probability)
    
    return true_match_probability, prediction, explanation