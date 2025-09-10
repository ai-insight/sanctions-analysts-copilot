"""
Synthetic dataset generator for Sanctions False Positive Detection.

This module generates realistic synthetic data for training and testing
the sanctions screening false positive detection model.
"""

import numpy as np
import pandas as pd
from typing import Optional


# =========================
# Data Generation Constants
# =========================

# Default dataset parameters
DEFAULT_DATASET_SIZE = 1200
DEFAULT_RANDOM_SEED = 42
DECIMAL_PRECISION = 3

# Feature distribution parameters
NAME_SIMILARITY_BETA_PARAMS = (2, 2)  # Beta distribution parameters for name similarity
DOB_MATCH_PROBABILITY = 0.25  # Probability of DOB match
COUNTRY_RISK_PROBABILITIES = [0.6, 0.3, 0.1]  # Low, Medium, High risk probabilities
PEP_FLAG_PROBABILITY = 0.1  # Probability of PEP flag
PRIOR_FP_RATE_MEAN = 0.7  # Mean for prior false positive rate
PRIOR_FP_RATE_STD = 0.15  # Standard deviation for prior false positive rate

# Categorical feature probabilities
LIST_TYPE_PROBABILITIES = [0.8, 0.2]  # WORLDCHECK, IWL
ALERT_TYPE_PROBABILITIES = [0.7, 0.2, 0.1]  # name, dob, address

# True match probability model coefficients
BASE_TRUE_MATCH_PROBABILITY = 0.05
NAME_SIMILARITY_WEIGHT = 0.7
DOB_MATCH_WEIGHT = 0.25
COUNTRY_RISK_WEIGHT = 0.12
PEP_FLAG_WEIGHT = 0.2
PRIOR_FP_RATE_WEIGHT = -0.6  # Negative because higher FP rate reduces true match probability
WORLDCHECK_BONUS = 0.05
NON_NAME_ALERT_BONUS = 0.03


def generate_numerical_features(dataset_size: int) -> dict:
    """
    Generate numerical features for the synthetic dataset.
    
    Args:
        dataset_size: Number of records to generate
        
    Returns:
        Dictionary containing numerical feature arrays
    """
    features = {}
    
    # Name similarity: Beta distribution clipped to [0,1]
    features['name_similarity'] = np.clip(
        np.random.beta(*NAME_SIMILARITY_BETA_PARAMS, dataset_size), 0, 1
    )
    
    # Date of birth match: Binary feature
    features['dob_match'] = np.random.binomial(1, DOB_MATCH_PROBABILITY, dataset_size)
    
    # Country risk: Categorical (1=low, 2=medium, 3=high)
    features['country_risk'] = np.random.choice(
        [1, 2, 3], p=COUNTRY_RISK_PROBABILITIES, size=dataset_size
    )
    
    # PEP (Politically Exposed Person) flag: Binary feature
    features['pep_flag'] = np.random.binomial(1, PEP_FLAG_PROBABILITY, dataset_size)
    
    # Prior false positive rate: Normal distribution clipped to [0,1]
    features['prior_name_fp_rate'] = np.clip(
        np.random.normal(PRIOR_FP_RATE_MEAN, PRIOR_FP_RATE_STD, dataset_size), 0, 1
    )
    
    return features


def generate_categorical_features(dataset_size: int) -> dict:
    """
    Generate categorical features for the synthetic dataset.
    
    Args:
        dataset_size: Number of records to generate
        
    Returns:
        Dictionary containing categorical feature arrays
    """
    features = {}
    
    # List type: Source of the watchlist entry
    features['list_type'] = np.random.choice(
        ["WORLDCHECK", "IWL"], p=LIST_TYPE_PROBABILITIES, size=dataset_size
    )
    
    # Alert type: Type of matching that triggered the alert
    features['alert_type'] = np.random.choice(
        ["name", "dob", "address"], p=ALERT_TYPE_PROBABILITIES, size=dataset_size
    )
    
    return features


def calculate_true_match_probabilities(
    name_similarity: np.ndarray,
    dob_match: np.ndarray,
    country_risk: np.ndarray,
    pep_flag: np.ndarray,
    prior_name_fp_rate: np.ndarray,
    list_type: np.ndarray,
    alert_type: np.ndarray
) -> np.ndarray:
    """
    Calculate the probability of true match based on feature values.
    
    This function implements a realistic model where:
    - Higher name similarity increases true match probability
    - DOB matches increase true match probability
    - Higher country risk increases true match probability
    - PEP flags increase true match probability
    - Higher prior FP rates decrease true match probability
    - WORLDCHECK entries have slightly higher true match probability
    - Non-name alerts have slightly higher true match probability
    
    Args:
        name_similarity: Name similarity scores
        dob_match: Date of birth match indicators
        country_risk: Country risk levels (1-3)
        pep_flag: PEP flag indicators
        prior_name_fp_rate: Prior false positive rates
        list_type: List type categories
        alert_type: Alert type categories
        
    Returns:
        Array of true match probabilities
    """
    # Start with base probability
    probability_logits = BASE_TRUE_MATCH_PROBABILITY
    
    # Add weighted feature contributions
    probability_logits += NAME_SIMILARITY_WEIGHT * name_similarity
    probability_logits += DOB_MATCH_WEIGHT * dob_match
    probability_logits += COUNTRY_RISK_WEIGHT * (country_risk - 1)  # Normalize to 0-2 scale
    probability_logits += PEP_FLAG_WEIGHT * pep_flag
    probability_logits += PRIOR_FP_RATE_WEIGHT * prior_name_fp_rate
    
    # Add categorical feature bonuses
    probability_logits += np.where(list_type == "WORLDCHECK", WORLDCHECK_BONUS, 0)
    probability_logits += np.where(alert_type != "name", NON_NAME_ALERT_BONUS, 0)
    
    # Apply sigmoid function to convert to probabilities [0,1]
    true_match_probabilities = 1 / (1 + np.exp(-probability_logits))
    
    return true_match_probabilities


def generate_target_labels(true_match_probabilities: np.ndarray) -> np.ndarray:
    """
    Generate binary target labels based on true match probabilities.
    
    Args:
        true_match_probabilities: Array of true match probabilities
        
    Returns:
        Binary array of true match labels
    """
    return np.random.binomial(1, true_match_probabilities)


def create_dataframe(
    numerical_features: dict,
    categorical_features: dict,
    target_labels: np.ndarray
) -> pd.DataFrame:
    """
    Create a pandas DataFrame from generated features and labels.
    
    Args:
        numerical_features: Dictionary of numerical feature arrays
        categorical_features: Dictionary of categorical feature arrays
        target_labels: Array of target labels
        
    Returns:
        DataFrame containing all features and target labels
    """
    # Combine all features
    all_features = {**numerical_features, **categorical_features}
    
    # Round numerical features for cleaner display
    all_features['name_similarity'] = np.round(all_features['name_similarity'], DECIMAL_PRECISION)
    all_features['prior_name_fp_rate'] = np.round(all_features['prior_name_fp_rate'], DECIMAL_PRECISION)
    
    # Add target labels
    all_features['label_true_match'] = target_labels
    
    return pd.DataFrame(all_features)


def generate_synthetic_dataset(
    dataset_size: int = DEFAULT_DATASET_SIZE,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED
) -> pd.DataFrame:
    """
    Generate a synthetic dataset for sanctions false positive detection.
    
    This function creates a realistic dataset with the following features:
    - name_similarity: Similarity score between customer and watchlist names (0-1)
    - dob_match: Whether date of birth matches (0/1)
    - country_risk: Risk level of customer's country (1=low, 2=medium, 3=high)
    - pep_flag: Politically Exposed Person indicator (0/1)
    - prior_name_fp_rate: Historical false positive rate for this name (0-1)
    - list_type: Source of watchlist entry (WORLDCHECK/IWL)
    - alert_type: Type of alert trigger (name/dob/address)
    - label_true_match: Target variable indicating true match (0/1)
    
    Args:
        dataset_size: Number of records to generate (default: 1200)
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        DataFrame containing synthetic sanctions screening data
        
    Raises:
        ValueError: If dataset_size is not positive
    """
    if dataset_size <= 0:
        raise ValueError(f"Dataset size must be positive, got {dataset_size}")
    
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate features
    numerical_features = generate_numerical_features(dataset_size)
    categorical_features = generate_categorical_features(dataset_size)
    
    # Calculate true match probabilities based on features
    true_match_probabilities = calculate_true_match_probabilities(
        numerical_features['name_similarity'],
        numerical_features['dob_match'],
        numerical_features['country_risk'],
        numerical_features['pep_flag'],
        numerical_features['prior_name_fp_rate'],
        categorical_features['list_type'],
        categorical_features['alert_type']
    )
    
    # Generate target labels
    target_labels = generate_target_labels(true_match_probabilities)
    
    # Create and return DataFrame
    return create_dataframe(numerical_features, categorical_features, target_labels)


def get_dataset_statistics(dataset: pd.DataFrame) -> dict:
    """
    Calculate and return key statistics about the generated dataset.
    
    Args:
        dataset: Generated synthetic dataset
        
    Returns:
        Dictionary containing dataset statistics
    """
    stats = {
        'total_records': len(dataset),
        'true_match_rate': dataset['label_true_match'].mean(),
        'false_positive_rate': 1 - dataset['label_true_match'].mean(),
        'feature_correlations': dataset.select_dtypes(include=[np.number]).corr()['label_true_match'].to_dict()
    }
    
    return stats


if __name__ == "__main__":
    # Example usage and testing
    print("Generating synthetic dataset...")
    synthetic_data = generate_synthetic_dataset()
    
    print(f"Dataset shape: {synthetic_data.shape}")
    print(f"Features: {list(synthetic_data.columns)}")
    
    stats = get_dataset_statistics(synthetic_data)
    print(f"True match rate: {stats['true_match_rate']:.3f}")
    print(f"False positive rate: {stats['false_positive_rate']:.3f}")
    
    print("\nFirst 5 rows:")
    print(synthetic_data.head())
