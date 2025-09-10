"""
Configuration constants for the Sanctions False Positive Detection application.

This module contains all configuration parameters, thresholds, and feature definitions
used throughout the application.
"""

# =========================
# Feature Interpretation Thresholds
# =========================
NAME_SIMILARITY_HIGH_THRESHOLD = 0.85
NAME_SIMILARITY_MODERATE_THRESHOLD = 0.6
PRIOR_FP_RATE_HIGH_THRESHOLD = 0.7

# =========================
# Risk Level Constants
# =========================
COUNTRY_RISK_HIGH = 3
COUNTRY_RISK_MEDIUM = 2
DOB_MATCH_TRUE = 1
PEP_FLAG_PRESENT = 1

# =========================
# Model Configuration
# =========================
DEFAULT_RANDOM_STATE = 7
MAX_ITERATIONS = 200

# =========================
# Feature Definitions
# =========================
NUM_FEATURES = [
    "name_similarity", 
    "dob_match", 
    "country_risk", 
    "pep_flag", 
    "prior_name_fp_rate"
]

CAT_FEATURES = [
    "list_type", 
    "alert_type"
]

ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

# =========================
# Data Schema
# =========================
TARGET_COLUMN = "label_true_match"

# =========================
# UI Configuration
# =========================
APP_TITLE = "🛡️ Sanctions False-Positive Reducer (Demo)"
APP_DESCRIPTION = "A machine learning demo to **prioritize likely false positives** and **assist analysts** in sanctions screening."

# Default slider values
DEFAULT_TEST_SIZE = 0.25
DEFAULT_THRESHOLD = 0.5
DEFAULT_NAME_SIMILARITY = 0.5
DEFAULT_PRIOR_FP_RATE = 0.7