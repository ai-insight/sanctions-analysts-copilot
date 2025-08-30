import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
NUM_FEATURES = ["name_similarity", "dob_match", "country_risk", "pep_flag", "prior_name_fp_rate"]
CAT_FEATURES = ["list_type", "alert_type"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


# =========================
# Data utilities
# =========================
def load_data(use_sample: bool, uploaded_file) -> pd.DataFrame:
    """Load dataframe from sample generator or uploaded CSV."""
    if use_sample or uploaded_file is None:
        return generate_synthetic_dataset()
    return pd.read_csv(uploaded_file)


# =========================
# Modeling utilities
# =========================
def build_pipeline() -> Pipeline:
    """Create the preprocessing + model pipeline."""
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )
    clf = LogisticRegression(max_iter=200, class_weight="balanced", solver="liblinear")
    return Pipeline([("prep", pre), ("clf", clf)])


def train_and_evaluate(df: pd.DataFrame, test_size: float, threshold: float, random_state: int = 7):
    """Train the pipeline, compute predictions and evaluation metrics."""
    X = df[ALL_FEATURES]
    y = df["label_true_match"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    cm = confusion_matrix(y_test, y_pred)

    metrics_df = pd.DataFrame(
        {"Metric": ["Precision", "Recall", "F1"], "Score": [precision, recall, f1]}
    )

    return pipe, X_test, y_test, proba, y_pred, cm, metrics_df


def get_coeff_table(pipe: Pipeline) -> pd.DataFrame:
    """Extract a tidy coefficients table with odds ratios."""
    clf = pipe.named_steps["clf"]
    encoder = pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = encoder.get_feature_names_out(CAT_FEATURES)
    all_names = np.concatenate([np.array(NUM_FEATURES), cat_names])

    coef = clf.coef_[0]
    table = pd.DataFrame(
        {
            "Feature": all_names,
            "Coefficient": coef,
            "Odds Ratio (exp(coef))": np.exp(coef),
        }
    ).sort_values(by="Coefficient", ascending=False)
    return table


def build_scored_test_df(X_test: pd.DataFrame, y_test: pd.Series, proba: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Combine test features, ground truth, and predictions into one dataframe."""
    out = X_test.copy()
    out["y_test"] = y_test.values
    out["proba_true_match"] = proba
    out["predicted_label"] = y_pred
    return out.reset_index(drop=True)


def explain_alert(row: pd.Series, proba_true: float) -> str:
    """Generate a simple analyst note from raw feature values."""
    bits = []
    if row["name_similarity"] >= 0.85:
        bits.append("very high name similarity")
    elif row["name_similarity"] >= 0.6:
        bits.append("moderate name similarity")
    else:
        bits.append("low name similarity")

    if int(row["dob_match"]) == 1:
        bits.append("DOB matches")

    if int(row["country_risk"]) == 3:
        bits.append("high-risk country")
    elif int(row["country_risk"]) == 2:
        bits.append("medium-risk country")

    if int(row["pep_flag"]) == 1:
        bits.append("PEP flag present")

    if row["prior_name_fp_rate"] >= 0.7:
        bits.append("name has high false-positive history")

    if row["list_type"] == "IWL":
        bits.append("from Internal Watch List")
    else:
        bits.append("from World-Check")

    return f"Analyst note: Probability={proba_true:.0%}. Signals: " + ", ".join(bits)


# =========================
# UI utilities
# =========================
def render_confusion_matrix(cm: np.ndarray):
    """Plot a small confusion matrix heatmap."""
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred FP", "Pred True"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual FP", "Actual True"])
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")
    st.pyplot(fig)


def render_single_alert_controls() -> pd.DataFrame:
    """Render controls for single-alert testing and return a 1-row DataFrame."""
    row = {
        "name_similarity": st.slider("name_similarity", 0.0, 1.0, 0.5, 0.01),
        "dob_match": st.selectbox("dob_match", [0, 1]),
        "country_risk": st.selectbox("country_risk", [1, 2, 3]),
        "pep_flag": st.selectbox("pep_flag", [0, 1]),
        "prior_name_fp_rate": st.slider("prior_name_fp_rate", 0.0, 1.0, 0.7, 0.01),
        "list_type": st.selectbox("list_type", ["WORLDCHECK", "IWL"]),
        "alert_type": st.selectbox("alert_type", ["name", "dob", "address"]),
    }
    return pd.DataFrame([row])


# =========================
# Main app
# =========================
def main():
    st.set_page_config(page_title="Sanctions False-Positive Reducer", layout="wide")
    st.title("🛡️ Sanctions False-Positive Reducer (Demo)")
    st.write("A simple ML demo to **prioritise likely false positives** and **assist analysts**.")

    # Sidebar
    st.sidebar.header("Controls")
    use_sample = st.sidebar.checkbox("Use built-in synthetic dataset", value=True)
    uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
    test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.25, 0.05)
    threshold = st.sidebar.slider("True-Match Threshold", 0.2, 0.8, 0.5, 0.05)

    # Data
    df = load_data(use_sample, uploaded)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(12))

    # Train & evaluate
    pipe, X_test, y_test, proba, y_pred, cm, metrics_df = train_and_evaluate(
        df=df, test_size=test_size, threshold=threshold
    )

    # Metrics & CM
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Metrics (True_Match class)")
        st.write(metrics_df)
    with c2:
        st.markdown("### Confusion Matrix")
        render_confusion_matrix(cm)

    # Coefficients table (+ download)
    st.markdown("---")
    st.markdown("### Model Coefficients (Explainability)")
    coef_table = get_coeff_table(pipe)
    st.dataframe(coef_table, use_container_width=True)

    coef_csv = coef_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Coefficients (CSV)",
        data=coef_csv,
        file_name="model_coefficients.csv",
        mime="text/csv",
    )

    st.caption(
        "Notes: Positive coefficients push probability **up**; negative push it **down**. "
        "Odds Ratio > 1 means feature increases odds; < 1 decreases odds. "
        "Categorical features are one-hot encoded (e.g., alert_type=name is a separate column)."
    )

    # Single alert tester
    st.markdown("---")
    st.subheader("Try a Single Alert")
    single = render_single_alert_controls()

    prob_true = pipe.predict_proba(single)[0, 1]
    pred = int(prob_true >= threshold)
    st.write(
        f"**True-Match probability:** {prob_true:.0%} → **Prediction:** "
        f"{'True Match' if pred else 'False Positive'} (threshold={threshold:.2f})"
    )
    st.info(explain_alert(single.iloc[0], prob_true))

    # Scored test set (+ download)
    st.markdown("---")
    st.markdown("### Scored Test Set")
    scored = build_scored_test_df(X_test, y_test, proba, y_pred)
    st.dataframe(scored.head(12), use_container_width=True)

    scored_csv = scored.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Scored Test Set (CSV)",
        data=scored_csv,
        file_name="scored_test_set.csv",
        mime="text/csv",
    )

    st.caption("⚠️ Demo only. Not for regulatory use.")


if __name__ == "__main__":
    main()
