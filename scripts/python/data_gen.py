import numpy as np
import pandas as pd

def generate_synthetic_dataset(N=1200, seed=42):
    np.random.seed(seed)

    # Numeric features
    name_similarity = np.clip(np.random.beta(2,2,N),0,1)
    dob_match = np.random.binomial(1,0.25,N)
    country_risk = np.random.choice([1,2,3], p=[0.6,0.3,0.1], size=N)  # 1=low,2=med,3=high
    pep_flag = np.random.binomial(1,0.1,N)
    prior_name_fp_rate = np.clip(np.random.normal(0.7,0.15,N),0,1)

    # Categorical features
    list_type = np.random.choice(["WORLDCHECK","IWL"], p=[0.8,0.2], size=N)
    alert_type = np.random.choice(["name","dob","address"], p=[0.7,0.2,0.1], size=N)

    # Probability of being a true match
    p_true = (
        0.05
        + 0.7*name_similarity
        + 0.25*dob_match
        + 0.12*(country_risk-1)
        + 0.2*pep_flag
        - 0.6*prior_name_fp_rate
    )
    p_true += np.where(list_type=="WORLDCHECK", 0.05, 0)
    p_true += np.where(alert_type!="name", 0.03, 0)
    p_true = 1/(1+np.exp(-p_true))  # squash to 0..1

    label_true_match = np.random.binomial(1, p_true)

    df = pd.DataFrame({
        "name_similarity": np.round(name_similarity,3),
        "dob_match": dob_match,
        "country_risk": country_risk,
        "pep_flag": pep_flag,
        "prior_name_fp_rate": np.round(prior_name_fp_rate,3),
        "list_type": list_type,
        "alert_type": alert_type,
        "label_true_match": label_true_match
    })

    return df
