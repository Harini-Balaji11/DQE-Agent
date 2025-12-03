# src/anomaly_detector.py
from typing import Dict, Any
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def _prepare_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num_df = df.select_dtypes(include="number").copy()
    num_df = num_df.dropna(axis=1, how="all")
    return num_df

def run_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42
) -> Dict[str, Any]:
    num_df = _prepare_numeric(df)
    if num_df.empty:
        return {"anomaly_scores": None, "anomaly_mask": None}
    scaler = StandardScaler()
    X = scaler.fit_transform(num_df)

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )
    iso.fit(X)
    scores = iso.decision_function(X)
    preds = iso.predict(X)  # -1 = anomaly, 1 = normal

    anomaly_mask = preds == -1

    return {
        "anomaly_scores": scores,
        "anomaly_mask": anomaly_mask,
    }

def run_anomaly_detection(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    ad_cfg = config["anomaly_detection"]
    method = ad_cfg.get("method", "isolation_forest")
    if method == "isolation_forest":
        return run_isolation_forest(
            df,
            contamination=ad_cfg.get("contamination", 0.03),
            random_state=ad_cfg.get("random_state", 42)
        )
    else:
        raise NotImplementedError(f"Anomaly detection method {method} not implemented.")
