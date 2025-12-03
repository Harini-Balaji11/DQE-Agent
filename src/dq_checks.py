# src/dq_checks.py
import pandas as pd
from typing import Dict, Any

def compute_missingness(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean()  # fraction missing per column

def find_duplicates(df: pd.DataFrame, id_column: str | None = None) -> pd.DataFrame:
    if id_column and id_column in df.columns:
        dup = df[df.duplicated(subset=[id_column], keep=False)]
    else:
        dup = df[df.duplicated(keep=False)]
    return dup

def basic_numeric_outliers(df: pd.DataFrame, std_threshold: float = 3.0) -> Dict[str, pd.Series]:
    outlier_mask = {}
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        series = df[col]
        if series.std() == 0 or series.isna().all():
            continue
        z = (series - series.mean()) / series.std()
        outlier_mask[col] = z.abs() > std_threshold
    return outlier_mask

def nyc_taxi_specific_checks(df: pd.DataFrame) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    if "fare_amount" in df.columns:
        results["negative_fares"] = df[df["fare_amount"] < 0]
        results["zero_fares"] = df[df["fare_amount"] == 0]
    if {"tpep_pickup_datetime", "tpep_dropoff_datetime"}.issubset(df.columns):
        df_tmp = df.copy()
        df_tmp["pickup"] = pd.to_datetime(df_tmp["tpep_pickup_datetime"], errors="coerce")
        df_tmp["dropoff"] = pd.to_datetime(df_tmp["tpep_dropoff_datetime"], errors="coerce")
        df_tmp["trip_duration_min"] = (df_tmp["dropoff"] - df_tmp["pickup"]).dt.total_seconds() / 60
        results["negative_duration"] = df_tmp[df_tmp["trip_duration_min"] < 0]
        results["too_long_duration"] = df_tmp[df_tmp["trip_duration_min"] > 240]  # > 4 hours
    return results

def telco_specific_checks(df: pd.DataFrame) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    if "TotalCharges" in df.columns:
        coerced = pd.to_numeric(df["TotalCharges"], errors="coerce")
        results["non_numeric_total_charges"] = df[coerced.isna()]
    return results

def run_dq_scan(
    df: pd.DataFrame,
    dataset_key: str,
    config: dict
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    ds_cfg = config["datasets"][dataset_key]
    id_col = ds_cfg.get("id_column")
    rules_cfg = config["dq_rules"]

    missingness = compute_missingness(df)
    report["missingness"] = missingness.sort_values(ascending=False)

    duplicates = find_duplicates(df, id_col)
    report["duplicates"] = duplicates

    outlier_mask = basic_numeric_outliers(df, std_threshold=rules_cfg["outlier_std_threshold"])
    report["std_outliers"] = outlier_mask

    if dataset_key == "nyc_taxi":
        spec = nyc_taxi_specific_checks(df)
        report.update(spec)
    elif dataset_key == "telco_churn":
        spec = telco_specific_checks(df)
        report.update(spec)

    return report
