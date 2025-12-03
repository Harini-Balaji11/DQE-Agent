# src/llm_agent.py
import os
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _build_context_summary(dq_report: Dict[str, Any], anomaly_report: Dict[str, Any]) -> str:
    lines = []

    missing = dq_report.get("missingness")
    if missing is not None:
        top_missing = missing[missing > 0].sort_values(ascending=False).head(5)
        lines.append("Top columns by missing values:")
        for col, frac in top_missing.items():
            lines.append(f"- {col}: {frac:.2%} missing")

    dup_df = dq_report.get("duplicates")
    if dup_df is not None:
        lines.append(f"Duplicate rows count: {len(dup_df)}")

    for key in ("negative_fares", "zero_fares", "negative_duration",
                "too_long_duration", "non_numeric_total_charges"):
        if key in dq_report:
            val = dq_report[key]
            if hasattr(val, "__len__"):
                lines.append(f"{key}: {len(val)} rows flagged.")

    if anomaly_report and anomaly_report.get("anomaly_mask") is not None:
        n_anom = anomaly_report["anomaly_mask"].sum()
        lines.append(f"IsolationForest anomalies detected: {n_anom}")

    return "\n".join(lines)

def ask_llm_for_analysis(
    dq_report: Dict[str, Any],
    anomaly_report: Dict[str, Any],
    dataset_key: str,
    model: str = "gpt-4.1-mini"
) -> str:
    context = _build_context_summary(dq_report, anomaly_report)
    prompt = f"""
You are an expert Data Quality Engineer.

You are given a high-level summary of data quality issues for the dataset: {dataset_key}.

Summary:
{context}

Tasks:
1. Provide a concise root-cause analysis of the main data quality problems.
2. Explain how these issues might impact downstream analytics or ML models.
3. Suggest concrete remediation strategies.
4. Show example Python (pandas) and SQL snippets to fix the issues.

Keep the answer structured with headings and bullet points.
"""
    response = _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.2,
    )
    return response.choices[0].message.content
