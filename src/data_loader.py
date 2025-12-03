# src/data_loader.py
from pathlib import Path
import pandas as pd
from .config import PROJECT_ROOT

def load_dataset(config: dict, dataset_key: str) -> pd.DataFrame:
    ds_cfg = config["datasets"][dataset_key]
    path = PROJECT_ROOT / ds_cfg["path"]
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Place the CSV there.")
    df = pd.read_csv(path)

    # If NYC taxi dataset doesn't have trip_id, create one
    if dataset_key == "nyc_taxi" and "trip_id" not in df.columns:
        df.insert(0, "trip_id", range(1, len(df) + 1))

    return df
