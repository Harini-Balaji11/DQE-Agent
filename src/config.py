# src/config.py
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config(path: str = "config.yaml") -> dict:
    cfg_path = PROJECT_ROOT / path
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)
