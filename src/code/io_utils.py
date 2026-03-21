import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

def load_named_sheets(filepath, name_map: dict):
    raw = pd.read_excel(filepath, sheet_name=list(name_map.keys()), engine="calamine")
    return {new: raw[old] for old, new in name_map.items()}