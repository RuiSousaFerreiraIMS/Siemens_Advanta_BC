from functools import lru_cache
import pandas as pd
from src.code.io_utils import load_named_sheets, DATA_RAW

@lru_cache(maxsize=None)
def load_all_data():
    extract = load_named_sheets(DATA_RAW / "Case2_data_extract_share.xlsx", {
        "training":   "training",
        "BUs":        "BUs",
        "validation": "validation"
    })
    market = load_named_sheets(DATA_RAW / "Case2_market_data_share.xlsx", {
        "Sheet1": "macro_data",
        "Sheet2": "period_data"
    })
    return {**extract, **market}

def load_training():    return load_all_data()["training"]
def load_bus():         return load_all_data()["BUs"]
def load_validation():  return load_all_data()["validation"]
def load_macro_data():  return load_all_data()["macro_data"]
def load_period_data(): return load_all_data()["period_data"]

def load_bus_parsed():
    from src.code.data_preparation import parse_bus
    return parse_bus(load_bus())

# ─────────────────────────────────────────────────────────────────────────────
# PREPARED DATA LOADERS (parquet files from data/prepared/)
# ─────────────────────────────────────────────────────────────────────────────

from src.code.io_utils import PROJECT_ROOT

DATA_PREPARED = PROJECT_ROOT / "data" / "prepared"

def load_macro_data_clean():
    return pd.read_parquet(DATA_PREPARED / "macro_data_clean.parquet")

def load_training_bu():
    return pd.read_parquet(DATA_PREPARED / "training_bu.parquet")


def load_training_segment():
    return pd.read_parquet(DATA_PREPARED / "training_segment.parquet")

def load_training_subsegment():
    return pd.read_parquet(DATA_PREPARED / "training_subsegment.parquet")

def load_training_total():
    return pd.read_parquet(DATA_PREPARED / "training_total.parquet")

def load_validation_bu():
    return pd.read_parquet(DATA_PREPARED / "validation_bu.parquet")

def load_validation_segment():
    return pd.read_parquet(DATA_PREPARED / "validation_segment.parquet")

def load_validation_subsegment():
    return pd.read_parquet(DATA_PREPARED / "validation_subsegment.parquet")

def load_validation_total():
    return pd.read_parquet(DATA_PREPARED / "validation_total.parquet")