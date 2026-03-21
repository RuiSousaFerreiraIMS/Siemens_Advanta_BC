from functools import lru_cache
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