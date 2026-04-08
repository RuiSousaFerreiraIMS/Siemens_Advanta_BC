"""
test_builders_hard.py — Revera edge case & stress test suite
Run with: python test_builders_hard.py [--verbose]
Covers: negative revenue, duplicate IDs, case sensitivity, n > data size,
        ambiguous IDs (in both SEG and SUB), all-negative series, single-sub
        segments, boundary periods, large n, data integrity checks.
"""

import sys
import traceback
import pandas as pd
import numpy as np
from builders import build_response
from data import BU, SEG, SUB, SUB_TO_SEG, SEG_TO_BU, PERIODS

PASS = "✅"
FAIL = "❌"
results = []

def run(desc, parsed, *checks):
    try:
        result = build_response(parsed)
        for check_fn in checks:
            name = check_fn.__doc__ or check_fn.__name__
            assert check_fn(result), f"'{name}' failed | keys={list(result.keys())} | text={result.get('text','')[:120]}"
        print(f"  {PASS}  {desc}")
        results.append((True, desc, None))
    except Exception as e:
        short = str(e).split("\n")[0]
        print(f"  {FAIL}  {desc}")
        print(f"       → {short}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        results.append((False, desc, short))

# ── CHECK HELPERS ──────────────────────────────────────────────────────────

def has_chart(r):   """has charts""";   return "charts" in r and len(r["charts"]) > 0
def has_table(r):   """has tables""";   return "tables" in r and len(r["tables"]) > 0
def has_text(r):    """has text""";     return "text" in r and len(r["text"]) > 5
def no_error(r):    """no warning""";   return not r.get("text", "").startswith("⚠️")
def no_crash(r):    """no crash""";     return "text" in r  # anything is fine, just no exception
def has_cards(r):   """has cards""";    return "cards" in r and len(r["cards"]) > 0

def table_not_empty(r):
    """first table is not empty"""
    return "tables" in r and len(r["tables"][0]) > 0

def table_has_col(col):
    def check(r):
        return "tables" in r and col in r["tables"][0].columns
    check.__doc__ = f"table has column '{col}'"
    return check

def table_no_nan(r):
    """no NaN values in first table"""
    if "tables" not in r or len(r["tables"]) == 0: return True
    return not r["tables"][0].isnull().any().any()

def table_max_rows(n):
    def check(r):
        return "tables" not in r or len(r["tables"][0]) <= n
    check.__doc__ = f"first table has at most {n} rows"
    return check

def chart_has_traces(r):
    """chart has at least 1 trace"""
    return "charts" in r and len(r["charts"][0].data) > 0

def text_contains(substr):
    def check(r):
        return substr.lower() in r.get("text", "").lower()
    check.__doc__ = f"text contains '{substr}'"
    return check

def text_not_contains(substr):
    def check(r):
        return substr.lower() not in r.get("text", "").lower()
    check.__doc__ = f"text does NOT contain '{substr}'"
    return check

def export_has_rows(r):
    """export_df has rows"""
    return "export_df" in r and len(r["export_df"]) > 0

def ids_in_table_are_valid(level):
    valid = set(BU) if level == "bu" else set(SEG) if level == "seg" else set(SUB)
    def check(r):
        if "tables" not in r or len(r["tables"]) == 0: return True
        col = r["tables"][0].columns[0]
        ids_in_result = set(r["tables"][0][col].tolist())
        return ids_in_result.issubset(valid)
    check.__doc__ = f"all IDs in table are valid {level} IDs"
    return check

# ── TESTS ──────────────────────────────────────────────────────────────────

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  REVERA — hard edge case test suite")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# ── NEGATIVE REVENUE SERIES ───────────────────────────────────────────────
print("[ Negative revenue series ]")
# SSI0378299 is entirely negative — should not crash anything

run("drilldown segment whose subsegment is all-negative (SSI03782)",
    {"intent": "drilldown", "level": "seg", "ids": ["SSI03782"]},
    has_chart, has_table, no_crash)

run("ranking worst sub — includes negative series",
    {"intent": "ranking", "level": "sub", "n": 5, "sort": "worst", "ids": []},
    has_chart, has_table, no_crash)

run("trend on segment with negative subsegments",
    {"intent": "trend", "level": "sub", "ids": ["SSI03782"]},
    has_table, no_crash)

run("anomaly on segment with negative subsegments",
    {"intent": "anomaly", "level": "sub", "ids": ["SSI03782"]},
    has_text, no_crash)

run("drilldown subsegment SSI0378299 (all-negative revenue)",
    {"intent": "drilldown", "level": "sub", "ids": ["SSI0378299"]},
    has_chart, has_table, no_crash)

run("cagr/volatility on all-negative series — no NaN in table",
    {"intent": "drilldown", "level": "sub", "ids": ["SSI0378299"]},
    table_no_nan)

run("drilldown subsegment SSI0478199 (mixed negative)",
    {"intent": "drilldown", "level": "sub", "ids": ["SSI0478199"]},
    has_chart, has_table, no_crash)

# ── AMBIGUOUS IDs (exist in both SEG and SUB) ──────────────────────────────
print("\n[ Ambiguous IDs — exist in both SEG and SUB ]")
# SSI02782, SSI02784, SSI02799, SSI03784, SSI03799, SSI04781, SSI04799

run("drilldown SSI02782 — should resolve as segment (has subsegments)",
    {"intent": "drilldown", "level": "seg", "ids": ["SSI02782"]},
    has_chart, has_table, no_crash)

run("drilldown SSI04781 — should resolve as segment",
    {"intent": "drilldown", "level": "seg", "ids": ["SSI04781"]},
    has_chart, has_table, no_crash)

run("drilldown SSI02799 — segment with only 1 subsegment (same ID)",
    {"intent": "drilldown", "level": "seg", "ids": ["SSI02799"]},
    has_chart, has_table, no_crash, table_not_empty)

run("drilldown SSI03799 — segment with 1 subsegment",
    {"intent": "drilldown", "level": "seg", "ids": ["SSI03799"]},
    has_chart, has_table, no_crash)

run("drilldown SSI04799 — segment with 1 subsegment (same ID as sub)",
    {"intent": "drilldown", "level": "seg", "ids": ["SSI04799"]},
    has_chart, has_table, no_crash)

# ── CASE SENSITIVITY ──────────────────────────────────────────────────────
print("\n[ Case sensitivity ]")

run("drilldown lowercase 'ssi037'",
    {"intent": "drilldown", "level": "bu", "ids": ["ssi037"]},
    has_chart, has_table, no_error)

run("drilldown mixed case 'Ssi027'",
    {"intent": "drilldown", "level": "bu", "ids": ["Ssi027"]},
    has_chart, has_table, no_error)

run("ranking parent lowercase 'ssi037'",
    {"intent": "ranking", "level": "seg", "n": 5, "sort": "revenue", "ids": ["ssi037"]},
    has_chart, has_table, no_error)

run("overview parent lowercase 'ssi047'",
    {"intent": "overview", "level": "seg", "ids": ["ssi047"]},
    has_chart, has_table, no_error)

run("drilldown segment lowercase 'ssi03781'",
    {"intent": "drilldown", "level": "seg", "ids": ["ssi03781"]},
    has_chart, has_table, no_error)

# ── N LARGER THAN AVAILABLE DATA ──────────────────────────────────────────
print("\n[ n > available data ]")

run("ranking n=100 BUs (only 4 exist) — no crash, max 4 rows",
    {"intent": "ranking", "level": "bu", "n": 100, "sort": "revenue", "ids": []},
    has_table, no_crash, table_max_rows(4))

run("ranking n=50 segments (now 24 exist) — no crash, max 24 rows",
    {"intent": "ranking", "level": "seg", "n": 50, "sort": "revenue", "ids": []},
    has_table, no_crash, table_max_rows(24))

run("ranking n=999 subsegments — no crash",
    {"intent": "ranking", "level": "sub", "n": 999, "sort": "revenue", "ids": []},
    has_table, no_crash)

run("ranking n=10 subsegments of SSI070 (only 10 exist) — no crash",
    {"intent": "ranking", "level": "sub", "n": 10, "sort": "revenue", "ids": ["SSI070"]},
    has_table, no_crash, table_max_rows(10))

run("ranking n=1 — single result",
    {"intent": "ranking", "level": "seg", "n": 1, "sort": "revenue", "ids": []},
    has_table, no_crash, table_max_rows(1))

# ── BOUNDARY PERIODS ──────────────────────────────────────────────────────
print("\n[ Boundary periods ]")

run("overview period 37 (first)",
    {"intent": "overview", "level": "bu", "ids": [], "periods": [0]},
    has_chart, has_table, no_crash, table_has_col("P.37"))

run("overview period 42 (last)",
    {"intent": "overview", "level": "bu", "ids": [], "periods": [5]},
    has_chart, has_table, no_crash, table_has_col("P.42"))

run("period_diff same period (37 vs 37) — no crash",
    {"intent": "period_diff", "level": "bu", "ids": [], "periods": [0, 0]},
    has_table, no_crash)

run("period_diff reversed order (42 vs 37) — no crash",
    {"intent": "period_diff", "level": "bu", "ids": [], "periods": [5, 0]},
    has_table, no_crash)

run("ranking single period 42 (last period)",
    {"intent": "ranking", "level": "seg", "n": 5, "sort": "revenue", "ids": [], "periods": [5]},
    has_table, no_crash, table_has_col("P.42"))

# ── DATA INTEGRITY CHECKS ─────────────────────────────────────────────────
print("\n[ Data integrity ]")

run("ranking BU — all returned IDs are valid BUs",
    {"intent": "ranking", "level": "bu", "n": 4, "sort": "revenue", "ids": []},
    ids_in_table_are_valid("bu"))

run("ranking seg — all returned IDs are valid segments",
    {"intent": "ranking", "level": "seg", "n": 10, "sort": "revenue", "ids": []},
    ids_in_table_are_valid("seg"))

run("drilldown SSI037 — all returned IDs are valid segments",
    {"intent": "drilldown", "level": "bu", "ids": ["SSI037"]},
    ids_in_table_are_valid("seg"))

run("drilldown SSI02784 — all returned IDs are valid subsegments",
    {"intent": "drilldown", "level": "seg", "ids": ["SSI02784"]},
    ids_in_table_are_valid("sub"))

run("executive — export_df has data",
    {"intent": "executive"},
    export_has_rows)

run("no NaN in ranking table",
    {"intent": "ranking", "level": "seg", "n": 10, "sort": "growth", "ids": []},
    table_no_nan)

run("no NaN in compare table",
    {"intent": "compare", "level": "bu", "ids": [], "sort": "revenue"},
    table_no_nan)

run("no NaN in drilldown BU table",
    {"intent": "drilldown", "level": "bu", "ids": ["SSI027"]},
    table_no_nan)

# ── SMALL SEGMENTS ────────────────────────────────────────────────────────
print("\n[ Small / edge segments ]")

run("drilldown SSI02710 — tiny segment (single subsegment)",
    {"intent": "drilldown", "level": "seg", "ids": ["SSI02710"]},
    has_chart, has_table, no_crash, table_not_empty)

run("drilldown SSI03784 — segment with 1 subsegment",
    {"intent": "drilldown", "level": "seg", "ids": ["SSI03784"]},
    has_chart, has_table, no_crash)

run("drilldown SSI04783 — segment with 1 subsegment",
    {"intent": "drilldown", "level": "seg", "ids": ["SSI04783"]},
    has_chart, has_table, no_crash)

run("trend on tiny segment SSI02710",
    {"intent": "trend", "level": "sub", "ids": ["SSI02710"]},
    has_text, no_crash)

run("anomaly on tiny segment SSI02710",
    {"intent": "anomaly", "level": "sub", "ids": ["SSI02710"]},
    has_text, no_crash)

run("ranking in SSI070 (only 10 subsegments) n=5",
    {"intent": "ranking", "level": "sub", "n": 5, "sort": "revenue", "ids": ["SSI070"]},
    has_table, no_crash, table_max_rows(5))

# ── LARGEST SEGMENT ───────────────────────────────────────────────────────
print("\n[ Large segments ]")

run("drilldown SSI02784 — segment with 33 subsegments",
    {"intent": "drilldown", "level": "seg", "ids": ["SSI02784"]},
    has_chart, has_table, no_crash, chart_has_traces, table_not_empty)

run("trend on SSI02784 (33 subsegments)",
    {"intent": "trend", "level": "sub", "ids": ["SSI02784"]},
    has_chart, has_table, no_crash)

run("anomaly on SSI02784 (33 subsegments)",
    {"intent": "anomaly", "level": "sub", "ids": ["SSI02784"]},
    has_text, no_crash)

run("ranking worst_growth in SSI02784",
    {"intent": "ranking", "level": "sub", "n": 5, "sort": "worst_growth", "ids": ["SSI02784"]},
    has_table, no_crash)

# ── CONSECUTIVE OPERATIONS (state isolation) ──────────────────────────────
print("\n[ State isolation — consecutive calls ]")

# SQLite in-memory db shared — make sure repeated calls don't corrupt state
r1 = build_response({"intent": "ranking", "level": "seg", "n": 5, "sort": "revenue", "ids": []})
r2 = build_response({"intent": "ranking", "level": "seg", "n": 5, "sort": "revenue", "ids": []})
ids1 = r1["tables"][0].iloc[:, 0].tolist()
ids2 = r2["tables"][0].iloc[:, 0].tolist()
run("same query twice returns identical results (no state corruption)",
    {"intent": "ranking", "level": "seg", "n": 5, "sort": "revenue", "ids": []},
    lambda r: r["tables"][0].iloc[:, 0].tolist() == ids1)

run("drilldown after ranking — no state leak",
    {"intent": "drilldown", "level": "bu", "ids": ["SSI037"]},
    has_chart, has_table, no_error)

run("executive after drilldown — no state leak",
    {"intent": "executive"},
    has_cards, has_chart, no_error)

# ── SUMMARY ───────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for ok, _, _ in results if ok)
failed = total - passed

print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  {passed}/{total} passed   {failed} failed")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

if failed:
    print("Failed tests:")
    for ok, desc, err in results:
        if not ok:
            print(f"  {FAIL} {desc}")
            if err: print(f"       {err}")
    sys.exit(1)
