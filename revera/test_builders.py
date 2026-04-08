"""
test_builders.py — Revera builder test suite
Run with: python test_builders.py
No Streamlit, no Groq, no tokens. Pure builders logic.
"""

import sys
import traceback
from builders import build_response

# ── HELPERS ────────────────────────────────────────────────────────────────

PASS = "✅"
FAIL = "❌"
results = []

def run(desc, parsed, *checks):
    """Run a single test case with one or more assertion functions."""
    try:
        result = build_response(parsed)
        for check_fn in checks:
            name = check_fn.__doc__ or check_fn.__name__
            assert check_fn(result), f"Check failed: '{name}' | keys={list(result.keys())}"
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

def has_chart(r):
    """has charts"""
    return "charts" in r and len(r["charts"]) > 0

def has_table(r):
    """has tables"""
    return "tables" in r and len(r["tables"]) > 0

def has_text(r):
    """has text"""
    return "text" in r and len(r["text"]) > 5

def no_error(r):
    """text does not start with warning"""
    return not r.get("text", "").startswith("⚠️")

def has_cards(r):
    """has cards"""
    return "cards" in r and len(r["cards"]) > 0

def has_export(r):
    """has export_df"""
    return "export_df" in r

def has_followups(r):
    """has followups"""
    return "followups" in r and len(r["followups"]) > 0

def table_n_rows(n):
    def check(r):
        f"""first table has {n} rows"""
        return "tables" in r and len(r["tables"][0]) == n
    check.__doc__ = f"first table has {n} rows"
    return check

def table_has_col(col):
    def check(r):
        return "tables" in r and col in r["tables"][0].columns
    check.__doc__ = f"table has column '{col}'"
    return check

def text_contains(substr):
    def check(r):
        return substr.lower() in r.get("text", "").lower()
    check.__doc__ = f"text contains '{substr}'"
    return check

# ── TESTS ──────────────────────────────────────────────────────────────────

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  REVERA — builders.py test suite")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# ── EXECUTIVE ──────────────────────────────────────────────────────────────
print("[ Executive ]")
run("executive summary",
    {"intent": "executive"},
    has_text, has_cards, has_chart, has_table, has_export,
    lambda r: "Volatility" in r["tables"][1].columns if len(r.get("tables",[])) > 1 else False,
    text_contains("forecast"))

# ── METRICS ───────────────────────────────────────────────────────────────
print("\n[ Metrics ]")
run("model metrics",
    {"intent": "metrics"},
    has_text, has_chart, has_table)

# ── OVERVIEW ──────────────────────────────────────────────────────────────
print("\n[ Overview ]")
run("overview BU default",
    {"intent": "overview", "level": "bu", "ids": []},
    has_chart, has_table, no_error)

run("overview segment default",
    {"intent": "overview", "level": "seg", "ids": []},
    has_chart, has_table, no_error)

run("overview subsegment default",
    {"intent": "overview", "level": "sub", "ids": []},
    has_chart, has_table, no_error)

run("overview BU single period 37",
    {"intent": "overview", "level": "bu", "ids": [], "periods": [0]},
    has_chart, has_table, no_error,
    table_has_col("P.37"))

run("overview BU single period 40",
    {"intent": "overview", "level": "bu", "ids": [], "periods": [3]},
    has_chart, has_table, no_error,
    table_has_col("P.40"))

run("overview subsegments of BU SSI037 (parent resolution)",
    {"intent": "overview", "level": "sub", "ids": ["SSI037"]},
    has_chart, has_table, no_error)

run("overview segments of BU SSI047 (parent resolution)",
    {"intent": "overview", "level": "seg", "ids": ["SSI047"]},
    has_chart, has_table, no_error)

run("overview subsegments of segment SSI03781",
    {"intent": "overview", "level": "sub", "ids": ["SSI03781"]},
    has_chart, has_table, no_error)

# ── RANKING ───────────────────────────────────────────────────────────────
print("\n[ Ranking ]")
run("top 5 BU by revenue",
    {"intent": "ranking", "level": "bu", "n": 5, "sort": "revenue", "ids": []},
    has_chart, has_table, table_n_rows(4),  # only 4 BUs exist
    table_has_col("CAGR"), no_error)

run("top 5 segments by revenue",
    {"intent": "ranking", "level": "seg", "n": 5, "sort": "revenue", "ids": []},
    has_chart, has_table, table_n_rows(5), no_error)

run("top 5 subsegments by revenue",
    {"intent": "ranking", "level": "sub", "n": 5, "sort": "revenue", "ids": []},
    has_chart, has_table, table_n_rows(5), no_error)

run("worst 5 segments by revenue",
    {"intent": "ranking", "level": "seg", "n": 5, "sort": "worst", "ids": []},
    has_chart, has_table, no_error)

run("highest growth segments",
    {"intent": "ranking", "level": "seg", "n": 5, "sort": "growth", "ids": []},
    has_chart, has_table, no_error)

run("worst growth segments",
    {"intent": "ranking", "level": "seg", "n": 5, "sort": "worst_growth", "ids": []},
    has_chart, has_table, no_error)

run("top 5 subsegments in BU SSI037 (parent filter)",
    {"intent": "ranking", "level": "sub", "n": 5, "sort": "revenue", "ids": ["SSI037"]},
    has_chart, has_table, no_error)

run("top 5 subsegments in segment SSI03782 (parent filter)",
    {"intent": "ranking", "level": "sub", "n": 5, "sort": "revenue", "ids": ["SSI03782"]},
    has_chart, has_table, no_error)

run("top 5 segments in BU SSI027 (parent filter)",
    {"intent": "ranking", "level": "seg", "n": 5, "sort": "revenue", "ids": ["SSI027"]},
    has_chart, has_table, no_error)

run("best growth segments in BU SSI037 (parent filter + growth)",
    {"intent": "ranking", "level": "seg", "n": 5, "sort": "growth", "ids": ["SSI037"]},
    has_chart, has_table, no_error)

run("ranking single period 40",
    {"intent": "ranking", "level": "bu", "n": 4, "sort": "revenue", "ids": [], "periods": [3]},
    has_chart, has_table, no_error,
    table_has_col("P.40"))

# ── DRILLDOWN ─────────────────────────────────────────────────────────────
print("\n[ Drilldown ]")
run("drilldown BU SSI027",
    {"intent": "drilldown", "level": "bu", "ids": ["SSI027"]},
    has_chart, has_table, no_error, has_followups)

run("drilldown BU SSI037",
    {"intent": "drilldown", "level": "bu", "ids": ["SSI037"]},
    has_chart, has_table, no_error)

run("drilldown BU SSI047",
    {"intent": "drilldown", "level": "bu", "ids": ["SSI047"]},
    has_chart, has_table, no_error)

run("drilldown BU SSI070",
    {"intent": "drilldown", "level": "bu", "ids": ["SSI070"]},
    has_chart, has_table, no_error)

run("drilldown segment SSI03781",
    {"intent": "drilldown", "level": "seg", "ids": ["SSI03781"]},
    has_chart, has_table, no_error, has_followups)

run("drilldown segment SSI02784",
    {"intent": "drilldown", "level": "seg", "ids": ["SSI02784"]},
    has_chart, has_table, no_error)

run("drilldown subsegment SSI0478150",
    {"intent": "drilldown", "level": "sub", "ids": ["SSI0478150"]},
    has_chart, has_table, no_error)

run("drilldown subsegment SSI0278002",
    {"intent": "drilldown", "level": "sub", "ids": ["SSI0278002"]},
    has_chart, has_table, no_error)

run("drilldown invalid ID → graceful error",
    {"intent": "drilldown", "level": "bu", "ids": ["INVALID123"]},
    has_text)  # should return error message, not crash

run("drilldown missing ID → graceful error",
    {"intent": "drilldown", "level": "bu", "ids": []},
    has_text)

# ── COMPARE ───────────────────────────────────────────────────────────────
print("\n[ Compare ]")
run("compare top 2 BUs by revenue",
    {"intent": "compare", "level": "bu", "ids": [], "sort": "revenue"},
    has_chart, has_table, no_error)

run("compare worst 2 BUs",
    {"intent": "compare", "level": "bu", "ids": [], "sort": "worst"},
    has_chart, has_table, no_error)

run("compare top 2 segments by revenue",
    {"intent": "compare", "level": "seg", "ids": [], "sort": "revenue"},
    has_chart, has_table, no_error)

run("compare worst 2 segments",
    {"intent": "compare", "level": "seg", "ids": [], "sort": "worst"},
    has_chart, has_table, no_error)

run("compare best growth 2 segments",
    {"intent": "compare", "level": "seg", "ids": [], "sort": "growth"},
    has_chart, has_table, no_error)

run("compare worst growth 2 segments",
    {"intent": "compare", "level": "seg", "ids": [], "sort": "worst_growth"},
    has_chart, has_table, no_error)

run("compare explicit 2 BUs",
    {"intent": "compare", "level": "bu", "ids": ["SSI037", "SSI027"], "sort": "revenue"},
    has_chart, has_table, no_error)

run("compare all BUs → redirects to overview",
    {"intent": "compare", "level": "bu", "ids": [], "n": 4, "sort": "revenue"},
    has_chart, has_table, no_error)

# ── PERIOD DIFF ───────────────────────────────────────────────────────────
print("\n[ Period diff ]")
run("period diff 37 vs 42",
    {"intent": "period_diff", "level": "bu", "ids": [], "periods": [0, 5]},
    has_chart, has_table, has_cards, no_error)

run("period diff 38 vs 40",
    {"intent": "period_diff", "level": "bu", "ids": [], "periods": [1, 3]},
    has_chart, has_table, has_cards, no_error)

run("period diff adjacent periods",
    {"intent": "period_diff", "level": "bu", "ids": [], "periods": [2, 3]},
    has_chart, has_table, no_error)

# ── TREND ─────────────────────────────────────────────────────────────────
print("\n[ Trend ]")
run("trend all BUs",
    {"intent": "trend", "level": "bu", "ids": []},
    has_chart, has_table, has_text, no_error)

run("trend all segments",
    {"intent": "trend", "level": "seg", "ids": []},
    has_chart, has_table, no_error)

run("trend specific BU SSI037",
    {"intent": "trend", "level": "bu", "ids": ["SSI037"]},
    has_chart, has_table, no_error)

run("trend subsegments of SSI03781",
    {"intent": "trend", "level": "sub", "ids": ["SSI03781"]},
    has_chart, has_table, no_error)

# ── ANOMALY ───────────────────────────────────────────────────────────────
print("\n[ Anomaly ]")
run("anomaly all BUs",
    {"intent": "anomaly", "level": "bu", "ids": []},
    has_text, no_error)

run("anomaly all segments",
    {"intent": "anomaly", "level": "seg", "ids": []},
    has_text, no_error)

run("anomaly specific BU SSI037",
    {"intent": "anomaly", "level": "bu", "ids": ["SSI037"]},
    has_text, no_error)

# ── ERROR HANDLING ─────────────────────────────────────────────────────────
print("\n[ Error handling ]")
run("error intent passes through",
    {"intent": "error", "error_msg": "Test error message"},
    has_text,
    text_contains("Test error message"))

run("unknown intent → overview fallback",
    {"intent": "unknown_xyz", "level": "bu", "ids": []},
    has_text, no_error)

run("missing all keys → no crash",
    {},
    has_text)

# ── SUMMARY ───────────────────────────────────────────────────────────────
total   = len(results)
passed  = sum(1 for ok, _, _ in results if ok)
failed  = total - passed

print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  {passed}/{total} passed   {failed} failed")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

if failed:
    print("Failed tests:")
    for ok, desc, err in results:
        if not ok:
            print(f"  {FAIL} {desc}")
            if err:
                print(f"       {err}")
    sys.exit(1)
