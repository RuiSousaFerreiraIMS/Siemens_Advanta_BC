import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from data import (BU as _BU, SEG as _SEG, SUB as _SUB,
                  BU_HIST, BU_FC, SEG_HIST, SEG_FC, SUB_HIST, SUB_FC,
                  PERIODS_HIST, PERIODS_FORECAST, PERIOD_LABELS)

def plabel(p):
    """Return calendar label for period number, e.g. 42 → 'Sep/24'."""
    return PERIOD_LABELS.get(p, f"P.{p}")

PERIODS = PERIODS_HIST + PERIODS_FORECAST   # P.1–48 full range (charts/tables)
NLU_PERIODS = list(range(37, 49))           # P.37–48 — maps NLU 0-based index → period number

# --- SQLITE DB INIT ---
# Table 'forecast'  : in-sample P.37-42 (legacy, used by existing intent builders)
# Table 'hist'      : full history P.1-42
# Table 'forecast48': predictions P.43-48
conn = sqlite3.connect(":memory:", check_same_thread=False)

_LEGACY_PERIODS = list(range(37, 43))   # P.37–42 — the 6 in-sample periods backing the legacy 'forecast' table
rows = []
for lvl, data_dict in [("bu", _BU), ("seg", _SEG), ("sub", _SUB)]:
    for id_, vals in data_dict.items():
        for i, p in enumerate(_LEGACY_PERIODS):
            rows.append({"level": lvl, "id": id_, "period": p, "revenue": vals[i], "period_idx": i})
pd.DataFrame(rows).to_sql("forecast", conn, index=False)

hist_rows = []
for lvl, data_dict in [("bu", BU_HIST), ("seg", SEG_HIST), ("sub", SUB_HIST)]:
    for id_, vals in data_dict.items():
        for i, p in enumerate(PERIODS_HIST):
            hist_rows.append({"level": lvl, "id": id_, "period": p, "revenue": vals[i]})
pd.DataFrame(hist_rows).to_sql("hist", conn, index=False)

fc_rows = []
for lvl, data_dict in [("bu", BU_FC), ("seg", SEG_FC), ("sub", SUB_FC)]:
    for id_, vals in data_dict.items():
        for i, p in enumerate(PERIODS_FORECAST):
            fc_rows.append({"level": lvl, "id": id_, "period": p, "revenue": vals[i]})
pd.DataFrame(fc_rows).to_sql("forecast48", conn, index=False)

# Combined view spanning all periods P.1–48
conn.execute("""
    CREATE VIEW all_revenue AS
    SELECT level, id, period, revenue FROM hist
    UNION ALL
    SELECT level, id, period, revenue FROM forecast48
""")
# ----------------------

COLORS = ["#00e5b8", "#3d9eff", "#ffb547", "#ff6060", "#b47fff", "#ff9fcc", "#7fffd4", "#ffd700"]
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#10151e",
    plot_bgcolor="#10151e",
    font=dict(family="DM Mono", color="#6b7e96", size=12),
    margin=dict(l=10, r=10, t=36, b=10),
    height=280,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", tickfont=dict(size=11)),
)

# ── HELPERS ────────────────────────────────────────────────────────────────
def avg(v): return sum(v) / len(v)
def pct(a, b): return round((b - a) / abs(a) * 100, 1) if a != 0 else 0

def cagr(v):
    """True annualised CAGR — periods are months.
    Formula: (last/first)^(12/months) - 1
    Guards:
      - Uses first/last strictly positive values
      - Requires ≥ 6 months between anchors
      - Base must be ≥ 0.5% of peak (filters near-zero starts)
      - Last positive value must be within the last 12 months (filters inactive segments)
    """
    n = len(v)
    if n < 2:
        return 0.0
    peak = max(v)
    min_base = peak * 0.005
    first_i = next((i for i, x in enumerate(v) if x >= min_base and x > 0), None)
    last_i  = next((i for i, x in enumerate(reversed(v)) if x > 0), None)
    if first_i is None or last_i is None:
        return 0.0
    last_i = n - 1 - last_i
    if last_i < n - 12:              # segment inactive in last 12 months — skip
        return 0.0
    months = last_i - first_i
    if months < 6:
        return 0.0
    v0, vn = v[first_i], v[last_i]
    return round(((vn / v0) ** (12 / months) - 1) * 100, 1)

def volatility(v):
    """Coefficient of variation as % — std/mean."""
    m = avg(v)
    if m == 0: return 0.0
    return round(np.std(v) / abs(m) * 100, 1)

def trend_slope(v):
    """Linear regression slope normalised by mean — gives % change per period."""
    x = np.arange(len(v), dtype=float)
    if np.std(x) == 0 or np.std(v) == 0: return 0.0
    slope = np.polyfit(x, v, 1)[0]
    m = avg(v)
    return round(slope / abs(m) * 100, 1) if m != 0 else 0.0

def fmt(v):
    a = abs(v)
    if a >= 1e9: return f"{v/1e9:.2f}B €"
    if a >= 1e6: return f"{v/1e6:.1f}M €"
    if a >= 1e3: return f"{v/1e3:.0f}K €"
    return f"{v} €"

def fmt_s(v):
    a = abs(v)
    if a >= 1e9: return f"{v/1e9:.2f}B"
    if a >= 1e6: return f"{v/1e6:.1f}M"
    if a >= 1e3: return f"{v/1e3:.1f}K"
    return f"{v:.2f}"

def get_label(level):
    return "Subsegment" if level == "sub" else "Segment" if level == "seg" else "BU"

def fetch_ds(ids, level=None):
    if not ids: return {}
    placeholders = ",".join("?" * len(ids))
    lc = "level=? AND " if level else ""
    params = (level, *ids) if level else tuple(ids)
    df_hist = pd.read_sql(
        f"SELECT id, period, revenue FROM hist WHERE {lc}id IN ({placeholders})",
        conn, params=params
    )
    df_fc = pd.read_sql(
        f"SELECT id, period, revenue FROM forecast48 WHERE {lc}id IN ({placeholders})",
        conn, params=params
    )
    df = pd.concat([df_hist, df_fc], ignore_index=True)
    ds = {}
    for id_, group in df.groupby("id"):
        v = group.sort_values("period")["revenue"].tolist()
        ds[id_] = v[:len(PERIODS)]
    return ds

def make_line_chart(title, series: dict, highlight_period=None):
    fig = go.Figure()
    x_labels = [plabel(p) for p in PERIODS]
    for i, (name, values) in enumerate(series.items()):
        fig.add_trace(go.Scatter(
            x=x_labels, y=values, name=name,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            marker=dict(size=5), mode="lines+markers",
            customdata=[name]*len(values)
        ))
    # Forecast boundary between Sep/24 (P.42) and Oct/24 (P.43)
    if 43 in PERIODS:
        boundary_idx = PERIODS.index(42) + 0.5
        fig.add_vline(x=boundary_idx, line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dash"))
        fig.add_annotation(x=boundary_idx + 0.6, y=0, yref="paper", text="forecast →",
                           showarrow=False, font=dict(size=9, color="#6b7e96"), xanchor="left")
    if highlight_period is not None and highlight_period in PERIODS:
        fig.add_vline(
            x=PERIODS.index(highlight_period),
            line=dict(color="rgba(255,181,71,0.4)", width=1, dash="dot")
        )
    layout = {**PLOTLY_LAYOUT,
              "xaxis": {**PLOTLY_LAYOUT["xaxis"], "tickangle": -45, "nticks": 12}}
    fig.update_layout(title=dict(text=title, font=dict(size=12, color="#dce8f5")), **layout)
    return fig

def make_bar_chart(title, x_labels, values, color_fn=None):
    colors = [color_fn(v) if color_fn else COLORS[i % len(COLORS)] for i, v in enumerate(values)]
    rgba_colors = [f"rgba({int(c[1:3],16)}, {int(c[3:5],16)}, {int(c[5:7],16)}, 0.27)" for c in colors]
    fig = go.Figure(go.Bar(
        x=x_labels, y=values,
        marker=dict(color=rgba_colors, line=dict(color=colors, width=1.5)),
        text=[fmt_s(v) for v in values], textposition="outside",
        textfont=dict(size=10, color="#6b7e96")
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=12, color="#dce8f5")), **PLOTLY_LAYOUT)
    return fig

def make_table_df(ids, data, level_label, single_period=None, show_volatility=False):
    rows = []
    for id_ in ids:
        v = data.get(id_, [0]*len(PERIODS))
        if len(v) < len(PERIODS):
            v = v + [0] * (len(PERIODS) - len(v))  # pad to full length
        g = cagr(v)
        peak_idx = int(np.argmax(v)) if v and max(v) != 0 else 0
        peak_idx = min(peak_idx, len(PERIODS) - 1)  # clamp to valid range
        peak_period = PERIODS[peak_idx]
        row = {level_label: id_}
        if single_period is not None:
            period_idx = PERIODS.index(single_period)
            row[plabel(single_period)] = fmt(v[period_idx])
            row["CAGR"] = f"{'▲' if g >= 0 else '▼'} {abs(g)}%"
        else:
            row[plabel(PERIODS[0])]  = fmt(v[0])
            row[plabel(PERIODS[-1])] = fmt(v[-1])
            row["CAGR"] = f"{'▲' if g >= 0 else '▼'} {abs(g)}%"
            row["Peak"] = plabel(peak_period)
        row["Avg/period"] = fmt(avg(v))
        if show_volatility:
            row["Volatility"] = f"{volatility(v)}%"
        rows.append(row)
    return pd.DataFrame(rows)

def _suggest_followups(level, ids, sort=None):
    """Generate contextual follow-up suggestion chips."""
    suggestions = []
    if not ids:
        return suggestions
    top_id = ids[0]
    ll = get_label(level)
    if level == "bu":
        suggestions.append(f"Drill down into {top_id}")
        suggestions.append(f"Which segments grew the most in {top_id}?")
        suggestions.append("Compare all BUs")
    elif level == "seg":
        suggestions.append(f"Show subsegments of {top_id}")
        suggestions.append(f"Which subsegments in {top_id} have highest growth?")
        if len(ids) >= 2:
            suggestions.append(f"Compare {ids[0]} vs {ids[1]}")
    elif level == "sub":
        suggestions.append(f"Tell me about {top_id}")
        suggestions.append("Show the trend for these subsegments")
    return suggestions[:3]

# ── RESPONSE BUILDERS ──────────────────────────────────────────────────────
def build_response(parsed: dict) -> dict:
    intent    = parsed.get("intent", "overview")
    level     = parsed.get("level", "bu")
    ids       = parsed.get("ids", [])
    n         = parsed.get("n", 5)
    sort      = parsed.get("sort", "revenue")
    periods   = parsed.get("periods")
    threshold    = parsed.get("threshold")
    threshold_op = parsed.get("threshold_op")
    error_msg = parsed.get("error_msg", "")

    ll = get_label(level)

    if intent == "error":
        return {"text": f"⚠️ {error_msg}"}
    elif intent == "executive":
        return _executive()
    elif intent == "metrics":
        return _metrics()
    elif intent == "trend":
        return _trend(ids, level, ll)
    elif intent == "forecast":
        return _forecast(ids, level, ll)
    elif intent == "heatmap":
        return _heatmap(level, ll)
    elif intent == "period_diff" and periods:
        return _period_diff(periods[0], periods[1])
    elif intent == "drilldown":
        return _drilldown(ids[0] if ids else None)
    elif intent == "compare":
        if len(ids) > 2 or (not ids and n and n > 2):
            return _overview(level, ll, ids)
        return _compare(ids, level, ll, sort)
    elif intent == "ranking":
        single_period = None
        if periods and len(periods) == 1:
            single_period = NLU_PERIODS[periods[0]]
        parent_id = None
        if ids:
            from data import SEG_TO_BU as _S2B, SUB_TO_SEG as _S2S
            candidate = ids[0].upper()
            if level == "seg" and candidate in _S2B.values():
                parent_id = candidate
            elif level == "sub" and (candidate in _S2B.values() or candidate in _S2B):
                parent_id = candidate
        return _ranking(level, ll, n, sort, single_period=single_period, parent_id=parent_id,
                        threshold=threshold, threshold_op=threshold_op)
    else:
        single_period = None
        if periods and len(periods) == 1:
            single_period = NLU_PERIODS[periods[0]]
        parent_id = None
        if ids:
            from data import SEG_TO_BU as _S2B, SUB_TO_SEG as _S2S
            candidate = ids[0].upper()
            if level == "sub" and (candidate in _S2B.values() or candidate in _S2B):
                parent_id = candidate
                ids = []
            elif level == "seg" and candidate in _S2B.values():
                parent_id = candidate
                ids = []
        return _overview(level, ll, ids, single_period=single_period, parent_id=parent_id)


def _executive():
    from data import SEG_TO_BU

    # ── Full historical aggregate (P.1-42) ──
    hist_agg = pd.read_sql(
        "SELECT period, SUM(revenue) as val FROM hist WHERE level='bu' GROUP BY period ORDER BY period", conn
    )["val"].tolist()
    hist_cagr_val = cagr(hist_agg)
    hist_vol      = volatility(hist_agg)
    peak_idx      = int(np.argmax(hist_agg))
    peak_period   = PERIODS_HIST[min(peak_idx, len(PERIODS_HIST)-1)]

    # ── Forecast aggregate (P.43-48) ──
    fc_agg = pd.read_sql(
        "SELECT period, SUM(revenue) as val FROM forecast48 WHERE level='bu' GROUP BY period ORDER BY period", conn
    )["val"].tolist()
    total_fc_p48 = fc_agg[-1] if fc_agg else 0
    total_fc_p43 = fc_agg[0]  if fc_agg else 0
    fc_growth    = round((total_fc_p48/total_fc_p43-1)*100, 1) if total_fc_p43 else 0

    # ── Best/worst BU by forecast total ──
    bu_fc_df = pd.read_sql(
        "SELECT id, SUM(revenue) as total FROM forecast48 WHERE level='bu' GROUP BY id ORDER BY total DESC", conn)
    best_fc_bu  = bu_fc_df.iloc[0]["id"]

    # ── In-sample BU growth (P.37-P.42) for context ──
    bu_df = pd.read_sql("""
        SELECT p1.id,
               (p2.revenue - p1.revenue)*100.0 / ABS(p1.revenue) as growth_pct
        FROM (SELECT id, revenue FROM hist WHERE level='bu' AND period=37) p1
        JOIN (SELECT id, revenue FROM hist WHERE level='bu' AND period=42) p2 ON p1.id = p2.id
        ORDER BY growth_pct DESC
    """, conn)
    best_bu_g  = round(bu_df.iloc[0]["growth_pct"], 1)
    worst_bu_g = round(bu_df.iloc[-1]["growth_pct"], 1)

    # ── Most volatile segment (in-sample) ──
    all_seg_ids = list(_SEG.keys())
    ds_seg = fetch_ds(all_seg_ids, level="seg")
    vol_list = sorted([(id_, volatility(ds_seg[id_])) for id_ in all_seg_ids if id_ in ds_seg],
                      key=lambda x: x[1], reverse=True)
    most_volatile_seg, most_volatile_val = vol_list[0] if vol_list else ("—", 0)

    # ── Cards ──
    cards = [
        ("Forecast P.48",   fmt_s(total_fc_p48),                          "all BUs"),
        ("Forecast Growth",  f"{'+' if fc_growth>=0 else ''}{fc_growth}%", "P.43 → P.48"),
        ("Historical CAGR",  f"+{hist_cagr_val}%",                         "P.1 → P.42"),
        ("Top Forecast BU",  best_fc_bu,                                   f"{fmt_s(bu_fc_df.iloc[0]['total'])} total"),
    ]

    # ── Narrative ──
    narrative = (
        f"Siemens Advanta is forecast to reach **{fmt(total_fc_p48)}** by P.48, "
        f"a **+{fc_growth}%** increase over the forecast horizon (P.43–P.48). "
        f"Historically, aggregate revenue grew at **{hist_cagr_val}% CAGR** across 42 periods "
        f"(volatility: {hist_vol}% CV), peaking at **{plabel(peak_period)}**. "
        f"In-sample: **{bu_df.iloc[0]['id']}** led at +{best_bu_g}%, **{bu_df.iloc[-1]['id']}** trailed at {worst_bu_g}%. "
        f"Most volatile segment: **{most_volatile_seg}** (CV {most_volatile_val}%)."
    )

    # ── Historical + Forecast chart (blue/orange) ──
    fig_hist_fc = go.Figure()
    fig_hist_fc.add_trace(go.Scatter(
        x=[plabel(p) for p in PERIODS_HIST], y=hist_agg, name="Historical Revenue (Apr/21–Sep/24)",
        fill="tozeroy", line=dict(color="#3d9eff", width=1.5),
        fillcolor="rgba(61,158,255,0.3)"
    ))
    x_fc = [plabel(p) for p in [PERIODS_HIST[-1]] + PERIODS_FORECAST]
    y_fc = [hist_agg[-1]] + fc_agg
    fig_hist_fc.add_trace(go.Scatter(
        x=x_fc, y=y_fc, name="FlowState-r1.1 Forecast (Oct/24–Mar/25)",
        fill="tozeroy", line=dict(color="#ff6b35", width=2),
        fillcolor="rgba(255,107,53,0.35)",
        mode="lines+markers", marker=dict(size=6, color="#ff6b35"),
        text=[""] + [f"{v/1e6:.0f}M" for v in fc_agg],
        textposition="top center", textfont=dict(size=9, color="#ff6b35")
    ))
    fig_hist_fc.add_annotation(x=plabel(45), y=max(fc_agg)*0.88, text="← forecast",
                                xref="x", yref="y",
                                showarrow=False, font=dict(size=9, color="#6b7e96"))
    layout_fc = {**PLOTLY_LAYOUT, "height": 300,
                 "xaxis": dict(title="", tickangle=-45, nticks=12, gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)"),
                 "yaxis": dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", tickformat=".2s")}
    fig_hist_fc.update_layout(
        title=dict(text="Aggregate Revenue — Historical (Apr/21–Sep/24) & Forecast (Oct/24–Mar/25)", font=dict(size=12, color="#dce8f5")),
        **layout_fc
    )

    # ── BU in-sample chart ──
    all_bus = list(_BU.keys())
    ds_bu   = fetch_ds(all_bus, level="bu")
    fig_bu  = make_line_chart("Revenue by BU — P.1–48 (historical + forecast)",
                               {id_: ds_bu[id_] for id_ in all_bus if id_ in ds_bu})

    # ── Tables: forecast P.43-48 + in-sample summary ──
    fc_rows = []
    for bu in sorted(BU_FC):
        v = BU_FC[bu]
        g = cagr(v)
        fc_rows.append({"BU": bu,
                         **{plabel(p): fmt(v[i]) for i, p in enumerate(PERIODS_FORECAST)},
                         "CAGR": f"{'▲' if g>=0 else '▼'} {abs(g)}%"})
    df_fc_table = pd.DataFrame(fc_rows)
    df_insample = make_table_df(all_bus, ds_bu, "BU", show_volatility=True)

    export_df = pd.read_sql(
        "SELECT level, id, period, revenue FROM hist WHERE level='bu' ORDER BY id, period", conn)
    return {
        "text":      narrative,
        "cards":     cards,
        "charts":    [fig_hist_fc, fig_bu],
        "tables":    [df_fc_table, df_insample],
        "export_df": export_df,
        "export_pdf": True,
        "followups": [f"Drill down into {best_fc_bu}",
                      f"Which segments in {best_fc_bu} have highest forecast?",
                      "Show model performance metrics"]
    }


def _metrics():
    fig = make_bar_chart(
        "Model Performance — FlowState-r1.1 + Bottom-Up Aggregation",
        ["R²", "1 − wMAPE"],
        [0.9871, 1 - 0.1070],
        color_fn=lambda v: "#00e5b8"
    )
    fig.update_layout(yaxis=dict(range=[0.85, 1.0], tickformat=".3f"))
    df = pd.DataFrame([
        {"Metric": "Model",       "Value": "FlowState-r1.1"},
                {"Metric": "Reconciliation", "Value": "Bottom-Up"},
        {"Metric": "R²",          "Value": "0.9871"},
        {"Metric": "wMAPE",       "Value": "9.40%"},
        {"Metric": "RMSE",        "Value": "7,743,554 €"},
        {"Metric": "MAE",         "Value": "3,663,463 €"},
    ])
    return {
        "text": "Model performance — FlowState-r1.1, Bottom-Up aggregated. R² 0.9871 · wMAPE 9.40% at the subsegment level.",
        "charts": [fig],
        "tables": [df],
        "followups": ["Executive summary", "Show revenue overview by BU"]
    }


def _trend(ids, level, ll):
    """Trend analysis: slope + direction for each series."""
    from data import SEG_TO_BU as _S2B, SUB_TO_SEG as _S2S

    if not ids:
        # default: all BUs
        level = "bu"
        ll = "BU"
        ids = list(_BU.keys())

    # Resolve parent
    resolved = []
    for id_ in ids:
        id_up = id_.upper()
        if level == "sub":
            if id_up in _S2B.values():
                segs = [s for s, b in _S2B.items() if b == id_up]
                resolved += [sub for sub, seg in _S2S.items() if seg in segs]
            elif id_up in _S2B:
                resolved += [sub for sub, seg in _S2S.items() if seg == id_up]
            else:
                resolved.append(id_up)
        else:
            resolved.append(id_up)

    if not resolved:
        resolved = ids

    ds = fetch_ds(resolved, level=level)
    valid = [id_ for id_ in resolved if id_ in ds]

    rows = []
    for id_ in valid:
        v = ds[id_]
        s = trend_slope(v)
        c = cagr(v)
        direction = "↗ Growing" if s > 1 else ("↘ Declining" if s < -1 else "→ Flat")
        rows.append({
            ll: id_,
            "Trend": direction,
            "Slope %/period": f"{'+' if s >= 0 else ''}{s}%",
            "CAGR": f"{'+' if c >= 0 else ''}{c}%",
            "Volatility": f"{volatility(v)}%",
            "Avg/period": fmt(avg(v)),
        })
    df = pd.DataFrame(rows)

    # Trend chart: actual + linear fit per series
    fig = go.Figure()
    period_labels = [plabel(p) for p in PERIODS]
    x = np.arange(len(PERIODS), dtype=float)
    for i, id_ in enumerate(valid[:8]):
        v = ds[id_]
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=period_labels, y=v, name=id_,
            line=dict(color=color, width=2),
            marker=dict(size=4), mode="lines+markers",
            customdata=[id_]*len(v)
        ))
        # trend line
        if np.std(v) > 0:
            fit = np.polyfit(x, v, 1)
            fitted = np.polyval(fit, x).tolist()
            fig.add_trace(go.Scatter(
                x=period_labels, y=fitted, name=f"{id_} trend",
                line=dict(color=color, width=1, dash="dot"),
                showlegend=False
            ))
    fig.update_layout(title=dict(text=f"Trend analysis — {ll}s", font=dict(size=12, color="#dce8f5")), **PLOTLY_LAYOUT)

    growing   = [r[ll] for _, r in df.iterrows() if "Growing"  in r["Trend"]]
    declining = [r[ll] for _, r in df.iterrows() if "Declining" in r["Trend"]]
    flat      = [r[ll] for _, r in df.iterrows() if "Flat"      in r["Trend"]]
    summary_parts = []
    if growing:   summary_parts.append(f"**{len(growing)}** growing ({', '.join(growing[:3])}{'...' if len(growing)>3 else ''})")
    if declining: summary_parts.append(f"**{len(declining)}** declining ({', '.join(declining[:3])}{'...' if len(declining)>3 else ''})")
    if flat:      summary_parts.append(f"**{len(flat)}** flat")

    return {
        "text": f"Trend analysis across {len(valid)} {ll}s: {'; '.join(summary_parts)}.",
        "charts": [fig],
        "tables": [df],
        "followups": [f"Tell me about {growing[0]}" if growing else "Executive summary",
                      f"Tell me about {declining[0]}" if declining else "Compare all BUs"]
    }


def _period_diff(i1, i2):
    p1, p2 = NLU_PERIODS[i1], NLU_PERIODS[i2]
    # Use all_revenue view so both historical and forecast periods work
    total_p1 = pd.read_sql("SELECT SUM(revenue) as v FROM all_revenue WHERE level='bu' AND period=?", conn, params=(p1,)).iloc[0]["v"]
    total_p2 = pd.read_sql("SELECT SUM(revenue) as v FROM all_revenue WHERE level='bu' AND period=?", conn, params=(p2,)).iloc[0]["v"]
    total_diff = total_p2 - total_p1
    tg = pct(total_p1, total_p2)

    cards = [
        (f"{plabel(p1)}", fmt_s(total_p1), "all BUs"),
        (f"{plabel(p2)}", fmt_s(total_p2), "all BUs"),
        ("Delta", f"{'+' if total_diff >= 0 else ''}{fmt_s(total_diff)}", f"{'+' if tg >= 0 else ''}{tg}%"),
    ]

    diff_df = pd.read_sql("""
        SELECT a.id, a.revenue as r1, b.revenue as r2, (b.revenue - a.revenue) as delta
        FROM (SELECT id, revenue FROM all_revenue WHERE level='bu' AND period=?) a
        JOIN (SELECT id, revenue FROM all_revenue WHERE level='bu' AND period=?) b ON a.id = b.id
    """, conn, params=(p1, p2))

    fig = make_bar_chart(
        f"Revenue delta by BU: {plabel(p2)} − {plabel(p1)}",
        diff_df["id"].tolist(), diff_df["delta"].tolist(),
        color_fn=lambda v: "#00e5b8" if v >= 0 else "#ff6060"
    )

    rows = []
    for _, r in diff_df.iterrows():
        g = pct(r["r1"], r["r2"])
        rows.append({
            "BU": r["id"],
            plabel(p1): fmt(r["r1"]),
            plabel(p2): fmt(r["r2"]),
            "Δ Abs": f"{'+' if r['delta'] >= 0 else ''}{fmt_s(r['delta'])}",
            "Δ %": f"{'▲' if g >= 0 else '▼'} {abs(g)}%",
        })
    df = pd.DataFrame(rows)
    export_df = pd.read_sql("""
        SELECT a.id, a.revenue as p1_rev, b.revenue as p2_rev, (b.revenue - a.revenue) as delta
        FROM (SELECT id, revenue FROM all_revenue WHERE level='bu' AND period=?) a
        JOIN (SELECT id, revenue FROM all_revenue WHERE level='bu' AND period=?) b ON a.id = b.id
    """, conn, params=(p1, p2))
    export_df.columns = ["BU", plabel(p1), plabel(p2), "Delta"]

    all_bus = list(_BU.keys())
    ds_bu = fetch_ds(all_bus)
    fig_line = make_line_chart(
        f"Revenue timeline · {plabel(p1)} and {plabel(p2)} highlighted",
        {id_: ds_bu[id_] for id_ in all_bus if id_ in ds_bu},
        highlight_period=p2
    )

    return {"text": f"Revenue change between {plabel(p1)} and {plabel(p2)}:",
            "cards": cards, "charts": [fig, fig_line], "tables": [df], "export_df": export_df,
            "followups": ["Show trend analysis", f"Which segments drove the change in {plabel(p2)}?"]}


def _compare(ids, level, ll, sort):
    if len(ids) < 2:
        if sort == "worst_growth":
            top_df = pd.read_sql(f"""
                SELECT p37.id, (p42.revenue - p37.revenue)*100.0 / ABS(p37.revenue) as val
                FROM (SELECT id, revenue FROM hist WHERE level=? AND period=37) p37
                JOIN (SELECT id, revenue FROM hist WHERE level=? AND period=42) p42 ON p37.id = p42.id
                WHERE p37.revenue != 0
                ORDER BY val ASC LIMIT 2
            """, conn, params=(level, level))
            ids = top_df["id"].tolist()
        elif sort == "growth":
            top_df = pd.read_sql(f"""
                SELECT p37.id, (p42.revenue - p37.revenue)*100.0 / ABS(p37.revenue) as val
                FROM (SELECT id, revenue FROM hist WHERE level=? AND period=37) p37
                JOIN (SELECT id, revenue FROM hist WHERE level=? AND period=42) p42 ON p37.id = p42.id
                WHERE p37.revenue != 0
                ORDER BY val DESC LIMIT 2
            """, conn, params=(level, level))
            ids = top_df["id"].tolist()
        else:
            order = "ASC" if sort == "worst" else "DESC"
            top_df = pd.read_sql(f"""
                SELECT id FROM hist WHERE level=? GROUP BY id HAVING sum(revenue)>0
                ORDER BY sum(revenue) {order} LIMIT 2
            """, conn, params=(level,))
            ids = top_df["id"].tolist()

    if len(ids) < 2:
        all_df = pd.read_sql("SELECT DISTINCT id FROM hist WHERE level=? LIMIT 2", conn, params=(level,))
        ids = all_df["id"].tolist()

    ds = fetch_ds(ids)
    ids = [id_ for id_ in ids if id_ in ds]

    fig = make_line_chart(f"Revenue comparison: {' vs '.join(ids)}", {id_: ds[id_] for id_ in ids})

    rows = []
    for i, p in enumerate(PERIODS):
        row = {"Period": plabel(p)}
        for id_ in ids:
            row[id_] = fmt(ds[id_][i])
        if len(ids) == 2:
            diff = ds[ids[0]][i] - ds[ids[1]][i]
            row["Δ"] = f"{'+' if diff >= 0 else ''}{fmt_s(diff)}"
        rows.append(row)
    df = pd.DataFrame(rows)

    # Summary stats comparison
    stat_rows = []
    for id_ in ids:
        v = ds[id_]
        stat_rows.append({
            ll: id_,
            "CAGR": f"{'+' if cagr(v) >= 0 else ''}{cagr(v)}%",
            "Volatility": f"{volatility(v)}%",
            "Peak": plabel(PERIODS[int(np.argmax(v))]),
            "Avg/period": fmt(avg(v)),
        })
    df_stats = pd.DataFrame(stat_rows)

    export_df = pd.read_sql(f"SELECT level, id, period, revenue FROM hist WHERE id IN ({','.join('?'*len(ids))}) ORDER BY id, period", conn, params=tuple(ids))
    return {"text": f"Comparing {' vs '.join(ids)} ({ll} level) — Bottom-Up aggregated:",
            "charts": [fig], "tables": [df_stats, df], "export_df": export_df,
            "followups": [f"Drill down into {ids[0]}", f"Drill down into {ids[1]}", "Show trend analysis"]}


def _ranking(level, ll, n, sort="revenue", single_period=None, parent_id=None, threshold=None, threshold_op=None):
    from data import SUB_TO_SEG as _S2S, SEG_TO_BU as _S2B

    candidate_ids = None
    parent_label = ""
    if parent_id:
        parent_id = parent_id.upper()
        if level == "sub":
            if parent_id in _S2B.values():
                segs = [s for s, b in _S2B.items() if b == parent_id]
                candidate_ids = [sub for sub, seg in _S2S.items() if seg in segs]
                parent_label = f" in BU {parent_id}"
            elif parent_id in _S2B:
                candidate_ids = [sub for sub, seg in _S2S.items() if seg == parent_id]
                parent_label = f" in {parent_id}"
        elif level == "seg":
            if parent_id in _S2B.values():
                candidate_ids = [s for s, b in _S2B.items() if b == parent_id]
                parent_label = f" in BU {parent_id}"

    def id_filter_sql():
        """Return IN clause and params using plain 'id' — safe inside subqueries."""
        if candidate_ids:
            ph = ",".join("?" * len(candidate_ids))
            return f" AND id IN ({ph})", list(candidate_ids)
        return "", []

    if sort in ("growth", "worst_growth"):
        bottom = (sort == "worst_growth")
        order = "ASC" if bottom else "DESC"
        f1, p1 = id_filter_sql()
        f2, p2 = id_filter_sql()
        top_df = pd.read_sql(f"""
            SELECT p37.id, (p42.revenue - p37.revenue)*100.0 / ABS(p37.revenue) as val
            FROM (SELECT id, revenue FROM hist WHERE level=? AND period=37{f1}) p37
            JOIN (SELECT id, revenue FROM hist WHERE level=? AND period=42{f2}) p42 ON p37.id = p42.id
            WHERE p37.revenue != 0
            AND p37.id IN (
                SELECT id FROM hist WHERE level=?
                GROUP BY id HAVING SUM(CASE WHEN revenue > 0 THEN 1 ELSE 0 END) >= 30
            )
            ORDER BY val {order} LIMIT ?
        """, conn, params=[level] + p1 + [level] + p2 + [level] + [n])
        direction = "Lowest" if bottom else "Highest"
        metric_name = f"growth %{parent_label}"
    elif single_period is not None:
        bottom = (sort == "worst")
        order = "ASC" if bottom else "DESC"
        f1, p1 = id_filter_sql()
        top_df = pd.read_sql(f"""
            SELECT id, revenue as val FROM all_revenue WHERE level=? AND period=?
            AND revenue > 0{f1}
            ORDER BY val {order} LIMIT ?
        """, conn, params=[level, single_period] + p1 + [n])
        direction = "Bottom" if bottom else "Top"
        metric_name = f"revenue in P.{single_period}{parent_label}"
    else:
        bottom = (sort == "worst")
        order = "ASC" if bottom else "DESC"
        f1, p1 = id_filter_sql()
        top_df = pd.read_sql(f"""
            SELECT id, SUM(revenue) as val FROM hist WHERE level=?{f1}
            GROUP BY id HAVING val>0 ORDER BY val {order} LIMIT ?
        """, conn, params=[level] + p1 + [n])
        direction = "Bottom" if bottom else "Top"
        metric_name = f"revenue{parent_label}"

    ids = top_df["id"].tolist()
    values = top_df["val"].tolist()

    # Apply threshold filter on average period revenue
    if threshold is not None and threshold_op and ids:
        _op_map = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<="}
        _op = _op_map.get(threshold_op, ">")
        _avg_df = pd.read_sql(
            f"SELECT id, AVG(revenue) as avg_rev FROM all_revenue WHERE level=? AND id IN ({','.join('?'*len(ids))}) GROUP BY id HAVING avg_rev {_op} ?",
            conn, params=[level] + ids + [threshold]
        )
        _keep = set(_avg_df["id"].tolist())
        ids    = [i for i in ids    if i in _keep]
        values = [v for i, v in zip(top_df["id"].tolist(), values) if i in _keep]
        if not ids:
            return {"text": f"⚠️ No {ll}s found matching your revenue filter ({threshold_op} {fmt(threshold)})."}

    fig_bar = make_bar_chart(
        f"{direction} {n} {ll}s by {metric_name}",
        ids, values,
        color_fn=lambda v: "#ff6060" if bottom else "#00e5b8"
    )
    ds = fetch_ds(ids)
    fig_line = make_line_chart(
        f"Revenue timeline: {direction} {n} {ll}s",
        {id_: ds[id_] for id_ in ids}
    )
    df = make_table_df(ids, ds, ll, single_period=single_period, show_volatility=True)
    export_df = pd.read_sql(f"SELECT level, id, period, revenue FROM hist WHERE id IN ({','.join('?'*len(ids))}) ORDER BY id, period", conn, params=tuple(ids))

    followups = _suggest_followups(level, ids, sort)
    return {"text": f"{direction} {n} {ll}s by {metric_name} — Bottom-Up aggregated:",
            "charts": [fig_bar, fig_line], "tables": [df], "export_df": export_df,
            "followups": followups}


def _drilldown(id_: str):
    if not id_:
        return {"text": "⚠️ Please specify a BU or segment ID to drill down into."}

    id_up = id_.upper()

    if id_up in _BU:
        from data import SEG_TO_BU
        seg_ids = [s for s, b in SEG_TO_BU.items() if b == id_up]
        ds = fetch_ds(seg_ids)
        # Sort by avg revenue desc
        seg_ids_sorted = sorted([s for s in seg_ids if s in ds], key=lambda s: avg(ds[s]), reverse=True)
        fig = make_line_chart(f"Segments inside {id_up} — revenue P.1–42", {s: ds[s] for s in seg_ids_sorted})
        df = make_table_df(seg_ids_sorted, ds, "Segment", show_volatility=True)
        best_seg = seg_ids_sorted[0] if seg_ids_sorted else "?"
        export_df = pd.read_sql(
            f"SELECT level, id, period, revenue FROM hist WHERE id IN ({','.join('?'*len(seg_ids))}) ORDER BY id, period",
            conn, params=tuple(seg_ids)
        )
        return {"text": f"Here are all segments inside BU **{id_up}** — Bottom-Up aggregated:",
                "charts": [fig], "tables": [df], "export_df": export_df,
                "followups": [f"Show subsegments of {best_seg}", f"Which segments in {id_up} have highest growth?", f"Show trend for {id_up}"]}

    if id_up in _SEG:
        from data import SUB_TO_SEG
        sub_ids = [s for s, sg in SUB_TO_SEG.items() if sg == id_up]
        ds = fetch_ds(sub_ids)
        valid = sorted([s for s in sub_ids if s in ds], key=lambda s: avg(ds[s]), reverse=True)
        fig = make_line_chart(f"Subsegments inside {id_up} — revenue P.1–42", {s: ds[s] for s in valid})
        df = make_table_df(valid, ds, "Subsegment", show_volatility=True)
        export_df = pd.read_sql(
            f"SELECT level, id, period, revenue FROM hist WHERE id IN ({','.join('?'*len(valid))}) ORDER BY id, period",
            conn, params=tuple(valid)
        )
        return {"text": f"Here are all subsegments inside segment **{id_up}** — Bottom-Up aggregated:",
                "charts": [fig], "tables": [df], "export_df": export_df,
                "followups": [f"Which subsegments in {id_up} have highest growth?", f"Show trend for {id_up} subsegments", f"Compare top subsegments in {id_up}"]}

    if id_up in _SUB:
        from data import SUB_TO_SEG, SEG_TO_BU
        ds = fetch_ds([id_up])
        seg = SUB_TO_SEG.get(id_up, "unknown")
        bu = SEG_TO_BU.get(seg, "unknown")
        v = ds.get(id_up, [0]*len(PERIODS))
        fig = make_line_chart(f"Revenue timeline — {id_up}", {id_up: ds[id_up]})
        df = make_table_df([id_up], ds, "Subsegment", show_volatility=True)
        return {"text": f"Revenue for subsegment **{id_up}** (segment {seg} · BU {bu}) — CAGR {'+' if cagr(v)>=0 else ''}{cagr(v)}%, volatility {volatility(v)}% CV:",
                "charts": [fig], "tables": [df],
                "followups": [f"Show all subsegments in {seg}", f"Show trend for {seg}", f"Compare {id_up} with top subsegment"]}

    return {"text": f"⚠️ ID **{id_up}** not found in the hierarchy. Valid BUs: SSI027, SSI037, SSI047, SSI070."}


def _overview(level, ll, ids=None, single_period=None, parent_id=None):
    from data import SUB_TO_SEG as _S2S, SEG_TO_BU as _S2B

    if parent_id and not ids:
        parent_id = parent_id.upper()
        if level == "sub":
            if parent_id in _S2B.values():
                segs = [s for s, b in _S2B.items() if b == parent_id]
                ids = [sub for sub, seg in _S2S.items() if seg in segs]
            elif parent_id in _S2B:
                ids = [sub for sub, seg in _S2S.items() if seg == parent_id]
        elif level == "seg":
            if parent_id in _S2B.values():
                ids = [s for s, b in _S2B.items() if b == parent_id]

    if not ids:
        if single_period is not None:
            top_df = pd.read_sql("""
                SELECT id FROM all_revenue WHERE level=? AND period=? AND revenue>0
                ORDER BY revenue DESC LIMIT 8
            """, conn, params=(level, single_period))
        else:
            top_df = pd.read_sql("""
                SELECT id FROM hist WHERE level=? GROUP BY id HAVING sum(revenue)>0
                ORDER BY sum(revenue) DESC LIMIT 8
            """, conn, params=(level,))
        ids = top_df["id"].tolist()

    valid_level_ids = set(pd.read_sql(
        "SELECT DISTINCT id FROM hist WHERE level=?", conn, params=(level,)
    )["id"].tolist())
    ids = [i for i in ids if i in valid_level_ids]

    ds = fetch_ds(ids)
    period_label = f"P.{single_period}" if single_period is not None else "P.1–48"
    parent_label = f" · {parent_id}" if parent_id else ""

    if single_period is not None:
        period_idx = PERIODS.index(single_period)
        bar_ids  = [id_ for id_ in ids if id_ in ds]
        bar_vals = [ds[id_][period_idx] for id_ in bar_ids]
        fig = make_bar_chart(
            f"Revenue by {ll}{parent_label} — {period_label} · Bottom-Up aggregated",
            bar_ids, bar_vals,
            color_fn=lambda v: "#00e5b8"
        )
    else:
        fig = make_line_chart(
            f"Revenue by {ll}{parent_label} — Bottom-Up aggregated · P.1–48",
            {id_: ds[id_] for id_ in ids if id_ in ds}
        )

    df = make_table_df(ids, ds, ll, single_period=single_period)
    export_df = pd.read_sql(f"SELECT level, id, period, revenue FROM hist WHERE id IN ({','.join('?'*len(ids))}) ORDER BY id, period", conn, params=tuple(ids))
    followups = _suggest_followups(level, ids)
    return {"text": f"Revenue overview — {ll}{parent_label} · {period_label} · Bottom-Up aggregated:",
            "charts": [fig], "tables": [df], "export_df": export_df,
            "followups": followups}


def _forecast(ids, level, ll):
    """Dedicated P.43–48 forecast view."""
    from data import SEG_TO_BU as _S2B, SUB_TO_SEG as _S2S

    fc_data = {"bu": BU_FC, "seg": SEG_FC, "sub": SUB_FC}[level]

    # Resolve parent filtering (e.g. "forecast for segments in SSI037")
    target_ids = None
    parent_label = ""
    if ids:
        id_up = ids[0].upper()
        if level == "seg" and id_up in _S2B.values():
            target_ids = [s for s, b in _S2B.items() if b == id_up]
            parent_label = f" in BU {id_up}"
        elif level == "sub" and id_up in _S2B.values():
            segs = [s for s, b in _S2B.items() if b == id_up]
            target_ids = [sub for sub, seg in _S2S.items() if seg in segs]
            parent_label = f" in BU {id_up}"
        elif level == "sub" and id_up in _S2B:
            target_ids = [sub for sub, seg in _S2S.items() if seg == id_up]
            parent_label = f" in {id_up}"
        elif id_up in fc_data:
            target_ids = [id_up]
        else:
            target_ids = [id_up]

    if target_ids is None:
        target_ids = list(fc_data.keys())

    valid_ids = [id_ for id_ in target_ids if id_ in fc_data]
    if not valid_ids:
        return {"text": f"⚠️ No forecast data found for the requested {ll}s."}

    # Chart: P.43–48 lines
    fig = go.Figure()
    period_labels = [plabel(p) for p in PERIODS_FORECAST]
    for i, id_ in enumerate(valid_ids[:8]):
        v = fc_data[id_]
        fig.add_trace(go.Scatter(
            x=period_labels, y=v, name=id_,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            marker=dict(size=6), mode="lines+markers",
            customdata=[id_] * len(PERIODS_FORECAST)
        ))
    fig.update_layout(
        title=dict(text=f"FlowState-r1.1 Forecast Oct/24–Mar/25 — {ll}s{parent_label}", font=dict(size=12, color="#dce8f5")),
        **PLOTLY_LAYOUT
    )

    # Table
    rows = []
    for id_ in valid_ids:
        v = fc_data[id_]
        g = cagr(v)
        row = {ll: id_}
        for i, p in enumerate(PERIODS_FORECAST):
            row[plabel(p)] = fmt(v[i])
        row["CAGR Oct/24–Mar/25"] = f"{'▲' if g >= 0 else '▼'} {abs(g)}%"
        rows.append(row)
    df = pd.DataFrame(rows)

    export_df = pd.read_sql(
        f"SELECT level, id, period, revenue FROM forecast48 WHERE level=? AND id IN ({','.join('?'*len(valid_ids))}) ORDER BY id, period",
        conn, params=[level] + valid_ids
    )

    top_id = valid_ids[0] if valid_ids else None
    fu = []
    if top_id:
        fu.append(f"Drill down into {top_id}")
    fu += ["Show historical trend", "Executive summary"]
    return {
        "text": f"FlowState-r1.1 forecast Oct/24–Mar/25 · {len(valid_ids)} {ll}s{parent_label}:",
        "charts": [fig],
        "tables": [df],
        "export_df": export_df,
        "followups": fu,
    }


def _heatmap(level, ll):
    """Growth heatmap — CAGR colour grid for all entities at a given level (historical P.1–42)."""
    from data import SEG_TO_BU as _S2B

    if level == "bu":
        ids = list(BU_HIST.keys())
        data_dict = BU_HIST
        z = [[cagr(data_dict[id_]) for id_ in ids]]
        text = [[f"{id_}<br>{cagr(data_dict[id_]):+.1f}%" for id_ in ids]]
        y_labels = ["BU"]
        x_labels = ids

    elif level == "seg":
        # One row per BU, segments as columns (grouped)
        bus = sorted(set(_S2B.values()))
        bu_segs = {bu: sorted([s for s, b in _S2B.items() if b == bu]) for bu in bus}
        max_cols = max(len(v) for v in bu_segs.values())
        z, text = [], []
        for bu in bus:
            segs = bu_segs[bu]
            row_z, row_t = [], []
            for i in range(max_cols):
                if i < len(segs):
                    v = SEG_HIST.get(segs[i], [0])
                    c = cagr(v)
                    row_z.append(c)
                    row_t.append(f"{segs[i]}<br>{c:+.1f}%")
                else:
                    row_z.append(None)
                    row_t.append("")
            z.append(row_z)
            text.append(row_t)
        y_labels = bus
        x_labels = [f"Seg {i+1}" for i in range(max_cols)]
        data_dict = SEG_HIST
        ids = list(SEG_HIST.keys())

    else:  # sub — top 60 by average revenue for readability
        ids_all = sorted(
            [id_ for id_ in SUB_HIST],
            key=lambda x: avg(SUB_HIST[x]),
            reverse=True
        )[:60]
        data_dict = {id_: SUB_HIST[id_] for id_ in ids_all}
        ids = ids_all
        ncols = 6
        nrows = (len(ids) + ncols - 1) // ncols
        z, text = [], []
        for row in range(nrows):
            row_z, row_t = [], []
            for col in range(ncols):
                idx = row * ncols + col
                if idx < len(ids):
                    v = data_dict[ids[idx]]
                    c = cagr(v)
                    row_z.append(c)
                    row_t.append(f"{ids[idx]}<br>{c:+.1f}%")
                else:
                    row_z.append(None)
                    row_t.append("")
            z.append(row_z)
            text.append(row_t)
        y_labels = [f"Row {i+1}" for i in range(nrows)]
        x_labels = [f"Col {i+1}" for i in range(ncols)]

    fig = go.Figure(go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=9, color="#dce8f5"),
        colorscale=[[0, "#ff4444"], [0.45, "#2a3f52"], [0.55, "#2a3f52"], [1, "#00e5b8"]],
        zmid=0,
        zmin=-20,
        zmax=20,
        showscale=True,
        colorbar=dict(title=dict(text="CAGR %", font=dict(color="#6b7e96")), tickfont=dict(size=10, color="#6b7e96"),
                      tickvals=[-20, -10, 0, 10, 20],
                      ticktext=["≤−20%", "−10%", "0%", "+10%", "≥+20%"]),
        xgap=2, ygap=2,
        y=y_labels,
        x=x_labels,
    ))
    height = max(280, 80 * len(z))
    layout = {**PLOTLY_LAYOUT, "height": height}
    fig.update_layout(
        title=dict(text=f"Growth Heatmap — {ll}s by CAGR (Apr/21–Sep/24)", font=dict(size=12, color="#dce8f5")),
        **layout
    )

    # Summary table sorted by CAGR desc
    rows = sorted(
        [{"ID": id_, "CAGR": f"{cagr(data_dict[id_]):+.1f}%", "Avg/period": fmt(avg(data_dict[id_]))}
         for id_ in ids if id_ in data_dict],
        key=lambda r: float(r["CAGR"].rstrip("%")),
        reverse=True
    )
    df = pd.DataFrame(rows)

    growing  = sum(1 for id_ in ids if id_ in data_dict and cagr(data_dict[id_]) > 0)
    declining = len([id_ for id_ in ids if id_ in data_dict]) - growing

    return {
        "text": f"Growth heatmap across {len(ids)} {ll}s — **{growing}** growing (green), **{declining}** declining (red).",
        "charts": [fig],
        "tables": [df],
        "followups": [f"Top {min(5, len(ids))} {ll}s by growth", "Show trend analysis", "Executive summary"],
    }


def make_excel_bytes():
    """Generate multi-sheet Excel: BU / Segment / Subsegment (historical + forecast)."""
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, hist_d, fc_d in [
            ("BU",         BU_HIST,  BU_FC),
            ("Segment",    SEG_HIST, SEG_FC),
            ("Subsegment", SUB_HIST, SUB_FC),
        ]:
            rows = []
            for id_ in sorted(hist_d.keys()):
                row = {"ID": id_}
                for i, p in enumerate(PERIODS_HIST):
                    row[plabel(p)] = round(hist_d[id_][i], 2)
                for i, p in enumerate(PERIODS_FORECAST):
                    row[f"{plabel(p)} (FC)"] = round(fc_d.get(id_, [0] * 6)[i], 2)
                row["Hist CAGR"] = f"{cagr(hist_d[id_]):+.1f}%"
                row["FC CAGR"]   = f"{cagr(fc_d.get(id_, [1, 1])):+.1f}%"
                rows.append(row)
            pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)
    buf.seek(0)
    return buf.read()
