"""
pdf_export.py — Revera business-oriented PDF report generator
Uses reportlab + matplotlib for charts (no browser/kaleido dependency).
Called from app.py when user clicks "Export PDF".
"""
import io
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── COLOURS ────────────────────────────────────────────────────────────────
TEAL    = colors.HexColor("#00e5b8")
DARK    = colors.HexColor("#090d12")
DARK2   = colors.HexColor("#10151e")
SLATE   = colors.HexColor("#6b7e96")
LIGHT   = colors.HexColor("#dce8f5")
ORANGE  = colors.HexColor("#ff6b35")
WHITE   = colors.white

BG      = "#090d12"
BG2     = "#10151e"
TEAL_C  = "#00e5b8"
BLUE_C  = "#3d9eff"
ORNG_C  = "#ff6b35"
SLATE_C = "#6b7e96"
LIGHT_C = "#dce8f5"
CHART_COLORS = ["#00e5b8","#3d9eff","#ffb547","#ff6060","#b47fff","#ff9fcc"]

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm

# ── STYLES ─────────────────────────────────────────────────────────────────
def _styles():
    return {
        "title":      ParagraphStyle("title",      fontName="Helvetica-Bold", fontSize=22, textColor=LIGHT,  spaceAfter=4,  leading=28),
        "subtitle":   ParagraphStyle("subtitle",   fontName="Helvetica",      fontSize=11, textColor=SLATE,  spaceAfter=16, leading=16),
        "section":    ParagraphStyle("section",    fontName="Helvetica-Bold", fontSize=13, textColor=TEAL,   spaceBefore=18,spaceAfter=6, leading=18),
        "body":       ParagraphStyle("body",       fontName="Helvetica",      fontSize=10, textColor=LIGHT,  spaceAfter=6,  leading=15),
        "small":      ParagraphStyle("small",      fontName="Helvetica",      fontSize=8,  textColor=SLATE,  spaceAfter=4,  leading=12),
        "metric_val": ParagraphStyle("metric_val", fontName="Helvetica-Bold", fontSize=17, textColor=TEAL,   leading=22),
        "metric_lbl": ParagraphStyle("metric_lbl", fontName="Helvetica-Bold", fontSize=8,  textColor=LIGHT,  leading=12),
        "metric_sub": ParagraphStyle("metric_sub", fontName="Helvetica",      fontSize=7,  textColor=SLATE,  leading=11),
        "footer":     ParagraphStyle("footer",     fontName="Helvetica",      fontSize=8,  textColor=SLATE,  alignment=TA_CENTER),
        "right":      ParagraphStyle("right",      fontName="Helvetica",      fontSize=9,  textColor=SLATE,  alignment=TA_RIGHT),
    }

# ── HELPERS ────────────────────────────────────────────────────────────────
def _fmt(v):
    a = abs(v)
    if a >= 1e9: return f"EUR{v/1e9:.2f}B"
    if a >= 1e6: return f"EUR{v/1e6:.1f}M"
    if a >= 1e3: return f"EUR{v/1e3:.0f}K"
    return f"EUR{v:.0f}"

def _cagr(v):
    """True annualised CAGR — periods are months.
    Guards: ≥ 6 months; base ≥ 0.5% of peak; last value within last 12 months."""
    n = len(v)
    if n < 2: return 0.0
    peak = max(v)
    min_base = peak * 0.005
    first_i = next((i for i, x in enumerate(v) if x >= min_base and x > 0), None)
    last_i  = next((i for i, x in enumerate(reversed(v)) if x > 0), None)
    if first_i is None or last_i is None: return 0.0
    last_i = n - 1 - last_i
    if last_i < n - 12: return 0.0
    months = last_i - first_i
    if months < 6: return 0.0
    return round(((v[last_i]/v[first_i])**(12/months)-1)*100, 1)

def _mfmt(x, pos):
    a = abs(x)
    if a >= 1e9: return f"{x/1e9:.1f}B"
    if a >= 1e6: return f"{x/1e6:.0f}M"
    if a >= 1e3: return f"{x/1e3:.0f}K"
    return str(int(x))

_img_buffers = []  # keep PNG bytes alive until PDF is built

def _fig_to_img(fig, w_cm=16, h_cm=7):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    png_bytes = buf.getvalue()          # capture bytes while buffer still valid
    _img_buffers.append(png_bytes)      # keep reference alive
    fresh = io.BytesIO(png_bytes)       # fresh stream for RLImage
    return RLImage(fresh, width=w_cm*cm, height=h_cm*cm)

def _table_style(header_bg=DARK2):
    return TableStyle([
        ("BACKGROUND",   (0,0), (-1,0),  header_bg),
        ("TEXTCOLOR",    (0,0), (-1,0),  TEAL),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,0),  8),
        ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",     (0,1), (-1,-1), 8),
        ("TEXTCOLOR",    (0,1), (-1,-1), LIGHT),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [DARK, DARK2]),
        ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#1e2a3a")),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ])

# ── MATPLOTLIB CHART BUILDERS ──────────────────────────────────────────────
def _chart_hist_fc(bu_hist, bu_fc, hist_periods, fc_periods):
    """Full history + forecast area chart, blue/orange, matches FlowState-r1.1 plot style."""
    from data import PERIOD_LABELS as _PL
    hist_agg = [sum(bu_hist[bu][i] for bu in bu_hist) for i in range(len(hist_periods))]
    fc_agg   = [sum(bu_fc[bu][i]   for bu in bu_fc)   for i in range(len(fc_periods))]

    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)

    ax.fill_between(hist_periods, hist_agg, alpha=0.45, color=BLUE_C, label="Historical Revenue (Apr/21–Sep/24)")
    ax.plot(hist_periods, hist_agg, color=BLUE_C, linewidth=1.2)

    x_fc = [hist_periods[-1]] + fc_periods
    y_fc = [hist_agg[-1]] + fc_agg
    ax.fill_between(x_fc, y_fc, alpha=0.45, color=ORNG_C, label="FlowState-r1.1 Forecast (Oct/24–Mar/25)")
    ax.plot(x_fc, y_fc, color=ORNG_C, linewidth=2, marker="o", markersize=5)
    for x, y in zip(fc_periods, fc_agg):
        ax.annotate(f"{y/1e6:.0f}M", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5, color=ORNG_C)

    ax.axvline(x=42.5, color="white", linewidth=0.8, linestyle="--", alpha=0.35)
    ax.text(43.2, max(fc_agg)*0.92, "forecast →", fontsize=8, color=SLATE_C)

    # Calendar month tick labels every 6 periods
    tick_ps = [p for p in hist_periods if (p - 1) % 6 == 0] + list(fc_periods)
    ax.set_xticks(tick_ps)
    ax.set_xticklabels([_PL.get(p, str(p)) for p in tick_ps], rotation=45, ha="right", fontsize=7.5)

    ax.yaxis.set_major_formatter(FuncFormatter(_mfmt))
    ax.set_ylabel("Total Revenue", color=SLATE_C, fontsize=9)
    ax.tick_params(colors=SLATE_C, labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor("#1e2a3a")
    ax.grid(axis="y", color="#1e2a3a", linewidth=0.5)
    legend = ax.legend(facecolor=BG2, edgecolor="#1e2a3a", labelcolor=LIGHT_C, fontsize=8)
    ax.set_title("Aggregate Revenue: Historical & Forecast", color=LIGHT_C, fontsize=10, pad=8)
    return _fig_to_img(fig, 16, 5.5)

def _chart_bu_fc(bu_fc, fc_periods):
    """BU-level forecast line chart."""
    from data import PERIOD_LABELS as _PL
    fig, ax = plt.subplots(figsize=(10.5, 2.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)

    for i, (bu, vals) in enumerate(sorted(bu_fc.items())):
        ax.plot(fc_periods, vals, marker="o", markersize=4,
                color=CHART_COLORS[i % len(CHART_COLORS)], linewidth=2, label=bu)

    ax.set_xticks(fc_periods)
    ax.set_xticklabels([_PL.get(p, str(p)) for p in fc_periods], rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(_mfmt))
    ax.tick_params(colors=SLATE_C, labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor("#1e2a3a")
    ax.grid(axis="y", color="#1e2a3a", linewidth=0.5)
    ax.legend(facecolor=BG2, edgecolor="#1e2a3a", labelcolor=LIGHT_C, fontsize=8)
    ax.set_title("Revenue Forecast by Business Unit (Oct/24–Mar/25)", color=LIGHT_C, fontsize=10, pad=8)
    return _fig_to_img(fig, 16, 6)

def _chart_bu_hist(bu_hist, hist_periods):
    """BU-level historical revenue line chart."""
    from data import PERIOD_LABELS as _PL
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)
    for i, (bu, vals) in enumerate(sorted(bu_hist.items())):
        ax.plot(hist_periods, vals, marker="o", markersize=3,
                color=CHART_COLORS[i % len(CHART_COLORS)], linewidth=1.8, label=bu)
    tick_ps = [p for p in hist_periods if (p - 1) % 6 == 0]
    ax.set_xticks(tick_ps)
    ax.set_xticklabels([_PL.get(p, str(p)) for p in tick_ps], rotation=45, ha="right", fontsize=7.5)
    ax.yaxis.set_major_formatter(FuncFormatter(_mfmt))
    ax.tick_params(colors=SLATE_C, labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor("#1e2a3a")
    ax.grid(axis="y", color="#1e2a3a", linewidth=0.5)
    ax.legend(facecolor=BG2, edgecolor="#1e2a3a", labelcolor=LIGHT_C, fontsize=8)
    ax.set_title("Historical Revenue by Business Unit (Apr/21–Sep/24)", color=LIGHT_C, fontsize=10, pad=8)
    return _fig_to_img(fig, 16, 4.5)

def _chart_top_segs(seg_fc, fc_periods, top_n=6):
    """Horizontal bar chart of top segments by total forecast."""
    top = sorted(seg_fc, key=lambda s: sum(seg_fc[s]), reverse=True)[:top_n]
    vals = [sum(seg_fc[s]) / 1e6 for s in top]

    fig, ax = plt.subplots(figsize=(10, 2.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)

    bars = ax.barh(top[::-1], vals[::-1], color=TEAL_C, alpha=0.8, height=0.6)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
                f"{v:.0f}M", va="center", ha="left", fontsize=8, color=TEAL_C)

    ax.set_xlabel("Total Forecast Revenue (EUR M)", color=SLATE_C, fontsize=8)
    ax.tick_params(colors=LIGHT_C, labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor("#1e2a3a")
    ax.grid(axis="x", color="#1e2a3a", linewidth=0.5)
    ax.set_title(f"Top {top_n} Segments by Total Forecast Revenue", color=LIGHT_C, fontsize=10, pad=8)
    ax.set_xlim(0, max(vals) * 1.18)
    return _fig_to_img(fig, 16, 5)

# ── MAIN EXPORT FUNCTION ───────────────────────────────────────────────────
def generate_pdf():
    """Generate the Revera business report. Returns PDF bytes."""
    from data import (BU_HIST, BU_FC, SEG_HIST, SEG_FC,
                      PERIODS_HIST, PERIODS_FORECAST, SEG_TO_BU, PERIOD_LABELS)
    def _plabel(p):
        return PERIOD_LABELS.get(p, f"P.{p}")

    _img_buffers.clear()
    S   = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="Revera — Siemens Advanta Forecast Report",
        author="Revera · NOVA Business Case 2026",
    )
    story = []
    now   = datetime.datetime.now().strftime("%B %d, %Y")

    # ── COVER / HEADER ─────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=14))
    story.append(Paragraph("Revera", S["title"]))
    story.append(Paragraph("Siemens Advanta · Revenue Forecast Intelligence Report", S["subtitle"]))
    story.append(Paragraph("FlowState-r1.1 · Bottom-Up Hierarchical Aggregation · Oct/24–Mar/25", S["small"]))
    story.append(Paragraph(f"Generated {now}", S["right"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=SLATE, spaceBefore=10, spaceAfter=18))

    # ── KPI CARDS ──────────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", S["section"]))

    total_p42 = sum(BU_HIST[bu][-1] for bu in BU_HIST)
    total_p48 = sum(BU_FC[bu][-1]   for bu in BU_FC)
    total_p43 = sum(BU_FC[bu][0]    for bu in BU_FC)
    fc_growth = round((total_p48/total_p43-1)*100, 1) if total_p43 else 0
    hist_agg  = [sum(BU_HIST[bu][i] for bu in BU_HIST) for i in range(len(PERIODS_HIST))]
    h_cagr    = _cagr(hist_agg)
    bu_totals = {bu: sum(BU_FC[bu]) for bu in BU_FC}
    best_bu   = max(bu_totals, key=bu_totals.get)

    kpis = [
        (_fmt(total_p48),          "Total Forecast Mar/25",  "all BUs"),
        (f"+{fc_growth}%",         "Forecast Growth",         "Oct/24 → Mar/25"),
        (f"+{h_cagr}%",            "Historical CAGR",         "Apr/21 → Sep/24"),
        (best_bu,                  "Top Forecast BU",         f"{_fmt(bu_totals[best_bu])} total"),
    ]
    col_w = (PAGE_W - 2*MARGIN) / 4
    kpi_data = [
        [Paragraph(v,  S["metric_val"]) for v,_,_ in kpis],
        [Paragraph(l,  S["metric_lbl"]) for _,l,_ in kpis],
        [Paragraph(s,  S["metric_sub"]) for _,_,s in kpis],
    ]
    kt = Table(kpi_data, colWidths=[col_w]*4)
    kt.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), DARK2),
        ("BOX",          (0,0),(-1,-1), 0.5, colors.HexColor("#1e2a3a")),
        ("INNERGRID",    (0,0),(-1,-1), 0.3, colors.HexColor("#1e2a3a")),
        ("ALIGN",        (0,0),(-1,-1), "CENTER"),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0),(-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
    ]))
    story.append(kt)
    story.append(Spacer(1, 8))

    # ── NARRATIVE ──────────────────────────────────────────────────────────
    top3_segs = sorted(SEG_FC, key=lambda s: sum(SEG_FC[s]), reverse=True)[:3]
    top3_str  = ", ".join(f"{s} ({_fmt(sum(SEG_FC[s])//6)} avg/period)" for s in top3_segs)
    story.append(Paragraph(
        f"Siemens Advanta is forecast to reach <b>{_fmt(total_p48)}</b> in aggregate revenue by Mar/25, "
        f"a <b>+{fc_growth}%</b> increase over the forecast horizon (Oct/24–Mar/25). "
        f"Historically, aggregate revenue grew at an annualised CAGR of <b>{h_cagr}%</b> across 42 monthly periods (Apr/21–Sep/24). "
        f"<b>{best_bu}</b> leads the forecast with the highest total projected volume. "
        f"The top three segments by forecast volume are: {top3_str}.",
        S["body"]))

    # ── HISTORICAL + FORECAST CHART ────────────────────────────────────────
    story.append(Paragraph("Revenue: Full History and Forecast (Apr/21–Mar/25)", S["section"]))
    story.append(_chart_hist_fc(BU_HIST, BU_FC, PERIODS_HIST, PERIODS_FORECAST))
    story.append(Paragraph(
        "Blue: historical revenue (Apr/21–Sep/24) · Orange: FlowState-r1.1 forecast (Oct/24–Mar/25). "
        "Dashed line marks the forecast boundary.", S["small"]))

    # ── PAGE BREAK ─────────────────────────────────────────────────────────
    story.append(PageBreak())

    # ── BU CHARTS (page 2) ────────────────────────────────────────────────
    story.append(Paragraph("Historical Revenue by Business Unit (Apr/21–Sep/24)", S["section"]))
    story.append(_chart_bu_hist(BU_HIST, PERIODS_HIST))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Revenue Forecast by Business Unit (Oct/24–Mar/25)", S["section"]))
    story.append(_chart_bu_fc(BU_FC, PERIODS_FORECAST))

    # ── BU FORECAST TABLE ──────────────────────────────────────────────────
    story.append(Paragraph("Forecast by Business Unit (Oct/24–Mar/25)", S["section"]))
    hdr = ["BU"] + [_plabel(p) for p in PERIODS_FORECAST] + ["CAGR", "Total Oct/24-Mar/25"]
    rows = [hdr]
    for bu in sorted(BU_FC):
        v = BU_FC[bu]
        c = f"+{_cagr(v)}%" if _cagr(v) >= 0 else f"{_cagr(v)}%"
        rows.append([bu] + [_fmt(x) for x in v] + [c, _fmt(sum(v))])
    cw = [2*cm] + [1.9*cm]*6 + [1.5*cm, 2.2*cm]
    t  = Table(rows, colWidths=cw)
    t.setStyle(_table_style())
    story.append(t)

    # ── TOP SEGMENTS CHART ─────────────────────────────────────────────────
    story.append(Paragraph("Top Segments by Forecast Revenue", S["section"]))
    story.append(_chart_top_segs(SEG_FC, PERIODS_FORECAST, top_n=8))

    # ── TOP 10 SEGMENTS TABLE ──────────────────────────────────────────────
    story.append(Paragraph("Top 10 Segments — Detailed Forecast", S["section"]))
    top10 = sorted(SEG_FC, key=lambda s: sum(SEG_FC[s]), reverse=True)[:10]
    shdr  = ["Segment", "BU", "Oct/24", "Mar/25", "CAGR", "Avg/Period"]
    srows = [shdr]
    for seg in top10:
        v = SEG_FC[seg]
        c = f"+{_cagr(v)}%" if _cagr(v) >= 0 else f"{_cagr(v)}%"
        srows.append([seg, SEG_TO_BU.get(seg,"?"), _fmt(v[0]), _fmt(v[-1]), c, _fmt(sum(v)//6)])
    scw = [3.2*cm, 2*cm, 2.4*cm, 2.4*cm, 1.8*cm, 2.6*cm]
    st  = Table(srows, colWidths=scw)
    st.setStyle(_table_style())
    story.append(st)

    # ── SEGMENTS REQUIRING ATTENTION ──────────────────────────────────────
    story.append(Paragraph("Segments Requiring Attention", S["section"]))
    story.append(Paragraph(
        "These segments show the lowest projected revenue over the forecast horizon "
        "and may warrant further business review or strategic intervention.", S["body"]))
    bot5  = sorted(SEG_FC, key=lambda s: sum(SEG_FC[s]))[:5]
    brows = [shdr]
    for seg in bot5:
        v = SEG_FC[seg]
        c = f"+{_cagr(v)}%" if _cagr(v) >= 0 else f"{_cagr(v)}%"
        brows.append([seg, SEG_TO_BU.get(seg,"?"), _fmt(v[0]), _fmt(v[-1]), c, _fmt(sum(v)//6)])
    bt = Table(brows, colWidths=scw)
    bts = _table_style()
    for i, seg in enumerate(bot5, 1):
        if _cagr(SEG_FC[seg]) < 0:
            bts.add("TEXTCOLOR", (4,i), (4,i), ORANGE)
    bt.setStyle(bts)
    story.append(bt)

    # ── MODEL PERFORMANCE ──────────────────────────────────────────────────
    story.append(Paragraph("Model Performance", S["section"]))
    story.append(Paragraph(
        "FlowState-r1.1 (Google). "
        "Bottom-up aggregation: subsegment → segment → BU.", S["body"]))
    pdata = [
        ["Metric",        "Value",        "Notes"],
        ["Model",         "FlowState-r1.1",      ""],
        ["Reconciliation","Bottom-Up",  "Subsegment level"],
        ["R²",            "0.9871",       "Excellent fit"],
        ["wMAPE",         "9.40%",       "Subsegment level"],
        ["RMSE",          "7,743,554 €",  "Subsegment level"],
        ["MAE",           "3,663,463 €",  "Subsegment level"],
    ]
    pt  = Table(pdata, colWidths=[4*cm, 4*cm, 5*cm])
    pts = _table_style()
    pts.add("TEXTCOLOR", (1,1), (1,1), TEAL)   # FlowState-r1.1 value teal
    pts.add("TEXTCOLOR", (1,3), (1,3), TEAL)   # R² value teal
    pt.setStyle(pts)
    story.append(pt)

    # ── FOOTER ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 18))
    story.append(HRFlowable(width="100%", thickness=0.5, color=SLATE, spaceAfter=6))
    story.append(Paragraph(
        f"Revera · Siemens Advanta Forecast Intelligence · NOVA Business Case 2026 · {now}  |  "
        "Confidential — for internal use only", S["footer"]))

    # ── BUILD ──────────────────────────────────────────────────────────────
    def _bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(DARK)
        canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        canvas.restoreState()

    doc.build(story, onFirstPage=_bg, onLaterPages=_bg)
    buf.seek(0)
    return buf.read()
