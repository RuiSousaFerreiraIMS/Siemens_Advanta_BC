import re
import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from PIL import Image as _PILImage
from data import BU, SEG, SUB, SUB_TO_SEG, SEG_TO_BU, PERIODS
from nlu import parse_intent
from builders import build_response, make_excel_bytes
from pdf_export import generate_pdf

def _md(text):
    """Convert markdown bold/italic to HTML for use inside raw HTML divs."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = text.replace('\n', '<br>')
    return text

# ── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Revera — Forecast Intelligence",
    page_icon=_PILImage.open("revera_icon.png"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STYLES ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; font-size: 15px; }

.stApp { background-color: #090d12; }
section[data-testid="stSidebar"] { background-color: #10151e; border-right: 1px solid rgba(255,255,255,0.07); }
section[data-testid="stSidebar"] * { color: #dce8f5 !important; font-size: 13px !important; }

#MainMenu, footer, header { display: none; }
[data-testid="stSidebarCollapsedControl"] { display: flex !important; visibility: visible !important; }
[data-testid="stLogo"] img { width: 72px !important; height: 72px !important; }
.block-container { padding: 1rem 1.5rem; max-width: 1000px; }

/* Chat messages */
.user-msg {
    background: #161d29; border: 1px solid rgba(255,255,255,0.11);
    border-radius: 12px 4px 12px 12px; padding: 10px 16px;
    margin: 4px 0; font-size: 15px !important; color: #dce8f5;
    margin-left: 20%; line-height: 1.6;
}
.agent-msg {
    background: #10151e; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 4px 12px 12px 12px; padding: 12px 16px;
    margin: 4px 0; font-size: 15px !important; color: #dce8f5;
    margin-right: 5%; line-height: 1.6;
}
.agent-label {
    font-family: 'DM Mono', monospace; font-size: 11px !important;
    color: #00e5b8; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.1em;
}
.user-label {
    font-family: 'DM Mono', monospace; font-size: 11px !important;
    color: #6b7e96; margin-bottom: 6px; text-align: right; text-transform: uppercase; letter-spacing: 0.1em;
}

/* Streamlit dataframe font size */
[data-testid="stDataFrame"] * { font-size: 13px !important; }

/* Chat input */
[data-testid="stChatInput"] textarea { font-size: 15px !important; }

/* Sidebar metric rows */
.sb-metric { display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
.sb-key { font-family: 'DM Mono', monospace; font-size: 11px; color: #38475a; }
.sb-val { font-family: 'DM Mono', monospace; font-size: 11px; color: #00e5b8; font-weight: 500; }

/* Breadcrumb */
.breadcrumb {
    font-family: 'DM Mono', monospace; font-size: 11px; color: #38475a;
    padding: 6px 0; margin-bottom: 4px;
}
.breadcrumb span { color: #00e5b8; }
.breadcrumb .sep { color: #38475a; margin: 0 4px; }

/* Follow-up chips */
.followup-label {
    font-family: 'DM Mono', monospace; font-size: 10px; color: #38475a;
    text-transform: uppercase; letter-spacing: 0.08em; margin: 10px 0 4px 0;
}

/* Uniform height for all quick-chip and follow-up buttons */
[data-testid="stHorizontalBlock"] .stButton button {
    height: 56px;
    white-space: normal;
    line-height: 1.3;
}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "breadcrumb"  not in st.session_state: st.session_state.breadcrumb  = []  # list of (level, id)
if "last_followups" not in st.session_state: st.session_state.last_followups = []

# ── HELPERS ────────────────────────────────────────────────────────────────
def _update_breadcrumb(parsed: dict):
    """Update breadcrumb based on the last parsed intent."""
    intent = parsed.get("intent")
    level  = parsed.get("level", "bu")
    ids    = parsed.get("ids", [])

    if intent in ("error", "metrics", "executive", "forecast", "heatmap"):
        return  # don't change breadcrumb for meta queries

    if intent == "drilldown" and ids:
        id_up = ids[0].upper()
        if id_up in BU:
            st.session_state.breadcrumb = [("bu", id_up)]
        elif id_up in SEG:
            bu = SEG_TO_BU.get(id_up, "?")
            st.session_state.breadcrumb = [("bu", bu), ("seg", id_up)]
        elif id_up in SUB:
            seg = SUB_TO_SEG.get(id_up, "?")
            bu  = SEG_TO_BU.get(seg, "?")
            st.session_state.breadcrumb = [("bu", bu), ("seg", seg), ("sub", id_up)]
    elif ids:
        # For ranking/overview with a parent id, set partial breadcrumb
        id_up = ids[0].upper()
        if id_up in BU:
            st.session_state.breadcrumb = [("bu", id_up)]
        elif id_up in SEG:
            bu = SEG_TO_BU.get(id_up, "?")
            st.session_state.breadcrumb = [("bu", bu), ("seg", id_up)]

def _render_breadcrumb():
    if not st.session_state.breadcrumb:
        return
    labels = {"bu": "BU", "seg": "Segment", "sub": "Subsegment"}
    parts = []
    for lvl, id_ in st.session_state.breadcrumb:
        parts.append(f'<span class="sep">›</span><span>{labels[lvl]}: {id_}</span>')
    html = '<div class="breadcrumb">Context ' + "".join(parts) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────
st.logo("revera_icon.png")

with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 8px 0;">
      <div style="font-family:'DM Mono',monospace;font-size:20px;font-weight:600;color:#dce8f5;letter-spacing:0.12em;">REVERA</div>
      <div style="font-family:'DM Mono',monospace;font-size:12px;color:#38475a;letter-spacing:0.06em;">Forecast Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # Read from Streamlit secrets if available (Streamlit Cloud deploy)
    api_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""

    st.divider()
    st.markdown("**Hierarchy**")
    st.markdown("""
    <div class="sb-metric"><span class="sb-key">BU</span><span class="sb-val">4 units</span></div>
    <div class="sb-metric"><span class="sb-key">Segments</span><span class="sb-val">24</span></div>
    <div class="sb-metric"><span class="sb-key">Subsegments</span><span class="sb-val">134</span></div>
    <div class="sb-metric"><span class="sb-key">Periods</span><span class="sb-val">1–42 + forecast 43–48</span></div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**Model Performance**")
    st.markdown("""
    <div class="sb-metric"><span class="sb-key">Model</span><span class="sb-val">FlowState-r1.1</span></div>
        <div class="sb-metric"><span class="sb-key">Reconciliation</span><span class="sb-val">Bottom-Up</span></div>
    <div class="sb-metric"><span class="sb-key">R²</span><span class="sb-val">0.9871</span></div>
    <div class="sb-metric"><span class="sb-key">wMAPE</span><span class="sb-val">9.40%</span></div>
    <div class="sb-metric"><span class="sb-key">RMSE</span><span class="sb-val">7.74M €</span></div>
    <div class="sb-metric"><span class="sb-key">MAE</span><span class="sb-val">3.66M €</span></div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.breadcrumb = []
        st.session_state.last_followups = []
        st.rerun()

    if st.button("Export all data (Excel)", use_container_width=True):
        with st.spinner("Building Excel…"):
            excel_bytes = make_excel_bytes()
            st.session_state["sidebar_excel"] = excel_bytes
            st.rerun()

    if st.session_state.get("sidebar_excel"):
        st.download_button(
            "⬇ Download Excel",
            st.session_state["sidebar_excel"],
            "revera_full_data.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="sidebar_excel_dl",
            use_container_width=True,
        )

# ── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:8px;">
  <div style="font-size:18px;font-weight:600;color:#dce8f5;">
    <span style="color:#00e5b8">◆</span> Forecast Intelligence
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:10px;color:#6b7e96;display:flex;align-items:center;gap:8px;">
    SIEMENS ADVANTA · BOTTOM-UP HIERARCHICAL AGGREGATION
    <span style="background:rgba(0,229,184,0.09);color:#00e5b8;padding:2px 8px;border-radius:20px;border:1px solid rgba(0,229,184,0.2);">● model active</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── BREADCRUMB ─────────────────────────────────────────────────────────────
_render_breadcrumb()

# ── QUICK CHIPS ────────────────────────────────────────────────────────────
cols = st.columns(7)
chips = [
    ("Revenue by BU",           "Show revenue overview by BU"),
    ("Top 5 subsegments",        "What are the top 5 subsegments by revenue?"),
    ("Best segment growth",      "Which segment has the highest growth?"),
    ("Compare top 2 segments",   "Compare the top 2 segments by revenue"),
    ("Trend analysis",           "Show trend analysis for all BUs"),
    ("Executive summary",        "Executive summary"),
    ("What changed most?",       "What changed most between Aug/24 and Sep/24?"),
]
for i, (label, query) in enumerate(chips):
    with cols[i]:
        if st.button(label, key=f"chip_{i}", use_container_width=True):
            st.session_state.pending_query = query
            st.rerun()

st.divider()

# ── CHAT HISTORY ───────────────────────────────────────────────────────────
with st.container():
    if not st.session_state.messages:
        st.markdown("""
        <div class="agent-msg">
            <div class="agent-label">Revera</div>
            Hello! I'm <b style="color:#00e5b8">Revera</b>, the Siemens Advanta forecast intelligence agent.
            I have access to Revenue data across <b style="color:#00e5b8">4 BUs, 24 segments and 134 subsegments</b> —
            42 historical periods (Apr/21–Sep/24) plus a 6-period FlowState-r1.1 forecast (Oct/24–Mar/25), Bottom-Up aggregated
            (R² 0.9871 · wMAPE 9.40%).<br><br>
            Ask me anything — I can show overviews, rankings, comparisons, trend analysis, and more.
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f'<div class="user-label">You</div><div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="agent-label">Revera</div><div class="agent-msg">{_md(msg["text"])}</div>', unsafe_allow_html=True)
                if "charts" in msg:
                    for j, chart in enumerate(msg["charts"]):
                        event = st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False}, key=f"chart_{i}_{j}", on_select="rerun")
                        if event and event.get("selection") and event["selection"].get("points"):
                            pt = event["selection"]["points"][0]
                            val = None
                            if "customdata" in pt and pt["customdata"]:
                                val = pt["customdata"][0] if isinstance(pt["customdata"], list) else pt["customdata"]
                            elif "x" in pt:
                                val = pt["x"]
                            if val and isinstance(val, str) and "SSI" in val and st.session_state.get(f"handled_{i}_{j}") != val:
                                st.session_state[f"handled_{i}_{j}"] = val
                                st.session_state.pending_query = f"Tell me about {val}"
                                st.rerun()
                if "tables" in msg:
                    for df in msg["tables"]:
                        st.dataframe(df, use_container_width=True, hide_index=True)
                if "cards" in msg:
                    card_cols = st.columns(len(msg["cards"]))
                    for j, (label, value, sub) in enumerate(msg["cards"]):
                        with card_cols[j]:
                            st.metric(label=label, value=value, delta=sub)
                if "export_df" in msg:
                    btn_cols = st.columns([1, 1, 4])
                    with btn_cols[0]:
                        csv = msg["export_df"].to_csv(index=False).encode()
                        st.download_button("↓ Export CSV", csv, "revera_forecast.csv",
                                           "text/csv", key=f"csv_{i}", use_container_width=True)
                    if msg.get("export_pdf"):
                        with btn_cols[1]:
                            if st.button("↓ Export PDF", key=f"pdf_btn_{i}", use_container_width=True):
                                with st.spinner("Generating PDF report..."):
                                    try:
                                        pdf_bytes = generate_pdf()
                                        st.session_state[f"pdf_ready_{i}"] = pdf_bytes
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"PDF error: {e}")
                    if st.session_state.get(f"pdf_ready_{i}"):
                        st.download_button(
                            "⬇ Download PDF report",
                            st.session_state[f"pdf_ready_{i}"],
                            "revera_report.pdf", "application/pdf",
                            key=f"pdf_dl_{i}", use_container_width=False
                        )

                # ── Follow-up suggestions (only on last assistant message) ──
                if i == len(st.session_state.messages) - 1 and msg.get("followups"):
                    st.markdown('<div class="followup-label">Suggested follow-ups</div>', unsafe_allow_html=True)
                    fu_cols = st.columns(len(msg["followups"]))
                    for j, fu in enumerate(msg["followups"]):
                        with fu_cols[j]:
                            if st.button(fu, key=f"fu_{i}_{j}", use_container_width=True):
                                st.session_state.pending_query = fu
                                st.rerun()

# ── INPUT ──────────────────────────────────────────────────────────────────
st.divider()
query = st.chat_input("Ask about any BU, segment or subsegment...")

if "pending_query" in st.session_state:
    query = st.session_state.pending_query
    del st.session_state.pending_query

# ── PROCESS QUERY ──────────────────────────────────────────────────────────
if query:
    if not api_key:
        st.warning("Groq API key not configured. Add GROQ_API_KEY to Streamlit secrets.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        try:
            history_subset = []
            # Keep last 6 turns; truncate assistant text to 150 chars to save tokens
            for m in st.session_state.messages[-6:]:
                if m["role"] == "user":
                    history_subset.append({"role": "user", "content": m.get("content", "")})
                else:
                    text = (m.get("text") or "")[:150]
                    history_subset.append({"role": "assistant", "content": text})
            history_json = json.dumps(history_subset)
            parsed = parse_intent(query, api_key, history_json)
            _update_breadcrumb(parsed)
            result = build_response(parsed)

            msg = {
                "role":      "assistant",
                "text":      result["text"],
                "id":        len(st.session_state.messages),
                "followups": result.get("followups", []),
            }
            if "charts"    in result: msg["charts"]    = result["charts"]
            if "tables"    in result: msg["tables"]    = result["tables"]
            if "cards"     in result: msg["cards"]     = result["cards"]
            if "export_df"  in result: msg["export_df"]  = result["export_df"]
            if "export_pdf" in result: msg["export_pdf"] = result["export_pdf"]

            st.session_state.messages.append(msg)

        except Exception as e:
            import re as _re
            err_str = str(e)
            # Friendly rate-limit message instead of raw stack trace
            if "rate_limit_exceeded" in err_str or "429" in err_str:
                wait_match = _re.search(r"try again in ([\w.]+)", err_str)
                wait = wait_match.group(1) if wait_match else "a few minutes"
                st.session_state.messages.append({
                    "role": "assistant",
                    "text": (
                        f"⚠️ **Groq daily token limit reached** (free tier: 100k tokens/day). "
                        f"Please try again in **{wait}**.\n\n"
                        "While you wait, you can still use the **Export CSV/PDF/Excel** buttons "
                        "on previous responses, or browse the charts already loaded."
                    )
                })
            else:
                import traceback
                st.session_state.messages.append({
                    "role": "assistant",
                    "text": f"⚠️ Error: {err_str}\n```\n{traceback.format_exc()}\n```"
                })

    st.rerun()
