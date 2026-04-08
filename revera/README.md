# Revera — Forecast Intelligence
**Siemens Advanta · AI Agent · Business Case NOVA 2026**

Natural-language chat interface for exploring hierarchical Revenue forecasts across **4 BUs, 24 segments and 134 subsegments** — 42 months of history (Apr/21–Sep/24) plus a 6-month FlowState-r1.1 forecast (Oct/24–Mar/25), Bottom-Up aggregated (R² 0.9871 · wMAPE 9.40%).

## Live App

**[https://revera-werjwwlwytprqbskpvtehd.streamlit.app](https://revera-werjwwlwytprqbskpvtehd.streamlit.app)**

> ⚠️ **If the app shows a "This app is sleeping" screen** — Streamlit Community Cloud automatically puts inactive apps to sleep after a period of inactivity. Just click **"Wake up"** and wait ~30 seconds for it to start.

---

## What you can ask

| Intent | Example query |
|---|---|
| Executive summary | "Executive summary" |
| Overview / ranking | "Show revenue by BU", "Top 5 segments by revenue" |
| Drilldown | "Drill down into SSI037" |
| Compare | "Compare SSI047 vs SSI027" |
| Period diff | "What changed between Aug/24 and Sep/24?" |
| Trend analysis | "Show trend analysis for all BUs" |
| Forecast | "Show FlowState forecast for all segments" |
| Growth heatmap | "Show growth heatmap by segment" |
| Model metrics | "Show model performance metrics" |

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| NLU | Groq API — Llama 3.3 70B Versatile |
| Forecast model | FlowState-r1.1 (Google) |
| Reconciliation | Bottom-Up hierarchical aggregation |
| Charts | Plotly · Matplotlib |
| Exports | PDF (ReportLab) · Excel (openpyxl) · CSV |

---

## Report

Executive summary PDF: [`revera_report.pdf`](./revera_report.pdf)
