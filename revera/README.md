# Revera — Forecast Intelligence
**Siemens Advanta · AI Agent · Business Case NOVA 2026**

Natural-language chat interface for exploring hierarchical Revenue forecasts across **4 BUs, 24 segments and 134 subsegments** — 42 months of history (Apr/21–Sep/24) plus a 6-month FlowState-r1.1 forecast (Oct/24–Mar/25), Bottom-Up aggregated (R² 0.9871 · wMAPE 9.40%).

---

## Live App

> **[revera.streamlit.app](https://revera.streamlit.app)**
>
> ⚠️ **The app may be in sleep mode** — Streamlit Community Cloud automatically puts inactive apps to sleep after a period of inactivity. If you see a "This app is sleeping" screen, just click **"Wake up"** and wait ~30 seconds for it to start.

---

## Run locally (5 minutes)

### 1. Get a free Groq API key
Go to **[console.groq.com](https://console.groq.com)** → sign up → API Keys → Create key
No credit card needed. Free tier: 100k tokens/day.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API key
Create the file `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_..."
```

### 4. Run
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## Deploy on Streamlit Cloud (free)

1. Push this repo to GitHub (`.streamlit/secrets.toml` is in `.gitignore` — never committed)
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → New app → select repo and `app.py`
3. Settings → Secrets → paste:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
4. Deploy

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
| Frontend | Streamlit 1.35+ |
| NLU | Groq API — Llama 3.3 70B Versatile (free) |
| Forecast model | FlowState-r1.1 (Google) |
| Reconciliation | Bottom-Up hierarchical aggregation |
| Charts | Plotly (app) · Matplotlib (PDF) |
| PDF export | ReportLab + Matplotlib |
| Excel export | openpyxl |
| Data | SQLite in-memory + pandas |

---

## Tests

```bash
python test_builders.py          # 52 unit tests
python test_builders_hard.py     # 48 edge-case tests
```

---

## Report

A business-oriented executive summary PDF is included at [`revera_report.pdf`](./revera_report.pdf).
