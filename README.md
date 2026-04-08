# Siemens SI — Hierarchical Revenue Forecasting

**NOVA IMS · MSc Data Science & Advanced Analytics · Siemens Advanta Consulting Case Study · 2025/26**

A hierarchical forecasting system for Siemens Smart Infrastructure (SI) revenue, ensuring coherent predictions across four organizational levels: Subsegment, Segment, Business Unit, and SI Total. The project benchmarks 20+ modelling approaches (from classical time series to foundation models) and implements a middle-out hybrid strategy for production submission.

## Problem Statement

Siemens SI needs reliable, hierarchically coherent revenue forecasts at the GCK (subsegment) level that remain consistent when aggregated upward through the organizational hierarchy. The challenge combines two dimensions: forecasting accuracy at the most granular level (134 subsegments, 6 periods ahead) and hierarchical consistency across all aggregation levels.

## Experimental Setup

| Parameter | Value |
|---|---|
| Training | Periods 1 to 36 |
| Validation | Periods 37 to 42 |
| Test (blind) | Periods 43 to 48 |
| Target | Revenue cons. (anonymized) |
| Series count | 134 TGL Business Subsegments |
| Forecast horizon | 6 periods ahead |
| Evaluation | RMSE, MAE, wMAPE, R², Siemens Hierarchical Scorer (NWRMSE) |

## Modelling Approaches

The project explores four distinct forecasting strategies, each tested across multiple model families:

**Time Series Baselines** operate on raw revenue series without engineered features. Models include Moving Average (window 3 and 6), Seasonal Naive, Prophet, ETS (Holt-Winters), SARIMA, VAR, and Naive persistence. Moving Average (window=6) proved the strongest baseline with an RMSE of 9.71M and wMAPE of 12.0%.

**ML Recursive Forecasting** trains a single model that predicts one step ahead and feeds its own predictions back as input for subsequent steps. Tested with XGBoost, LightGBM, CatBoost, Random Forest, Gradient Boosting, Ridge, Lasso, and ElasticNet at Subsegment, Segment, and BU levels. Features include lag values, rolling statistics, and calendar encodings.

**ML Direct Multi-Step Forecasting** trains a separate model for each forecast horizon (h=1 through h=6), eliminating recursive error accumulation. Same model families as above. CatBoost achieved the best subsegment RMSE (10.27M), while XGBoost was selected for the final pipeline due to superior cross-level stability.

**Foundation Models (Zero-Shot)** apply pretrained time series transformers with no task-specific training or feature engineering. Models tested: FlowState-r1.1, Amazon Chronos-2, and Google TimesFM-2.5.

**Walk-Forward Cross-Validation** uses an expanding window strategy across 6 folds, retraining on all available data up to each cutoff point. This provides the most statistically robust performance estimates for supervised models.

## Key Results

### Siemens Hierarchical Scorer (Validation, Periods 37 to 42)

The official evaluation metric combines Normalized Weighted RMSE across three hierarchy levels:

`Final Revenue Error = 0.50 * NWRMSE_subseg + 0.25 * NWRMSE_seg + 0.25 * NWRMSE_BU`

**Top performers by approach:**

| Model | Approach | Final Error | Deviation |
|---|---|---|---|
| FlowState-r1.1 | Foundation (zero-shot) | 0.0830 | ~8.3% |
| Chronos-2 | Foundation (zero-shot) | 0.0839 | ~8.4% |
| XGBoost | Walk-Forward CV | 0.0926 | ~9.3% |
| CatBoost | Walk-Forward CV | 0.0944 | ~9.4% |
| LightGBM | Walk-Forward CV | 0.0968 | ~9.7% |
| TimesFM-2.5 | Foundation (zero-shot) | ~0.10 | ~10% |
| CatBoost | Direct Multi-Step | 0.1185 | ~11.9% |

### Key Findings

Foundation models lead the overall ranking, achieving roughly 40% lower error than the best supervised ML approaches, with zero training and no feature engineering required.

Walk-Forward CV is the strongest supervised strategy. The expanding-window approach produces better-calibrated forecasts across all hierarchy levels compared to single-split training.

Linear models (Ridge, Lasso, ElasticNet) collapse at subsegment granularity in recursive mode (wMAPE > 5000%), but remain viable at segment/BU levels or when trained per-horizon (Direct Multi-Step) or per-fold (Walk-Forward).

**Caveat:** Walk-Forward validation folds predict only one period ahead using the latest actuals. In the blind test (periods 43 to 48), models must forecast 6 periods ahead without intermediate actuals, so WF scores are likely optimistic. Foundation Model and Direct Multi-Step results better approximate true test conditions.

## Final Pipeline: Middle-Out Hybrid Strategy

The submission pipeline applies a hybrid assignment rule at subsegment level:

| Subsegment Type | Assigned Model |
|---|---|
| Predictable, sufficient history | XGBoost Direct Multi-Step + MinT reconciliation |
| Erratic / low-volume | Moving Average (window=6) |
| Dead series (zero revenue, no orders) | Zero forecast |

Forecasts are generated at subsegment level and reconciled upward using MinT (Minimum Trace) to ensure hierarchical coherence.

## Repository Structure

```
Siemens_Advanta_BC/
│
├── data/                               # Data pipeline (gitignored)
│   ├── raw/                            # Original Siemens data files
│   ├── prepared/                       # Cleaned train/val splits by hierarchy level
│   ├── features/                       # Engineered and selected feature sets
│   ├── processed/                      # Intermediate processing outputs
│   └── results/                        # Model predictions, benchmarking exports
│
├── notebooks/                          # Analysis and modelling notebooks
│   ├── 1)Data_Understanding.ipynb      # Initial data exploration
│   ├── 2)Data_Preparation.ipynb        # Cleaning, splitting, hierarchy mapping
│   ├── 3)EDA.ipynb                     # Exploratory data analysis
│   ├── 4)Feature_Engineering.ipynb     # Lag features, rolling stats, macro indicators
│   ├── 5)Feature_Selection.ipynb       # Feature importance and selection
│   ├── 6a)Modelling_Base.ipynb         # TS baselines + ML recursive forecasting
│   ├── 6b)Modelling_MultiStep.ipynb    # ML direct multi-step forecasting
│   ├── 6c)PretrainedModels/            # Foundation model experiments
│   │   ├── chronos_restructured.ipynb  #   Amazon Chronos-2
│   │   ├── flowstate_restructured.ipynb#   FlowState-r1.1
│   │   └── timesfm_restructured.ipynb  #   Google TimesFM-2.5
│   ├── 6d)Modelling_WalkForward_Validation.ipynb  # Expanding window CV
│   ├── 7)Benchmarking_Results.ipynb    # Consolidated results and Siemens scorer
│   └── X)OpenEnd.ipynb                 # Experimental / exploratory work
│
├── src/code/                           # Reusable Python modules
│   ├── data_load.py                    # Data loading utilities
│   ├── data_preparation.py             # Preprocessing and splitting functions
│   ├── eda_functions.py                # EDA helper functions
│   ├── functions_models.py             # Core modelling: lag recomputation, recursive
│   │                                   #   forecasting, expanding window CV
│   ├── baseline_models.py              # Time series baseline implementations
│   ├── hierarchical_functions.py       # Hierarchy building and MinT reconciliation
│   ├── hierarchical_single_file_scorer.py  # Siemens official scorer (NWRMSE)
│   ├── io_utils.py                     # I/O and export utilities
│   └── class_pipeline_functions.py     # Class-based pipeline (legacy)
│
├── revera/                             # AI forecast intelligence agent
│   ├── app.py                          # Streamlit chat interface
│   ├── builders.py                     # Intent handlers, charts, tables
│   ├── data.py                         # Historical + forecast data (Apr/21–Mar/25)
│   ├── nlu.py                          # Groq API / Llama 3.3 70B intent parsing
│   ├── pdf_export.py                   # Executive PDF report generator
│   ├── requirements.txt                # Python dependencies
│   ├── revera_report.pdf               # Executive summary report
│   └── README.md                       # Revera usage and stack
│
├── MLProjectOld/                       # Previous iteration (archived)
├── requirements.txt
├── .gitignore
└── README.md

```

## Data

The project uses two anonymized datasets provided by Siemens:

**Sales data** (`Case2_data_extract_share.xlsx`) contains ~4,200 training rows and ~700 validation rows with columns for anonymized period, hierarchy identifiers (BU, Segment, Subsegment), and two KPIs: Orders and Revenue (both anonymized and consolidated).

**Market data** (`Case2_market_data_share.xlsx`) provides 78 macroeconomic indicators (GDP, Industrial Production, Inflation, and others) across 9 countries and 180 monthly periods, along with a period-to-calendar mapping.

> Data files are not included in this repository. Place them in `data/raw/` after cloning.

## Setup

```bash
git clone https://github.com/<username>/Siemens_Advanta_BC.git
cd Siemens_Advanta_BC
pip install -r requirements.txt
```

Place the Siemens data files in `data/raw/` and run the notebooks sequentially (1 through 7).

## Tech Stack

**Core:** Python 3.13, Pandas, NumPy, Scikit-learn

**ML Models:** XGBoost, LightGBM, CatBoost, Scikit-learn (Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting)

**Time Series:** Prophet, Statsmodels (SARIMA, ETS, VAR)

**Foundation Models:** Amazon Chronos-2, Google TimesFM-2.5, FlowState-r1.1

**Reconciliation:** MinT (Minimum Trace) hierarchical reconciliation

**Visualization:** Matplotlib, Seaborn

**Environment:** PyCharm, Jupyter Notebooks

## Authors

Developed as a capstone case study for the MSc in Data Science & Advanced Analytics at NOVA IMS, in partnership with Siemens Advanta Consulting.

## License

This repository is private and intended for academic evaluation only. The data is proprietary to Siemens and is not distributed with this repository.
