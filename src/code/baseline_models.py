"""
Baseline time series forecasting models and evaluation utilities.

Functions for:
- Per-subsegment time series extraction
- Naive baselines (naive, seasonal naive, moving average)
- SARIMA (via pmdarima.auto_arima)
- VAR (Vector Autoregression)
- Facebook Prophet
- Amazon Chronos (zero-shot transformer)
- Evaluation metrics (RMSE, MAE, wMAPE)
"""

import os
import sys
import logging
import warnings

# ──────────────────────────────────────────────────────────────────────────────
# EARLY DLL REGISTRATION (Windows only)
# Must happen BEFORE torch or any heavy native library is imported.
# Fixes WinError 127 ("The specified procedure could not be found") by
# ensuring conda Library/bin and torch/lib are on the DLL search path.
# ──────────────────────────────────────────────────────────────────────────────
if os.name == 'nt':
    # 1. Add conda environment's Library/bin
    _lib_bin = os.path.join(sys.prefix, 'Library', 'bin')
    if os.path.isdir(_lib_bin):
        os.environ['PATH'] = _lib_bin + os.pathsep + os.environ.get('PATH', '')
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(_lib_bin)
            except OSError:
                pass

    # 2. Proactively locate and register torch/lib directory
    try:
        import importlib.util as _ilu
        _torch_spec = _ilu.find_spec('torch')
        if _torch_spec and _torch_spec.origin:
            _torch_lib_dir = os.path.join(os.path.dirname(_torch_spec.origin), 'lib')
            if os.path.isdir(_torch_lib_dir):
                os.environ['PATH'] = _torch_lib_dir + os.pathsep + os.environ.get('PATH', '')
                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(_torch_lib_dir)
                    except OSError:
                        pass
            # Also try torch/bin if it exists
            _torch_bin_dir = os.path.join(os.path.dirname(_torch_spec.origin), 'bin')
            if os.path.isdir(_torch_bin_dir):
                os.environ['PATH'] = _torch_bin_dir + os.pathsep + os.environ.get('PATH', '')
                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(_torch_bin_dir)
                    except OSError:
                        pass
    except Exception:
        pass

# Suppress noisy loggers BEFORE any imports that may trigger them
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('prophet.models').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', module='prophet')
warnings.filterwarnings('ignore', module='cmdstanpy')
warnings.filterwarnings('ignore', module='statsmodels')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ──────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_subsegment_series(df, subsegment, period_col, col):
    """
    Extract a clean time series for a given subsegment.
    Returns a pandas Series indexed by period.
    """
    subseg_col = 'TGL Business Subsegment'
    mask = df[subseg_col] == subsegment
    sub = (df.loc[mask, [period_col, col]]
           .sort_values(period_col)
           .set_index(period_col)[col])
    return sub


def get_valid_subsegments(df, target, min_points=2):
    """Get list of subsegments that have enough non-null target data."""
    subseg_col = 'TGL Business Subsegment'
    valid = df.dropna(subset=[target])
    counts = valid.groupby(subseg_col, observed=True).size()
    return sorted(counts[counts >= min_points].index.tolist())


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """
    Compute RMSE, MAE, wMAPE (weighted MAPE), and R².
    
    wMAPE = sum(|actual - pred|) / sum(|actual|) × 100
    
    wMAPE is preferred over standard MAPE for revenue data because:
    - Standard MAPE explodes when actual values are near zero
    - wMAPE naturally weights larger subsegments more heavily
    - It's more interpretable: "on average, the model is off by X% of total revenue"
    
    R² measures how well the predictions explain the variance of the target.
    R² = 1 is perfect; R² = 0 means no better than predicting the mean;
    R² < 0 means worse than predicting the mean.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {'RMSE': np.nan, 'MAE': np.nan, 'wMAPE': np.nan, 'R2': np.nan}
    
    yt, yp = y_true[mask], y_pred[mask]
    
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    r2   = r2_score(yt, yp)
    
    # Weighted MAPE — robust to near-zero values
    total_actual = np.sum(np.abs(yt))
    if total_actual > 1e-8:
        wmape = np.sum(np.abs(yt - yp)) / total_actual * 100
    else:
        wmape = np.nan
    
    return {'RMSE': rmse, 'MAE': mae, 'wMAPE': wmape, 'R2': r2}


def evaluate_model(results_dict, val_df, period_col, target, model_name):
    """Aggregate metrics across all subsegments for a given model."""
    all_true, all_pred = [], []
    for seg, preds in results_dict.items():
        actuals = get_subsegment_series(val_df, seg, period_col, target).dropna().values
        n = min(len(actuals), len(preds))
        if n > 0:
            all_true.extend(actuals[:n])
            all_pred.extend(preds[:n])
    
    metrics = compute_metrics(all_true, all_pred)
    metrics['Model'] = model_name
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# NAIVE BASELINES
# ──────────────────────────────────────────────────────────────────────────────

def naive_forecast(series, horizon):
    """Naïve: repeat the last observed value."""
    last_val = series.dropna().iloc[-1]
    return np.full(horizon, last_val)


def seasonal_naive_forecast(series, horizon, season=12):
    """Seasonal Naïve: repeat values from the same season last year."""
    vals = series.dropna().values
    n = len(vals)
    if n < season:
        return naive_forecast(series, horizon)
    preds = []
    for h in range(horizon):
        idx = n - season + (h % season)
        if idx < 0 or idx >= n:
            preds.append(vals[-1])
        else:
            preds.append(vals[idx])
    return np.array(preds)


def moving_average_forecast(series, horizon, window=3):
    """Moving Average: average of the last `window` values."""
    vals = series.dropna().values
    avg = np.mean(vals[-window:])
    return np.full(horizon, avg)


def run_naive_baselines(subsegments, train_df, val_df, period_col, target):
    """Run all naive baselines on all subsegments. Returns dict of results dicts."""
    naive_res, snaive_res, ma3_res, ma6_res = {}, {}, {}, {}
    
    for seg in subsegments:
        train_s = get_subsegment_series(train_df, seg, period_col, target).dropna()
        val_s   = get_subsegment_series(val_df, seg, period_col, target).dropna()
        
        if len(train_s) < 2 or len(val_s) == 0:
            continue
        
        h = len(val_s)
        naive_res[seg]  = naive_forecast(train_s, h)
        snaive_res[seg] = seasonal_naive_forecast(train_s, h)
        ma3_res[seg]    = moving_average_forecast(train_s, h, window=3)
        ma6_res[seg]    = moving_average_forecast(train_s, h, window=6)
    
    return {
        'Naïve': naive_res,
        'Seasonal Naïve': snaive_res,
        'Moving Avg (3)': ma3_res,
        'Moving Avg (6)': ma6_res,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SARIMA
# ──────────────────────────────────────────────────────────────────────────────

def run_sarima(subsegments, train_df, val_df, period_col, target, verbose=True):
    """
    Fit SARIMA per subsegment using auto_arima.
    Uses seasonal m=12 if enough data (>=24 points), otherwise non-seasonal.
    """
    import pmdarima as pm
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', module='statsmodels')
    
    results, errors = {}, []
    
    for i, seg in enumerate(subsegments):
        train_s = get_subsegment_series(train_df, seg, period_col, target).dropna()
        val_s   = get_subsegment_series(val_df, seg, period_col, target).dropna()
        
        if len(train_s) < 4 or len(val_s) == 0:
            continue
        
        h = len(val_s)
        
        try:
            seasonal = len(train_s) >= 24
            model = pm.auto_arima(
                train_s,
                seasonal=seasonal,
                m=12 if seasonal else 1,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_p=3, max_q=3,
                max_P=2, max_Q=2,
                max_d=2, max_D=1,
                trace=False
            )
            results[seg] = model.predict(n_periods=h)
        except Exception as e:
            errors.append((seg, str(e)))
        
        if verbose and (i + 1) % 25 == 0:
            print(f'  SARIMA progress: {i+1}/{len(subsegments)} subsegments...')
    
    if verbose:
        print(f'\nSARIMA fitted for {len(results)} subsegments. Errors: {len(errors)}')
    
    return results, errors


# ──────────────────────────────────────────────────────────────────────────────
# VAR
# ──────────────────────────────────────────────────────────────────────────────

def run_var(subsegments, train_df, val_df, period_col, target, orders_col, verbose=True):
    """
    Fit VAR (Vector Autoregression) per subsegment.
    Jointly models Revenue + Orders to capture the lead-lag relationship.
    """
    from statsmodels.tsa.api import VAR
    
    results, errors = {}, []
    
    for i, seg in enumerate(subsegments):
        train_rev = get_subsegment_series(train_df, seg, period_col, target).dropna()
        train_ord = get_subsegment_series(train_df, seg, period_col, orders_col).dropna()
        val_s     = get_subsegment_series(val_df, seg, period_col, target).dropna()
        
        common_idx = train_rev.index.intersection(train_ord.index)
        if len(common_idx) < 10 or len(val_s) == 0:
            continue
        
        h = len(val_s)
        
        try:
            mv = pd.DataFrame({
                'Revenue': train_rev.loc[common_idx].values,
                'Orders':  train_ord.loc[common_idx].values
            })
            
            # Skip constant series
            if mv['Revenue'].std() < 1e-8 or mv['Orders'].std() < 1e-8:
                continue
            
            model = VAR(mv)
            max_lag = min(6, len(mv) // 3)
            if max_lag < 1:
                continue
            
            # Select optimal lag order using AIC
            lag_order = model.select_order(maxlags=max_lag)
            best_lag = max(lag_order.aic, 1)
            best_lag = min(best_lag, max_lag)
            
            fitted = model.fit(best_lag)
            forecast = fitted.forecast(mv.values[-best_lag:], steps=h)
            results[seg] = forecast[:, 0]  # Revenue column
            
        except Exception as e:
            errors.append((seg, str(e)))
        
        if verbose and (i + 1) % 25 == 0:
            print(f'  VAR progress: {i+1}/{len(subsegments)} subsegments...')
    
    if verbose:
        print(f'\nVAR fitted for {len(results)} subsegments. Errors: {len(errors)}')
    
    return results, errors


# ──────────────────────────────────────────────────────────────────────────────
# PROPHET
# ──────────────────────────────────────────────────────────────────────────────

def run_prophet(subsegments, train_df, val_df, period_col, target,
                changepoint_prior_scale=0.01, verbose=True):
    """
    Fit Facebook Prophet per subsegment.
    Creates synthetic monthly dates from Anon Period.
    """
    from prophet import Prophet
    # cmdstanpy adds its own StreamHandler at import time.
    # Completely disable it to silence the noisy chain processing logs.
    _cmdstanpy_logger = logging.getLogger('cmdstanpy')
    _cmdstanpy_logger.disabled = True
    _cmdstanpy_logger.handlers = []
    _cmdstanpy_logger.propagate = False
    
    results, errors = {}, []
    
    for i, seg in enumerate(subsegments):
        train_s = get_subsegment_series(train_df, seg, period_col, target).dropna()
        val_s   = get_subsegment_series(val_df, seg, period_col, target).dropna()
        
        if len(train_s) < 6 or len(val_s) == 0:
            continue
        
        h = len(val_s)
        
        try:
            prophet_df = pd.DataFrame({
                'ds': pd.date_range(start='2020-01-01', periods=len(train_s), freq='MS'),
                'y':  train_s.values
            })
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_mode='multiplicative'
            )
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=h, freq='MS')
            forecast = model.predict(future)
            results[seg] = forecast['yhat'].iloc[-h:].values
            
        except Exception as e:
            errors.append((seg, str(e)))
        
        if verbose and (i + 1) % 25 == 0:
            print(f'  Prophet progress: {i+1}/{len(subsegments)} subsegments...')
    
    if verbose:
        print(f'\nProphet fitted for {len(results)} subsegments. Errors: {len(errors)}')
    
    return results, errors


# ──────────────────────────────────────────────────────────────────────────────
# CHRONOS (Zero-Shot)
# ──────────────────────────────────────────────────────────────────────────────

def run_chronos(subsegments, train_df, val_df, period_col, target,
                model_name="amazon/chronos-t5-small", verbose=True):
    """
    Run Amazon Chronos zero-shot forecasting per subsegment.
    Returns results dict and errors list.
    Handles PyTorch/CUDA availability gracefully.
    """
    try:
        # DLL paths are registered at module level (top of this file)
        import torch
        from chronos import ChronosPipeline
    except (ImportError, OSError) as e:
        print(f'⚠️ Chronos unavailable: {e}')
        print('  Skipping Chronos — install torch + chronos-forecasting to enable.')
        return {}, [('all', str(e))]
    
    try:
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    except Exception as e:
        print(f'⚠️ Failed to load Chronos model: {e}')
        return {}, [('model_load', str(e))]
    
    if verbose:
        print('Chronos model loaded successfully.')
    
    results, errors = {}, []
    
    for i, seg in enumerate(subsegments):
        train_s = get_subsegment_series(train_df, seg, period_col, target).dropna()
        val_s   = get_subsegment_series(val_df, seg, period_col, target).dropna()
        
        if len(train_s) < 4 or len(val_s) == 0:
            continue
        
        h = len(val_s)
        
        try:
            context = torch.tensor(train_s.values, dtype=torch.float32)
            forecast = pipeline.predict(context, prediction_length=h, num_samples=20)
            results[seg] = np.median(forecast[0].numpy(), axis=0)
        except Exception as e:
            errors.append((seg, str(e)))
        
        if verbose and (i + 1) % 25 == 0:
            print(f'  Chronos progress: {i+1}/{len(subsegments)} subsegments...')
    
    if verbose:
        print(f'\nChronos predicted for {len(results)} subsegments. Errors: {len(errors)}')
    
    return results, errors


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(comparison_df):
    """Bar chart comparing all models on RMSE, MAE, and wMAPE."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, metric in zip(axes, ['RMSE', 'MAE', 'wMAPE']):
        data = comparison_df.dropna(subset=[metric]).sort_values(metric)
        if data.empty:
            continue
        colors = ['#55A868' if m == data[metric].min() else '#4C72B0' for m in data[metric]]
        
        bars = ax.barh(data['Model'], data[metric], color=colors)
        ax.set_xlabel(metric)
        ax.set_title(f'{metric} by Model', fontweight='bold')
        
        for bar, val in zip(bars, data[metric]):
            fmt = f'{val:,.0f}' if metric != 'wMAPE' else f'{val:.1f}%'
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                    fmt, va='center', fontsize=9)
        
        ax.set_xlim(0, data[metric].max() * 1.25)
    
    plt.suptitle('Baseline Model Comparison — Validation',
                 fontsize=14, fontweight='bold', y=1.02)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(all_results, train_full, val_df, period_col, target,
                             val_cutoff, n_plots=4):
    """
    Plot actual vs predicted for representative subsegments.
    Selects the best and worst subsegments by per-subsegment RMSE
    (using the first model in the dict as reference) so the plot
    shows a range of model performance.
    """
    import matplotlib.pyplot as plt
    
    # Find subsegments common to all models
    common = None
    for model_name, res in all_results.items():
        segs = set(res.keys())
        common = segs if common is None else common & segs
    
    if not common:
        print('No common subsegments across all models for plotting.')
        return
    
    # Rank subsegments by per-subsegment RMSE (using first model) to pick best + worst
    ref_model_name = list(all_results.keys())[0]
    ref_results = all_results[ref_model_name]
    seg_rmse = {}
    for seg in common:
        actuals = get_subsegment_series(val_df, seg, period_col, target).dropna().values
        preds = ref_results[seg]
        n = min(len(actuals), len(preds))
        if n > 0:
            seg_rmse[seg] = np.sqrt(mean_squared_error(actuals[:n], preds[:n]))
    
    ranked = sorted(seg_rmse.keys(), key=lambda s: seg_rmse[s])
    half = n_plots // 2
    best_segs  = ranked[:half]
    worst_segs = ranked[-half:]
    plot_segs = best_segs + worst_segs
    
    fig, axes = plt.subplots(len(plot_segs), 1, figsize=(14, 4 * len(plot_segs)))
    if len(plot_segs) == 1:
        axes = [axes]
    
    markers = {
        'Naïve': 'p-',
        'Seasonal Naïve': 'h-',
        'Moving Avg (3)': '+-',
        'Moving Avg (6)': 'x-',
        'SARIMA': 's-',
        'VAR': '^-',
        'Prophet': 'D-',
        'Chronos (Zero-Shot)': 'v-',
    }
    
    for idx, (ax, seg) in enumerate(zip(axes, plot_segs)):
        full_s = get_subsegment_series(train_full, seg, period_col, target).dropna()
        val_s  = get_subsegment_series(val_df, seg, period_col, target).dropna()
        val_periods = val_s.index
        
        ax.plot(full_s.index, full_s.values, 'k-', label='Training', linewidth=1.5)
        ax.plot(val_periods, val_s.values, 'ko--', label='Actual (Val)', markersize=5)
        
        for model_name, res in all_results.items():
            if seg in res:
                h = len(val_s)
                mk = markers.get(model_name, 'x-')
                ax.plot(val_periods[:len(res[seg])], res[seg][:h],
                        mk, label=model_name, alpha=0.8)
        
        ax.axvline(x=val_cutoff + 0.5, color='red', linestyle='--', alpha=0.5,
                   label='Train/Val split')
        
        category = 'BEST' if idx < half else 'WORST'
        ax.set_title(f'[{category}] Subsegment: {seg} (RMSE: {seg_rmse[seg]:,.0f})',
                     fontweight='bold')
        ax.set_xlabel('Anon Period')
        ax.set_ylabel('Revenue')
        ax.legend(fontsize=8, loc='upper left')
    
    plt.suptitle('Actual vs Predicted — Best & Worst Subsegments',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()
