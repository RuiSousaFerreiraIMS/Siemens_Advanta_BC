"""
functions_models.py - ML forecasting pipeline.

Recursive multi-step forecasting with leak-free evaluation
for hierarchical revenue prediction (Siemens Advanta BC).
"""

import re, time, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False
    warnings.warn("CatBoost not installed - skipping CatBoost model.")

# === DEFAULT COLUMN NAMES ===
DEFAULT_TARGET     = 'Revenue cons. (anon)'
DEFAULT_ORDERS_COL = 'Orders cons. (anon)'
DEFAULT_PERIOD_COL = 'Anon Period'
DEFAULT_SUBSEG_COL = 'TGL Business Subsegment'
DEFAULT_SEG_COL    = 'TGL Business Segment'
DEFAULT_BU_COL     = 'TGL Business Unit'

# === METRICS ===
def compute_metrics(y_true, y_pred, model_name, level):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {'Model': model_name, 'Level': level, 'RMSE': np.nan,
                'MAE': np.nan, 'wMAPE': np.nan, 'R2': np.nan, 'N_samples': 0}
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    denom = np.sum(np.abs(yt))
    wmape = np.sum(np.abs(yt - yp)) / denom * 100 if denom > 0 else np.nan
    return {'Model': model_name, 'Level': level, 'RMSE': rmse,
            'MAE': mae, 'wMAPE': wmape, 'R2': r2, 'N_samples': len(yt)}

# === DATA PREPARATION ===
def filter_leaky_features(feature_cols, orders_strategy='drop'):
    if orders_strategy == 'drop':
        blocked = ('ord', 'orders', 'asp')
        return [c for c in feature_cols
                if not any(tok in c.lower() for tok in blocked)]
    return list(feature_cols)

def prepare_subsegment_data(df, val_cutoff, target=DEFAULT_TARGET,
                            orders_col=DEFAULT_ORDERS_COL,
                            period_col=DEFAULT_PERIOD_COL,
                            bu_col=DEFAULT_BU_COL, seg_col=DEFAULT_SEG_COL,
                            subseg_col=DEFAULT_SUBSEG_COL,
                            orders_strategy='drop'):
    cat_cols = [bu_col, seg_col, subseg_col]
    drop_cols = [period_col, target, orders_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    feature_cols = filter_leaky_features(feature_cols, orders_strategy)
    df = df.copy()
    for c in cat_cols:
        if c in feature_cols:
            df[c] = df[c].astype('category')
    train = df[df[period_col] <= val_cutoff].copy()
    val = df[df[period_col] > val_cutoff].copy()
    return (train[feature_cols], train[target].values,
            val[feature_cols], val[target].values, feature_cols, cat_cols)

def prepare_segment_data(seg_df, val_cutoff, target=DEFAULT_TARGET,
                         orders_col=DEFAULT_ORDERS_COL,
                         period_col=DEFAULT_PERIOD_COL,
                         bu_col=DEFAULT_BU_COL, seg_col=DEFAULT_SEG_COL,
                         orders_strategy='drop'):
    cat_cols = [bu_col, seg_col]
    drop_cols = [period_col, target, orders_col]
    feature_cols = [c for c in seg_df.columns if c not in drop_cols]
    feature_cols = filter_leaky_features(feature_cols, orders_strategy)
    seg_df = seg_df.copy()
    for c in cat_cols:
        if c in feature_cols:
            seg_df[c] = seg_df[c].astype('category')
    train = seg_df[seg_df[period_col] <= val_cutoff].copy()
    val = seg_df[seg_df[period_col] > val_cutoff].copy()
    return (train[feature_cols], train[target].values,
            val[feature_cols], val[target].values, feature_cols, cat_cols)

# === LAG FEATURE RECOMPUTATION ===
_RE_PARENT_LAG      = re.compile(r'Parent_Lag_(\d+)', re.I)
_RE_PARENT_RMEAN    = re.compile(r'Parent_Rolling_Mean_(\d+)', re.I)
_RE_PARENT_RSTD     = re.compile(r'Parent_Rolling_Std_(\d+)', re.I)
_RE_PARENT_YOY_DIFF = re.compile(r'Parent_YoY_Diff', re.I)
_RE_PARENT_YOY_RAT  = re.compile(r'Parent_YoY_Ratio', re.I)
_RE_SHARE_PARENT    = re.compile(r'Share_of_Parent_Lag_(\d+)', re.I)
_RE_SHARE_PAR_YOY   = re.compile(r'Share_of_Parent_YoY', re.I)
_RE_LAG          = re.compile(r'_Lag_(\d+)', re.I)
_RE_ROLLING_MEAN = re.compile(r'_Rolling_Mean_(\d+)', re.I)
_RE_ROLLING_STD  = re.compile(r'_Rolling_Std_(\d+)', re.I)
_RE_YOY_DIFF     = re.compile(r'_YoY_Diff$', re.I)
_RE_YOY_RAT      = re.compile(r'_YoY_Ratio$', re.I)
_RE_TREND        = re.compile(r'_Trend_(\d+)', re.I)
_RE_TREND_SLOPE  = re.compile(r'_Trend_Slope_(\d+)', re.I)
_RE_MOMENTUM     = re.compile(r'_Momentum_(\d+)_(\d+)', re.I)
_RE_CV           = re.compile(r'_CV_(\d+)', re.I)
_RE_ZERO_SHARE   = re.compile(r'_Zero_Share_(\d+)', re.I)

_ALL_PATTERNS = [
    _RE_PARENT_LAG, _RE_PARENT_RMEAN, _RE_PARENT_RSTD,
    _RE_PARENT_YOY_DIFF, _RE_PARENT_YOY_RAT,
    _RE_SHARE_PARENT, _RE_SHARE_PAR_YOY,
    _RE_LAG, _RE_ROLLING_MEAN, _RE_ROLLING_STD,
    _RE_YOY_DIFF, _RE_YOY_RAT, _RE_TREND, _RE_TREND_SLOPE,
    _RE_MOMENTUM, _RE_CV, _RE_ZERO_SHARE,
]

def _vals(hist, entity, period, offsets):
    return [hist.get((entity, period - k), np.nan) for k in offsets]

def _nanmean(vals):
    c = [v for v in vals if not np.isnan(v)]
    return np.mean(c) if c else np.nan

def _nanstd(vals, mc=2):
    c = [v for v in vals if not np.isnan(v)]
    return float(np.std(c, ddof=1)) if len(c) >= mc else np.nan

def _slope(vals, mc=3):
    idx = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
    if len(idx) < mc: return np.nan
    xs = np.array([p[0] for p in idx], dtype=float)
    ys = np.array([p[1] for p in idx], dtype=float)
    return float(np.polyfit(xs, ys, 1)[0])

def _is_rev(col):
    cl = col.lower()
    blocked = ('gdp','cpi','france','china','japan','united','macro','pmi','confidence','ord','order','asp')
    return ('rev' in cl or 'revenue' in cl) and not any(t in cl for t in blocked)


def recompute_lag_features(history, period, entity_key, feature_cols):
    """
    Dynamically updates lag and rolling features for a specific entity and date.
    'entity_key' is a tuple (e.g., (BU, Segment)) identifying the series.
    """
    updates = {}
    for col in feature_cols:
        if not _is_rev(col): continue  # Only process revenue-based lags

        # 1. Simple Lags (e.g., Rev_Lag_1)
        m = _RE_LAG.search(col)
        if m:
            lag_val = int(m.group(1))
            updates[col] = history.get((entity_key, period - lag_val), np.nan)
            continue

        # 2. Rolling Means (e.g., Rev_Rolling_Mean_3)
        m = _RE_ROLLING_MEAN.search(col)
        if m:
            window = int(m.group(1))
            vals = [history.get((entity_key, period - x), np.nan) for x in range(1, window + 1)]
            updates[col] = np.nanmean(vals) if any(~np.isnan(vals)) else np.nan
            continue

        # 3. Rolling Standard Deviation
        m = _RE_ROLLING_STD.search(col)
        if m:
            window = int(m.group(1))
            vals = [history.get((entity_key, period - x), np.nan) for x in range(1, window + 1)]
            updates[col] = np.nanstd(vals) if len([x for x in vals if not np.isnan(x)]) >= 2 else np.nan
            continue

        # 4. Trend Slopes (Linear regression over window)
        m = _RE_TREND.search(col) or _RE_TREND_SLOPE.search(col)
        if m:
            window = int(m.group(1))
            vals = [history.get((entity_key, period - x), np.nan) for x in range(window, 0, -1)]
            v_clean = [x for x in vals if not np.isnan(x)]
            if len(v_clean) >= 3:
                updates[col] = np.polyfit(range(len(v_clean)), v_clean, 1)[0]
            else:
                updates[col] = np.nan
            continue
            
        # 5. YoY Difference
        m = _RE_YOY_DIFF.search(col)
        if m:
            val_t1 = history.get((entity_key, period - 1), np.nan)
            val_t13 = history.get((entity_key, period - 13), np.nan)
            if not np.isnan(val_t1) and not np.isnan(val_t13):
                updates[col] = val_t1 - val_t13
            else:
                updates[col] = np.nan
            continue
            
        # 6. YoY Ratio
        m = _RE_YOY_RAT.search(col)
        if m:
            val_t1 = history.get((entity_key, period - 1), np.nan)
            val_t13 = history.get((entity_key, period - 13), np.nan)
            if not np.isnan(val_t1) and not np.isnan(val_t13) and abs(val_t13) > 1e-8:
                updates[col] = val_t1 / val_t13
            else:
                updates[col] = np.nan
            continue

        # 7. Momentum
        m = _RE_MOMENTUM.search(col)
        if m:
            short_w = int(m.group(1))
            long_w  = int(m.group(2))
            vals_s = [history.get((entity_key, period - x), np.nan) for x in range(1, short_w + 1)]
            s_mean = np.nanmean(vals_s) if any(~np.isnan(vals_s)) else np.nan
            
            vals_l = [history.get((entity_key, period - x), np.nan) for x in range(1, long_w + 1)]
            l_mean = np.nanmean(vals_l) if any(~np.isnan(vals_l)) else np.nan
            
            if not np.isnan(s_mean) and not np.isnan(l_mean):
                updates[col] = s_mean - l_mean
            else:
                updates[col] = np.nan
            continue

        # 8. Coefficient of Variation (CV)
        m = _RE_CV.search(col)
        if m:
            w = int(m.group(1))
            vals = [history.get((entity_key, period - x), np.nan) for x in range(1, w + 1)]
            mean_v = np.nanmean(vals) if any(~np.isnan(vals)) else np.nan
            std_v = np.nanstd(vals) if len([x for x in vals if not np.isnan(x)]) >= 2 else np.nan
            if not np.isnan(mean_v) and not np.isnan(std_v) and abs(mean_v) > 1e-8:
                updates[col] = std_v / mean_v
            else:
                updates[col] = np.nan
            continue

        # 9. Zero Share
        m = _RE_ZERO_SHARE.search(col)
        if m:
            w = int(m.group(1))
            vals = [history.get((entity_key, period - x), np.nan) for x in range(1, w + 1)]
            clean_v = [x for x in vals if not np.isnan(x)]
            if len(clean_v) > 0:
                updates[col] = sum(1 for x in clean_v if abs(x) < 1e-8) / w
            else:
                updates[col] = np.nan
            continue

    return updates

def diagnose_feature_coverage(feature_cols, subseg_col=DEFAULT_SUBSEG_COL,
                              seg_col=DEFAULT_SEG_COL, bu_col=DEFAULT_BU_COL):
    recomputed, static, identity, orders_asp, unknown = [], [], [], [], []
    id_cols = {subseg_col, seg_col, bu_col}
    for col in feature_cols:
        if col in id_cols: identity.append(col); continue
        cl = col.lower()
        if any(t in cl for t in ('ord','orders','asp')): orders_asp.append(col); continue
        if _is_rev(col) and any(p.search(col) for p in _ALL_PATTERNS):
            recomputed.append(col); continue
        if any(t in cl for t in ['gdp','month','quarter','year','france','china','japan',
            'united','macro','cpi','inflation','unemployment','interest','exchange',
            'pmi','confidence','industrial','construction','nonzero','since']):
            static.append(col); continue
        unknown.append(col)
    print('=' * 60)
    print('FEATURE COVERAGE DIAGNOSTIC')
    print('=' * 60)
    print(f'  Recomputed (leak-free) : {len(recomputed):>3}')
    print(f'  Static (safe as-is)    : {len(static):>3}')
    print(f'  Identity (ID columns)  : {len(identity):>3}')
    if orders_asp: print(f'  Orders/ASP (LEAKED!)   : {len(orders_asp):>3}  <- drop these!')
    print(f'  UNKNOWN (check these!) : {len(unknown):>3}')
    print('-' * 60)
    if orders_asp:
        print('\nOrders/ASP still present:')
        for c in orders_asp: print(f'   {c}')
    if unknown:
        print('\nUnknown features (check manually):')
        for c in unknown: print(f'   {c}')
    if not orders_asp and not unknown:
        print('\nAll features accounted for. Pipeline is clean.')
    return {'recomputed': recomputed, 'static': static, 'identity': identity,
            'orders_asp': orders_asp, 'unknown': unknown}

# === MODEL DEFINITIONS ===
def get_models(cat_cols, feature_cols):
    cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
    models = {
        'LightGBM': (lgb.LGBMRegressor(
            n_estimators=300,        # era 500
            learning_rate=0.05,
            num_leaves=15,           # era 31 — menos folhas = menos complexidade
            min_child_samples=30,    # era 20 — mais amostras por folha
            subsample=0.7,           # era 0.8
            colsample_bytree=0.7,    # era 0.8
            reg_alpha=0.1,           # L1 — novo
            reg_lambda=1.0,          # L2 — novo
            random_state=42, verbosity=-1, n_jobs=-1), False),

        'XGBoost': (xgb.XGBRegressor(
            n_estimators=300,        # era 500
            learning_rate=0.05,
            max_depth=4,             # era 6
            min_child_weight=5,      # novo — evita splits em poucos pontos
            subsample=0.7,           # era 0.8
            colsample_bytree=0.7,    # era 0.8
            reg_alpha=0.1,           # L1 — novo
            reg_lambda=1.0,          # L2 — novo
            random_state=42, verbosity=0,
            enable_categorical=True, tree_method='hist'), False),

        'Ridge': (Ridge(
            alpha=10.0,              # era 1.0 — mais regularização
            random_state=42), True),

        'Lasso': (Lasso(
            alpha=10.0,              # era 1.0
            random_state=42, max_iter=5000), True),

        'ElasticNet': (ElasticNet(
            alpha=10.0,              # era 1.0
            l1_ratio=0.5,
            random_state=42, max_iter=5000), True),

        'Random Forest': (RandomForestRegressor(
            n_estimators=200,        # era 300
            max_depth=8,             # era 15 — muito profundo para poucos dados
            min_samples_leaf=20,     # era 10
            max_features=0.6,        # novo — subsample de features
            random_state=42, n_jobs=-1), True),

        'Gradient Boosting': (GradientBoostingRegressor(
            n_estimators=200,        # era 300
            learning_rate=0.05,
            max_depth=3,             # era 5
            min_samples_leaf=20,     # era 10
            subsample=0.7,           # novo
            random_state=42), True),
    }
    if _HAS_CATBOOST:
        models['CatBoost'] = (CatBoostRegressor(iterations=500, learning_rate=0.05,
            depth=6, random_seed=42, verbose=0,
            cat_features=cat_idx if cat_idx else None), False)
    return models

# === FIT / PREDICT PIPELINE ===
def _preprocess_arrays(X, cat_cols, feature_cols, imputer=None, scaler=None, fit=True):
    cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
    arr = X.values.copy() if hasattr(X, 'values') else X.copy()
    for i in cat_idx:
        col = X.iloc[:, i] if hasattr(X, 'iloc') else arr[:, i]
        if hasattr(col, 'cat'):
            arr[:, i] = col.cat.codes.values.astype(float)
        else:
            arr[:, i] = pd.Categorical(col).codes.astype(float)
    arr = arr.astype(float)
    arr[~np.isfinite(arr)] = np.nan
    if fit:
        imputer = SimpleImputer(strategy='median')
        arr = imputer.fit_transform(arr)
        scaler = RobustScaler()
        arr = scaler.fit_transform(arr)
    else:
        arr = imputer.transform(arr)
        arr = scaler.transform(arr)
    return arr, imputer, scaler

def fit_model(model, X_train, y_train, needs_preprocessing, cat_cols, feature_cols):
    preprocessors = {}
    if needs_preprocessing:
        X_arr, imp, scl = _preprocess_arrays(X_train, cat_cols, feature_cols, fit=True)
        model.fit(X_arr, y_train)
        preprocessors = {'imputer': imp, 'scaler': scl}
    else:
        fp = {}
        if isinstance(model, lgb.LGBMRegressor):
            fp['categorical_feature'] = [c for c in cat_cols if c in feature_cols]
        model.fit(X_train, y_train, **fp)
    return model, preprocessors

def predict_with_model(model, X_val, needs_preprocessing, preprocessors, cat_cols, feature_cols):
    if needs_preprocessing:
        X_arr, _, _ = _preprocess_arrays(X_val, cat_cols, feature_cols,
            imputer=preprocessors['imputer'], scaler=preprocessors['scaler'], fit=False)
        return model.predict(X_arr)
    return model.predict(X_val)

def train_and_predict(model, X_train, y_train, X_val, needs_preprocessing, cat_cols, feature_cols):
    fitted, pp = fit_model(model, X_train, y_train, needs_preprocessing, cat_cols, feature_cols)
    return predict_with_model(fitted, X_val, needs_preprocessing, pp, cat_cols, feature_cols)

# === RECURSIVE MULTI-STEP FORECASTING ===
def _build_parent_history(train_df, period_col, subseg_col, bu_col, target):
    ph = {}
    for period in train_df[period_col].unique():
        m = train_df[period_col] == period
        for bu, total in train_df.loc[m].groupby(bu_col)[target].sum().items():
            ph[(bu, period)] = total
    return ph

def _update_parent_history(ph, history, df_slice, period, subseg_col, bu_col):
    sums = {}
    for _, row in df_slice.iterrows():
        bu = row[bu_col]
        pred = history.get((row[subseg_col], period), 0.0)
        sums[bu] = sums.get(bu, 0.0) + pred
    for bu, t in sums.items():
        ph[(bu, period)] = t


def recursive_forecast(model, train_df, val_df, feature_cols, group_cols,
                       needs_preprocessing=False, preprocessors=None,
                       cat_cols=None, target=DEFAULT_TARGET, absolute_target=DEFAULT_TARGET,
                       period_col=DEFAULT_PERIOD_COL, predicts_delta=False):
    """
    Performs step-by-step forecasting, updating lags with previous predictions.
    Works for any granularity level defined in 'group_cols'.
    If predicts_delta=True, the model predicts the differenced target. It will 
    automatically reconstruct the absolute target before updating history.
    """
    if cat_cols is None: cat_cols = []

    # Helper to generate a unique key for an entity (supports multi-column keys)
    def get_entity_key(row):
        return tuple(row[c] for c in group_cols) if isinstance(group_cols, list) else row[group_cols]

    # Initialize history with known training values (ALWAYS absolute)
    history = {}
    for _, row in train_df.iterrows():
        history[(get_entity_key(row), row[period_col])] = row[absolute_target]

    val = val_df.copy()
    all_preds_abs = np.zeros(len(val))

    # Security clip to prevent recursive error explosion
    train_max = np.nanmax(np.abs(train_df[absolute_target].values))
    clip_val = train_max * 5

    # Iterate through each period in the validation set sequentially
    for period in sorted(val[period_col].unique()):
        mask = val[period_col] == period
        indices = val.index[mask]

        # A. Update features for the current time step using predicted history
        for i in indices:
            entity = get_entity_key(val.loc[i])
            updates = recompute_lag_features(history, period, entity, feature_cols)
            for col_name, value in updates.items():
                val.at[i, col_name] = value

        # B. Predict current period
        X_p = val.loc[mask, feature_cols]
        if needs_preprocessing and preprocessors:
            preds = predict_with_model(model, X_p, True, preprocessors, cat_cols, feature_cols)
        else:
            preds = model.predict(X_p)

        # C. Reconstruct absolute values if predicting deltas
        if predicts_delta:
            abs_preds = []
            for idx_orig, p_delta in zip(indices, preds):
                entity = get_entity_key(val.loc[idx_orig])
                prev_abs = history.get((entity, period - 1), np.nan)
                if np.isnan(prev_abs): # Fallback
                    prev_abs = 0.0 
                abs_preds.append(prev_abs + p_delta)
            preds = np.array(abs_preds)    

        preds = np.clip(preds, -clip_val, clip_val)

        # D. Update global prediction array and history for next steps
        mask_idx_positions = np.where(mask.values)[0]
        for m_pos, p_val in zip(mask_idx_positions, preds):
            all_preds_abs[m_pos] = p_val

        for idx_orig, p_val in zip(indices, preds):
            history[(get_entity_key(val.loc[idx_orig]), period)] = p_val

    return all_preds_abs

# === BENCHMARKING ===
def run_recursive_benchmark(train_full, val_cutoff, feature_cols, cat_cols,
                            level_name, group_cols,
                            models=None, target=DEFAULT_TARGET, absolute_target=DEFAULT_TARGET,
                            period_col=DEFAULT_PERIOD_COL, predicts_delta=False):
    """
    Clones, fits, and evaluates multiple models using the recursive engine.

    Returns
    -------
    results       : list[dict] — metrics per model (RMSE, MAE, wMAPE, R2, ...)
    all_forecasts : dict[str, dict] — per-model forecasts by entity key
                    Structure: { model_name: { entity_key: np.array(H) } }
                    entity_key matches group_cols (e.g. (bu, seg, sub) for Subsegment)
    """
    if models is None:
        models = get_models(cat_cols, feature_cols)

    # Temporal split
    train_df = train_full[train_full[period_col] <= val_cutoff].copy()
    val_df   = train_full[train_full[period_col] > val_cutoff].copy()
    y_val    = val_df[absolute_target].values

    # Pre-compute entity keys and val periods (needed to build forecast dicts)
    def get_entity_key(row):
        return tuple(row[c] for c in group_cols) if isinstance(group_cols, list) else row[group_cols]

    val_periods  = sorted(val_df[period_col].unique())
    entity_index = [(get_entity_key(val_df.loc[i]), val_df.loc[i, period_col])
                    for i in val_df.index]

    results       = []
    all_forecasts = {}   # { model_name: { entity_key: np.array(H) } }
    all_fitted    = {}   # { model_name: { entity_key: np.array(T) } }

    for name, (template, needs_pp) in models.items():
        print(f'  {name} @ {level_name} ...', end=' ', flush=True)
        t0 = time.time()
        try:
            # ── Fit ────────────────────────────────────────────────────────
            mdl = clone(template)
            X_tr = train_df[feature_cols]
            y_tr = train_df[target].values
            fitted, pp = fit_model(mdl, X_tr, y_tr, needs_pp, cat_cols, feature_cols)

            # Train R² for overfitting diagnostics
            preds_tr  = (predict_with_model(fitted, X_tr, True, pp, cat_cols, feature_cols)
                         if needs_pp and pp else fitted.predict(X_tr))
            train_r2  = r2_score(y_tr, preds_tr)


            # In-sample fitted values por série (para MinT mint_shrink)
            fitted_values = {}
            if needs_pp and pp:
                preds_tr_full = predict_with_model(fitted, X_tr, True, pp, cat_cols, feature_cols)
            else:
                preds_tr_full = fitted.predict(X_tr)

            # Map to entity_key → {period: fitted_value}
            for idx, pred_val in zip(train_df.index, preds_tr_full):
                entity = get_entity_key(train_df.loc[idx])
                period = train_df.loc[idx, period_col]
                if entity not in fitted_values:
                    fitted_values[entity] = {}
                fitted_values[entity][period] = pred_val

            # Map to entity_key → np.array(H) sorted by period
            fitted_values = {
                ek: np.array([pdict[p] for p in sorted(pdict)])
                for ek, pdict in fitted_values.items()
            }
            all_fitted[name] = fitted_values

            # ── Recursive forecast ─────────────────────────────────────────
            preds_flat = recursive_forecast(
                fitted, train_df, val_df, feature_cols,
                group_cols        = group_cols,
                needs_preprocessing = needs_pp,
                preprocessors     = pp,
                cat_cols          = cat_cols,
                target            = target,
                absolute_target   = absolute_target,
                period_col        = period_col,
                predicts_delta    = predicts_delta
            )

            # ── Build per-entity forecast dict ─────────────────────────────
            # preds_flat is a flat array aligned with val_df rows.
            # We reshape it into { entity_key: np.array(H) } for MinT.
            entity_forecasts = {}
            for (entity_key, period), pred_val in zip(entity_index, preds_flat):
                if entity_key not in entity_forecasts:
                    entity_forecasts[entity_key] = {}
                entity_forecasts[entity_key][period] = pred_val

            # Convert inner dict {period: value} → np.array sorted by period
            entity_forecasts = {
                ek: np.array([pdict[p] for p in sorted(pdict)])
                for ek, pdict in entity_forecasts.items()
            }
            all_forecasts[name] = entity_forecasts

            # ── Metrics ────────────────────────────────────────────────────
            m = compute_metrics(y_val, preds_flat, name, level_name)
            m['Time (s)']  = round(time.time() - t0, 1)
            m['Train R2']  = round(train_r2, 6)
            results.append(m)
            print(f'Val RMSE: {m["RMSE"]:>14,.0f} | Train R²: {train_r2:>7.4f} | {m["Time (s)"]}s')

        except Exception as e:
            print(f'FAILED - {str(e)}')
            results.append({'Model': name, 'Level': level_name, 'RMSE': np.nan})
            all_forecasts[name] = {}
            all_fitted[name] = {}

    return results, all_forecasts, all_fitted


# === EXPANDING-WINDOW CV ===
def expanding_window_cv(df, feature_cols, cat_cols, model_template, needs_preproc, group_cols,
                        min_train_periods=30, horizon=6, target=DEFAULT_TARGET, absolute_target=DEFAULT_TARGET,
                        period_col=DEFAULT_PERIOD_COL, subseg_col=DEFAULT_SUBSEG_COL,
                        bu_col=DEFAULT_BU_COL, predicts_delta=False):
    df = df.copy()
    max_p = df[period_col].max()
    for c in cat_cols:
        if c in feature_cols: df[c] = df[c].astype('category')
    folds = []
    for cutoff in range(min_train_periods, max_p - horizon + 1):
        vs, ve = cutoff+1, cutoff+horizon
        tf = df[df[period_col] <= cutoff].copy()
        vf = df[(df[period_col] >= vs) & (df[period_col] <= ve)].copy()
        if len(vf) == 0: continue
        try:
            mdl = clone(model_template)
            fitted, pp = fit_model(mdl, tf[feature_cols], tf[target].values,
                                   needs_preproc, cat_cols, feature_cols)
            preds = recursive_forecast(fitted, tf, vf, feature_cols,
                                       group_cols=group_cols,
                                       needs_preprocessing=needs_preproc, preprocessors=pp,
                                       cat_cols=cat_cols, target=target, absolute_target=absolute_target, period_col=period_col,
                                       predicts_delta=predicts_delta)
            m = compute_metrics(vf[absolute_target].values, preds, '', '')
            folds.append({'cutoff': cutoff, 'val_range': f'{vs}-{ve}',
                'RMSE': m['RMSE'], 'MAE': m['MAE'], 'R2': m['R2'],
                'n_train': len(tf), 'n_val': len(vf)})
        except:
            print(f'Fold {cutoff} failed: {e}')
            folds.append({'cutoff': cutoff, 'val_range': f'{vs}-{ve}',
                'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan,
                'n_train': len(tf), 'n_val': len(vf)})
    return pd.DataFrame(folds)

# === SEGMENT-LEVEL AGGREGATION ===
def build_segment_level_data(df, target=DEFAULT_TARGET, orders_col=DEFAULT_ORDERS_COL,
                             period_col=DEFAULT_PERIOD_COL, bu_col=DEFAULT_BU_COL,
                             seg_col=DEFAULT_SEG_COL):
    """
    Aggregates data at the Business Unit + Segment level and computes lag features.
    """
    # Aggregate to Segment Level
    agg = df.groupby([period_col, bu_col, seg_col]).agg(
        {target: 'sum', orders_col: 'sum'}).reset_index()

    # Carry over macro features (GDP, Industrial, etc.)
    mcols = [c for c in df.columns if any(t in c for t in ['GDP', 'France', 'Month', 'Quarter', 'Industrial'])]
    if mcols:
        mdf = df.groupby(period_col)[mcols].first().reset_index()
        agg = agg.merge(mdf, on=period_col, how='left')

    agg = agg.sort_values([bu_col, seg_col, period_col]).reset_index(drop=True)

    # Feature Engineering: Lags and Rolling Windows
    gk = [bu_col, seg_col]
    g = agg.groupby(gk)[target]

    for lag in [1, 3, 12]:
        agg[f'Rev_Lag_{lag}'] = g.shift(lag)

    for w in [3, 6, 12]:
        agg[f'Rev_Rolling_Mean_{w}'] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        agg[f'Rev_Rolling_Std_{w}'] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=2).std())

    agg['Rev_YoY_Diff'] = g.transform(lambda x: x.shift(1).diff(12))

    # Trend Slopes
    for w in [3, 6]:
        agg[f'Rev_Trend_Slope_{w}'] = g.transform(
            lambda x: x.shift(1).rolling(w, min_periods=3).apply(
                lambda v: (
                    np.polyfit(range(len(v)), v, 1)[0]
                    if len(v) >= 3 and np.isfinite(v).all() and np.std(v) > 0
                    else np.nan
                ),
                raw=True
            )
        )

    print(f'Segment-level dataset: {agg.shape}')
    return agg

# === BU-LEVEL DATA PREPARATION ===
def build_bu_level_data(df, target=DEFAULT_TARGET, orders_col=DEFAULT_ORDERS_COL,
                        period_col=DEFAULT_PERIOD_COL, bu_col=DEFAULT_BU_COL):
    """
    Aggregates data at the Business Unit level and computes lag features.
    """
    agg = df.groupby([period_col, bu_col]).agg(
        {target: 'sum', orders_col: 'sum'}).reset_index()

    # Carry over macro features
    mcols = [c for c in df.columns if any(t in c for t in [
        'GDP', 'France', 'Month', 'Quarter', 'Industrial',
        'China', 'Japan', 'United', 'CPI', 'PMI',
        'Confidence', 'Unemployment', 'Interest', 'Exchange', 'Construction'
    ])]
    if mcols:
        mdf = df.groupby(period_col)[mcols].first().reset_index()
        agg = agg.merge(mdf, on=period_col, how='left')

    agg = agg.sort_values([bu_col, period_col]).reset_index(drop=True)

    g = agg.groupby(bu_col)[target]

    for lag in [1, 3, 12]:
        agg[f'Rev_Lag_{lag}'] = g.shift(lag)

    for w in [3, 6, 12]:
        agg[f'Rev_Rolling_Mean_{w}'] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        agg[f'Rev_Rolling_Std_{w}']  = g.transform(lambda x: x.shift(1).rolling(w, min_periods=2).std())

    agg['Rev_YoY_Diff'] = g.transform(lambda x: x.shift(1).diff(12))

    for w in [3, 6]:
        agg[f'Rev_Trend_Slope_{w}'] = g.transform(
            lambda x: x.shift(1).rolling(w, min_periods=3).apply(
                lambda v: (
                    np.polyfit(range(len(v)), v, 1)[0]
                    if len(v) >= 3 and np.isfinite(v).all() and np.std(v) > 0
                    else np.nan
                ),
                raw=True
            )
        )

    print(f'BU-level dataset: {agg.shape}')
    return agg


def prepare_bu_data(bu_df, val_cutoff, target=DEFAULT_TARGET,
                    orders_col=DEFAULT_ORDERS_COL,
                    period_col=DEFAULT_PERIOD_COL,
                    bu_col=DEFAULT_BU_COL,
                    orders_strategy='drop'):
    cat_cols = [bu_col]
    drop_cols = [period_col, target, orders_col]
    feature_cols = [c for c in bu_df.columns if c not in drop_cols]
    feature_cols = filter_leaky_features(feature_cols, orders_strategy)
    bu_df = bu_df.copy()
    for c in cat_cols:
        if c in feature_cols:
            bu_df[c] = bu_df[c].astype('category')
    train = bu_df[bu_df[period_col] <= val_cutoff].copy()
    val   = bu_df[bu_df[period_col] > val_cutoff].copy()
    return (train[feature_cols], train[target].values,
            val[feature_cols], val[target].values, feature_cols, cat_cols)



import matplotlib.pyplot as plt
import seaborn as sns

def plot_forecast_comparison(
    train_df,
    submission_df,
    period_col,
    target_col,
    best_model_name,
    output_dir,
    forecast_col='Revenue_Predicted',
    hist_periods_label='Historical Revenue (periods 1–42)',
    forecast_periods_label=None
):
    """
    Generates a chart comparing aggregated historical revenue vs. aggregated forecast.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset containing historical values.
    submission_df : pd.DataFrame
        Dataset containing forecasted values.
    period_col : str
        Name of the period column.
    target_col : str
        Name of the historical target column.
    best_model_name : str
        Name of the model (used in title and filename).
    output_dir : Path
        Directory where the PNG file will be saved.
    forecast_col : str, optional
        Name of the forecast column in the submission file.
    hist_periods_label : str, optional
        Label for the historical area.
    forecast_periods_label : str, optional
        Label for the forecast area (default uses model name).
    """

    # Default label for forecast area
    if forecast_periods_label is None:
        forecast_periods_label = f"{best_model_name} Forecast (periods 43–48)"

    # Aggregate historical and forecast values by period
    hist = train_df.groupby(period_col)[target_col].sum()
    fore = submission_df.groupby(period_col)[forecast_col].sum()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 5))

    # Historical area + line
    ax.fill_between(hist.index, hist.values, alpha=0.75,
                    color='#378ADD', label=hist_periods_label)
    ax.plot(hist.index, hist.values, color='#185FA5', linewidth=0.8)

    # Forecast area + line
    ax.fill_between(fore.index, fore.values, alpha=0.80,
                    color='#D85A30', label=forecast_periods_label)
    ax.plot(fore.index, fore.values, color='#993C1D',
            linewidth=2, marker='o', markersize=6, zorder=5)

    # Annotate forecast values (in millions)
    for period, val in fore.items():
        ax.annotate(f'{val/1e6:.0f}M', xy=(period, val), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=8.5, color='#993C1D', fontweight='500')

    # Vertical line separating historical vs forecast
    ax.axvline(hist.index.max() + 0.5, color='#888780',
               linewidth=1.5, linestyle='--', alpha=0.8)
    ax.text(hist.index.max() + 0.7, ax.get_ylim()[1] * 0.97, 'forecast →',
            fontsize=9, color='#888780', va='top')

    # Format y-axis in millions
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M')
    )

    # Labels and title
    ax.set_xlabel('Period', fontsize=11)
    ax.set_ylabel('Total Revenue', fontsize=11)
    ax.set_title(
        f'{best_model_name} — Historical Revenue & Forecast  '
        f'(Subsegment level, all series aggregated)',
        fontsize=12, pad=12
    )

    # Legend and layout
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(1, max(fore.index) + 1)
    sns.despine()
    plt.tight_layout()

    # Save figure
    output_path = output_dir / f'forecast_plot_{best_model_name.lower()}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# MinT RECONCILIATION — Minimum Trace (Wickramasuriya et al. 2019)
# ═══════════════════════════════════════════════════════════════════════════════

def build_hierarchy_map(df, bu_col=DEFAULT_BU_COL, seg_col=DEFAULT_SEG_COL,
                        subseg_col=DEFAULT_SUBSEG_COL):
    """
    Build the hierarchy mapping from the training data.

    Returns
    -------
    hier : dict with keys:
        'bottom_keys'  : list of (bu, seg, sub) — one per bottom-level series
        'seg_keys'     : list of (bu, seg)      — one per segment series
        'bu_keys'      : list of (bu,)          — one per BU series
        'S'            : np.ndarray (n_all × n_bottom) summing matrix
    """
    triples = sorted(set(zip(df[bu_col], df[seg_col], df[subseg_col])))
    seg_keys = sorted(set((bu, seg) for bu, seg, _ in triples))
    bu_keys = sorted(set((bu,) for bu, _, _ in triples))

    bottom_keys = triples
    n_b = len(bottom_keys)
    n_s = len(seg_keys)
    n_u = len(bu_keys)
    n_all = 1 + n_u + n_s + n_b  # Total + BU + Segment + Subsegment

    # Summing matrix S: rows = all nodes, cols = bottom-level only
    # Order: [Total, BU_0..BU_u, Seg_0..Seg_s, Bottom_0..Bottom_b]
    S = np.zeros((n_all, n_b))

    # Total row: sums all bottom
    S[0, :] = 1.0

    # BU rows
    bu_idx = {k: i for i, k in enumerate(bu_keys)}
    for j, (bu, seg, sub) in enumerate(bottom_keys):
        S[1 + bu_idx[(bu,)], j] = 1.0

    # Segment rows
    seg_idx = {k: i for i, k in enumerate(seg_keys)}
    for j, (bu, seg, sub) in enumerate(bottom_keys):
        S[1 + n_u + seg_idx[(bu, seg)], j] = 1.0

    # Bottom rows: identity
    S[1 + n_u + n_s:, :] = np.eye(n_b)

    return {
        'bottom_keys': bottom_keys,
        'seg_keys': seg_keys,
        'bu_keys': bu_keys,
        'S': S,
        'n_total': 1, 'n_bu': n_u, 'n_seg': n_s, 'n_bottom': n_b,
    }


def _assemble_forecast_matrix(fc_sub, fc_seg, fc_bu, hier, horizon):
    """
    Stacks base forecasts from all levels into a (n_all × H) matrix.
    """
    n_all = hier['n_total'] + hier['n_bu'] + hier['n_seg'] + hier['n_bottom']
    Y = np.zeros((n_all, horizon))

    total = np.zeros(horizon)
    for i, bk in enumerate(hier['bu_keys']):
        vals = fc_bu.get(bk, np.zeros(horizon))
        Y[1 + i, :] = vals
        total += vals
    Y[0, :] = total

    n_u = hier['n_bu']
    for i, sk in enumerate(hier['seg_keys']):
        Y[1 + n_u + i, :] = fc_seg.get(sk, np.zeros(horizon))

    n_s = hier['n_seg']
    for i, bk in enumerate(hier['bottom_keys']):
        Y[1 + n_u + n_s + i, :] = fc_sub.get(bk, np.zeros(horizon))

    return Y


def _compute_residual_covariance(fitted_sub, fitted_seg, fitted_bu,
                                 actuals_sub, actuals_seg, actuals_bu,
                                 hier, method='mint_shrink'):
    """
    Compute the W_h matrix (covariance of reconciliation errors).

    method: 'ols' | 'wls_struct' | 'wls_var' | 'mint_shrink'
    """
    n_all = hier['n_total'] + hier['n_bu'] + hier['n_seg'] + hier['n_bottom']

    if method == 'ols':
        return np.eye(n_all)

    if method == 'wls_struct':
        S = hier['S']
        diag_vals = S @ np.ones(hier['n_bottom'])
        return np.diag(diag_vals)

    # For wls_var and mint_shrink, compute residuals
    def _get_residuals(fitted_dict, actuals_dict, keys):
        res_list = []
        for k in keys:
            if k in fitted_dict and k in actuals_dict:
                f = np.asarray(fitted_dict[k], dtype=float)
                a = np.asarray(actuals_dict[k], dtype=float)
                min_len = min(len(f), len(a))
                res_list.append(a[:min_len] - f[:min_len])
            else:
                res_list.append(np.array([0.0]))
        return res_list

    bu_res = _get_residuals(fitted_bu, actuals_bu, hier['bu_keys'])
    max_T = max(len(r) for r in bu_res) if bu_res else 1
    total_res = np.zeros(max_T)
    for r in bu_res:
        total_res[:len(r)] += r

    residual_matrix = [total_res]

    for r in bu_res:
        padded = np.zeros(max_T)
        padded[:len(r)] = r
        residual_matrix.append(padded)

    seg_res = _get_residuals(fitted_seg, actuals_seg, hier['seg_keys'])
    for r in seg_res:
        padded = np.zeros(max_T)
        padded[:len(r)] = r
        residual_matrix.append(padded)

    sub_res = _get_residuals(fitted_sub, actuals_sub, hier['bottom_keys'])
    for r in sub_res:
        padded = np.zeros(max_T)
        padded[:len(r)] = r
        residual_matrix.append(padded)

    E = np.column_stack(residual_matrix)  # (T × n_all)

    if method == 'wls_var':
        variances = np.var(E, axis=0, ddof=1)
        variances = np.maximum(variances, 1e-8)
        return np.diag(variances)

    # mint_shrink: Ledoit-Wolf shrinkage
    T, n = E.shape
    if T < 2:
        return np.eye(n_all)

    E_centered = E - E.mean(axis=0, keepdims=True)
    S_hat = (E_centered.T @ E_centered) / (T - 1)
    F = np.diag(np.diag(S_hat))

    sum_var_sij = 0.0
    sum_sij2 = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s_ij = S_hat[i, j]
            var_sij = np.mean((E_centered[:, i] * E_centered[:, j] - s_ij) ** 2) / T
            sum_var_sij += var_sij
            sum_sij2 += s_ij ** 2

    if sum_sij2 > 0:
        lam = min(max(sum_var_sij / sum_sij2, 0.0), 1.0)
    else:
        lam = 1.0

    W = (1 - lam) * S_hat + lam * F

    eigvals = np.linalg.eigvalsh(W)
    if eigvals.min() < 1e-10:
        W += np.eye(n) * (1e-8 - eigvals.min())

    return W


def mint_reconcile(fc_sub, fc_seg, fc_bu,
                   fitted_sub, fitted_seg, fitted_bu,
                   actuals_sub, actuals_seg, actuals_bu,
                   hier, horizon=6, method='mint_shrink',
                   non_negative=True):
    """
    Reconcile base forecasts using MinT (Minimum Trace).

    Returns
    -------
    reconciled_sub, reconciled_seg, reconciled_bu, reconciled_total
    """
    S = hier['S']
    n_all, n_b = S.shape

    Y_hat = _assemble_forecast_matrix(fc_sub, fc_seg, fc_bu, hier, horizon)

    W = _compute_residual_covariance(
        fitted_sub, fitted_seg, fitted_bu,
        actuals_sub, actuals_seg, actuals_bu,
        hier, method=method
    )

    try:
        W_inv = np.linalg.inv(W)
    except np.linalg.LinAlgError:
        print('  [MinT] W singular — falling back to pseudo-inverse')
        W_inv = np.linalg.pinv(W)

    StWinv = S.T @ W_inv
    StWinvS = StWinv @ S

    try:
        StWinvS_inv = np.linalg.inv(StWinvS)
    except np.linalg.LinAlgError:
        StWinvS_inv = np.linalg.pinv(StWinvS)

    P = StWinvS_inv @ StWinv
    Y_tilde_bottom = P @ Y_hat

    if non_negative:
        Y_tilde_bottom = np.maximum(Y_tilde_bottom, 0.0)

    Y_tilde = S @ Y_tilde_bottom

    n_u = hier['n_bu']
    n_s = hier['n_seg']

    reconciled_total = Y_tilde[0, :]

    reconciled_bu = {}
    for i, bk in enumerate(hier['bu_keys']):
        reconciled_bu[bk] = Y_tilde[1 + i, :]

    reconciled_seg = {}
    for i, sk in enumerate(hier['seg_keys']):
        reconciled_seg[sk] = Y_tilde[1 + n_u + i, :]

    reconciled_sub = {}
    for i, bk in enumerate(hier['bottom_keys']):
        reconciled_sub[bk] = Y_tilde[1 + n_u + n_s + i, :]

    base_total = Y_hat[0, :]
    pct_change = np.abs(reconciled_total - base_total) / (np.abs(base_total) + 1e-8) * 100
    print(f'  [MinT] Avg change per period: {pct_change.mean():.1f}%')

    return reconciled_sub, reconciled_seg, reconciled_bu, reconciled_total


def build_actuals_dict(df, group_cols, target=DEFAULT_TARGET,
                       period_col=DEFAULT_PERIOD_COL):
    """
    Build { entity_key: np.array(T) } of actual values from a DataFrame.
    """
    actuals = {}

    def get_key(row):
        return tuple(row[c] for c in group_cols) if isinstance(group_cols, list) else (row[group_cols],)

    for _, row in df.iterrows():
        k = get_key(row)
        p = row[period_col]
        if k not in actuals:
            actuals[k] = {}
        actuals[k][p] = row[target]

    return {
        ek: np.array([pdict[p] for p in sorted(pdict)])
        for ek, pdict in actuals.items()
    }