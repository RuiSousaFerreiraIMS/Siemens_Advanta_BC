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
    _RE_YOY_DIFF, _RE_TREND, _RE_TREND_SLOPE,
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
    return ('rev' in cl or 'revenue' in cl) and not any(
        t in cl for t in ('gdp','cpi','france','china','japan','united','macro','pmi','confidence'))


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
        'LightGBM': (lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=-1), False),
        'XGBoost': (xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0, enable_categorical=True, tree_method='hist'), False),
        'Ridge': (Ridge(alpha=1.0, random_state=42), True),
        'Lasso': (Lasso(alpha=1.0, random_state=42, max_iter=5000), True),
        'ElasticNet': (ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=5000), True),
        'Random Forest': (RandomForestRegressor(n_estimators=300, max_depth=15,
            min_samples_leaf=10, random_state=42, n_jobs=-1), True),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=300,
            learning_rate=0.05, max_depth=5, min_samples_leaf=10, random_state=42), True),
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
                       cat_cols=None, target=DEFAULT_TARGET,
                       period_col=DEFAULT_PERIOD_COL):
    """
    Performs step-by-step forecasting, updating lags with previous predictions.
    Works for any granularity level defined in 'group_cols'.
    """
    if cat_cols is None: cat_cols = []

    # Helper to generate a unique key for an entity (supports multi-column keys)
    def get_entity_key(row):
        return tuple(row[c] for c in group_cols) if isinstance(group_cols, list) else row[group_cols]

    # Initialize history with known training values
    history = {}
    for _, row in train_df.iterrows():
        history[(get_entity_key(row), row[period_col])] = row[target]

    val = val_df.copy()
    all_preds = np.zeros(len(val))

    # Security clip to prevent recursive error explosion
    train_max = np.nanmax(np.abs(train_df[target].values))
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

        preds = np.clip(preds, -clip_val, clip_val)

        # C. Update global prediction array and history for next steps (t+1, t+2...)
        mask_idx_positions = np.where(mask.values)[0]
        for m_pos, p_val in zip(mask_idx_positions, preds):
            all_preds[m_pos] = p_val

        for idx_orig, p_val in zip(indices, preds):
            history[(get_entity_key(val.loc[idx_orig]), period)] = p_val

    return all_preds

# === BENCHMARKING ===
def run_recursive_benchmark(train_full, val_cutoff, feature_cols, cat_cols,
                            level_name, group_cols,
                            models=None, target=DEFAULT_TARGET,
                            period_col=DEFAULT_PERIOD_COL):
    """
    Clones, fits, and evaluates multiple models using the recursive engine.
    """
    if models is None:
        models = get_models(cat_cols, feature_cols)

    # Temporal Split
    train_df = train_full[train_full[period_col] <= val_cutoff].copy()
    val_df = train_full[train_full[period_col] > val_cutoff].copy()
    y_val = val_df[target].values
    results = []

    for name, (template, needs_pp) in models.items():
        print(f'  {name} @ {level_name} ...', end=' ', flush=True)
        t0 = time.time()
        try:
            # Fit Model
            mdl = clone(template)
            X_tr = train_df[feature_cols]
            y_tr = train_df[target].values
            fitted, pp = fit_model(mdl, X_tr, y_tr, needs_pp, cat_cols, feature_cols)

            # Recursive Prediction
            preds = recursive_forecast(
                fitted, train_df, val_df, feature_cols,
                group_cols=group_cols,
                needs_preprocessing=needs_pp,
                preprocessors=pp,
                cat_cols=cat_cols,
                target=target,
                period_col=period_col
            )

            # Metric Calculation
            m = compute_metrics(y_val, preds, name, level_name)
            m['Time (s)'] = round(time.time() - t0, 1)
            results.append(m)
            print(f'RMSE: {m["RMSE"]:>14,.0f} | {m["Time (s)"]}s')

        except Exception as e:
            print(f'FAILED - {str(e)}')
            results.append({'Model': name, 'Level': level_name, 'RMSE': np.nan})

    return results


# === EXPANDING-WINDOW CV ===
def expanding_window_cv(df, feature_cols, cat_cols, model_template, needs_preproc,
                        min_train_periods=30, horizon=6, target=DEFAULT_TARGET,
                        period_col=DEFAULT_PERIOD_COL, subseg_col=DEFAULT_SUBSEG_COL,
                        bu_col=DEFAULT_BU_COL):
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
                needs_preprocessing=needs_preproc, preprocessors=pp,
                cat_cols=cat_cols, target=target, period_col=period_col,
                subseg_col=subseg_col, bu_col=bu_col)
            m = compute_metrics(vf[target].values, preds, '', '')
            folds.append({'cutoff': cutoff, 'val_range': f'{vs}-{ve}',
                'RMSE': m['RMSE'], 'MAE': m['MAE'], 'R2': m['R2'],
                'n_train': len(tf), 'n_val': len(vf)})
        except:
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

    agg['Rev_YoY_Diff'] = g.diff(12)

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