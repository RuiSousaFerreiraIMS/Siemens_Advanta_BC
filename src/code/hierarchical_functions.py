"""
hierarchical_functions.py - ML forecasting pipeline.
"""
import pandas as pd
import numpy as np
from hierarchicalforecast.reconciliation import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace
from src.code.functions_models import compute_metrics, run_recursive_benchmark


def prepare_mint_inputs(train_full, forecasts_sub, forecasts_seg, forecasts_bu,
                        fitted_sub, fitted_seg, fitted_bu,
                        target, period_col, bu_col, seg_col, subseg_col, val_periods):
    """
    Formats the hierarchy prediction objects into specific DataFrames required by Nixtla.
    Y_hat_df MUST only contain OUT-OF-SAMPLE forecasts (val_periods).
    Y_df MUST contain ACTUALS (all periods) AND IN-SAMPLE PREDICTIONS (train_periods) under 'base_model'.
    """
    n_periods      = len(val_periods)
    records_hat    = []
    records_y      = []
    records_fitted = []

    # --- Subsegment ---
    for (bu, seg, sub), preds in forecasts_sub.items():
        if len(preds) != n_periods:
            continue
        uid  = f'{bu}/{seg}/{sub}'
        mask = ((train_full[bu_col]     == bu)  &
                (train_full[seg_col]    == seg) &
                (train_full[subseg_col] == sub))

        # Actuals
        for _, row in train_full[mask].iterrows():
            records_y.append({'unique_id': uid, 'ds': row[period_col], 'y': row[target]})

        # In-sample fitted values (treino)
        entity_key = (bu, seg, sub)
        if entity_key not in fitted_sub and uid in fitted_sub:
            entity_key = uid
            
        if entity_key in fitted_sub:
            fitted_arr = fitted_sub[entity_key]
            sub_train  = train_full[mask].sort_values(period_col)
            for period, fv in zip(sub_train[period_col].values, fitted_arr):
                records_fitted.append({'unique_id': uid, 'ds': period, 'base_model': fv})

        # Out-of-sample forecasts (validação)
        for t, p in zip(val_periods, preds):
            records_hat.append({'unique_id': uid, 'ds': t, 'base_model': p})

    # --- Segment ---
    seg_agg = train_full.groupby([period_col, bu_col, seg_col])[target].sum().reset_index()
    for (bu, seg), preds in forecasts_seg.items():
        if len(preds) != n_periods:
            continue
        uid  = f'{bu}/{seg}'
        mask = (seg_agg[bu_col] == bu) & (seg_agg[seg_col] == seg)
        
        # Actuals
        for _, row in seg_agg[mask].iterrows():
            records_y.append({'unique_id': uid, 'ds': row[period_col], 'y': row[target]})

        # In-sample fitted values
        entity_key = (bu, seg)
        if entity_key not in fitted_seg and uid in fitted_seg:
            entity_key = uid
            
        if entity_key in fitted_seg:
            fitted_arr = fitted_seg[entity_key]
            seg_train  = seg_agg[mask].sort_values(period_col)
            for period, fv in zip(seg_train[period_col].values, fitted_arr):
                records_fitted.append({'unique_id': uid, 'ds': period, 'base_model': fv})

        # Out-of-sample forecasts
        for t, p in zip(val_periods, preds):
            records_hat.append({'unique_id': uid, 'ds': t, 'base_model': p})

    # --- BU ---
    bu_agg = train_full.groupby([period_col, bu_col])[target].sum().reset_index()
    for bu, preds in forecasts_bu.items():
        if len(preds) != n_periods:
            continue
        uid  = f'{bu}'
        mask = bu_agg[bu_col] == bu
        
        # Actuals
        for _, row in bu_agg[mask].iterrows():
            records_y.append({'unique_id': uid, 'ds': row[period_col], 'y': row[target]})

        # In-sample fitted values
        entity_key = bu if bu in fitted_bu else (bu,)
        if entity_key in fitted_bu:
            fitted_arr = fitted_bu[entity_key]
            bu_train   = bu_agg[mask].sort_values(period_col)
            for period, fv in zip(bu_train[period_col].values, fitted_arr):
                records_fitted.append({'unique_id': uid, 'ds': period, 'base_model': fv})

        # Out-of-sample forecasts
        for t, p in zip(val_periods, preds):
            records_hat.append({'unique_id': uid, 'ds': t, 'base_model': p})

    # Validate output length
    if len(records_hat) == 0:
        raise ValueError("Critical Error: Y_hat_df has no out-of-sample forecasts generated!")

    Y_hat_df  = pd.DataFrame(records_hat).sort_values(['unique_id', 'ds']).reset_index(drop=True)
    Y_df_base = pd.DataFrame(records_y)
    Y_fit_df  = pd.DataFrame(records_fitted)

    # Merge in-sample residuals into Y_df explicitly for MinT Shrink
    if len(Y_fit_df) > 0:
        Y_df = pd.merge(Y_df_base, Y_fit_df, on=['unique_id', 'ds'], how='left')
    else:
        Y_df = Y_df_base
        
    Y_df = Y_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    val_counts = Y_hat_df.groupby('unique_id')['ds'].count()
    if len(val_counts) == 0:
        print("CRITICAL ERROR: Y_hat_df has no out-of-sample forecasts!")
    else:
        print(f"Prepared MIN-T Inputs: {len(val_counts)} hierarchy nodes successfully mapped with residuals.")

    return Y_hat_df, Y_df

# ── Step 2: Build the summing matrix S ─────────────────────────────────────

def build_S_matrix(Y_hat_df):
    """
    Builds the summing matrix S that encodes the hierarchy structure purely
    based on the series that successfully produced forecasts.
    
    Rows    = all nodes (BUs + Segments + Subsegments)
    Columns = bottom-level series (Subsegments only)
    """
    sub_ids = Y_hat_df['unique_id'][Y_hat_df['unique_id'].str.count('/') == 2].unique().tolist()
    seg_ids = Y_hat_df['unique_id'][Y_hat_df['unique_id'].str.count('/') == 1].unique().tolist()
    bu_ids  = Y_hat_df['unique_id'][Y_hat_df['unique_id'].str.count('/') == 0].unique().tolist()

    all_ids = bu_ids + seg_ids + sub_ids
    S = pd.DataFrame(0, index=all_ids, columns=sub_ids)

    for sub_id in sub_ids:
        parts = sub_id.split('/')
        if len(parts) == 3:
            bu, seg = parts[0], parts[1]
            S.loc[sub_id, sub_id] = 1
            
            seg_id = f'{bu}/{seg}'
            if seg_id in S.index:
                S.loc[seg_id, sub_id] = 1
                
            bu_id = bu
            if bu_id in S.index:
                S.loc[bu_id, sub_id] = 1

    return S.reset_index().rename(columns={'index': 'unique_id'})


# ── Step 3: Apply MinT reconciliation ──────────────────────────────────────

def apply_mint(Y_hat_df, Y_df, S_df, method='mint_shrink'):
    """
    Applies MinT (Minimum Trace) reconciliation to base forecasts.

    MinT finds the reconciled forecasts ỹ that are:
      1. Coherent  — satisfy all aggregation constraints in S
      2. Optimal   — minimise the trace of the reconciled forecast error
                     covariance matrix

    The closed-form solution is:
        ỹ = S (S' W⁻¹ S)⁻¹ S' W⁻¹ ŷ

    where W is estimated from in-sample residuals.

    Parameters
    ----------
    Y_hat_df : pd.DataFrame — base forecasts (unique_id, ds, base_model)
    Y_df     : pd.DataFrame — historical actuals  (unique_id, ds, y)
    S_df     : pd.DataFrame — summing matrix from build_S_matrix()
    method   : str — covariance estimator for W:
                 'ols'         — identity (assumes homogeneous errors)
                 'wls_var'     — diagonal with per-series variances
                 'mint_sample' — full sample covariance (needs many series)
                 'mint_shrink' — shrinkage estimator (recommended)

    Returns
    -------
    reconciled : pd.DataFrame — reconciled forecasts with column
                                'base_model/MinTrace_method-<method>'
    """
    hrec = HierarchicalReconciliation(reconcilers=[
        MinTrace(method=method)
    ])

    tags = {
        'BU':         Y_hat_df['unique_id'][Y_hat_df['unique_id'].str.count('/') == 0].unique(),
        'Segment':    Y_hat_df['unique_id'][Y_hat_df['unique_id'].str.count('/') == 1].unique(),
        'Subsegment': Y_hat_df['unique_id'][Y_hat_df['unique_id'].str.count('/') == 2].unique(),
    }

    reconciled = hrec.reconcile(
        Y_hat_df = Y_hat_df,
        Y_df     = Y_df,
        S_df     = S_df,
        tags     = tags
    )
    return reconciled

def evaluate_mint(reconciled_df, train_full, target, period_col,
                  bu_col, seg_col, subseg_col, val_periods):
    """
    Computes RMSE, MAE, wMAPE and R² for reconciled forecasts
    at each hierarchy level separately.

    Parameters
    ----------
    reconciled_df : pd.DataFrame — output of apply_mint()
    train_full    : pd.DataFrame — contains actual values for val periods
    val_periods   : list[int]

    Returns
    -------
    pd.DataFrame with one row per hierarchy level
    """
    mint_col = [c for c in reconciled_df.columns if 'MinTrace' in c][0]
    results  = []

    level_configs = {
        'BU':         (0, bu_col,  None,    None),
        'Segment':    (1, bu_col,  seg_col, None),
        'Subsegment': (2, bu_col,  seg_col, subseg_col),
    }

    for level_name, (n_slashes, c1, c2, c3) in level_configs.items():
        # Filter reconciled forecasts for this level
        mask_hat  = reconciled_df['unique_id'].str.count('/') == n_slashes
        level_hat = reconciled_df[mask_hat & reconciled_df['ds'].isin(val_periods)]

        # Build actuals at this level
        group_cols = [c for c in [c1, c2, c3] if c is not None]
        actuals    = (train_full[train_full[period_col].isin(val_periods)]
                      .groupby([period_col] + group_cols)[target]
                      .sum().reset_index())

        # Construct `unique_id` inside actuals to match `level_hat`
        if level_name == 'BU':
            actuals['unique_id'] = actuals[c1].astype(str)
        elif level_name == 'Segment':
            actuals['unique_id'] = actuals[c1].astype(str) + '/' + actuals[c2].astype(str)
        elif level_name == 'Subsegment':
            actuals['unique_id'] = actuals[c1].astype(str) + '/' + actuals[c2].astype(str) + '/' + actuals[c3].astype(str)

        # Merge them to perfectly align periods and dropped sparse series
        merged = pd.merge(actuals, level_hat, left_on=['unique_id', period_col], right_on=['unique_id', 'ds'], how='inner')

        y_true = merged[target].values
        y_pred = merged[mint_col].values

        if len(y_true) == 0:
            print(f"Warning: No overlapping series found for {level_name} level evaluation.")
            continue

        m = compute_metrics(y_true, y_pred, f'MinT ({level_name})', level_name)
        results.append(m)

    return pd.DataFrame(results)[['Model', 'Level', 'RMSE', 'MAE', 'wMAPE', 'R2']]