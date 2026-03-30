import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from scipy.stats import spearmanr

class ClientDataCleaner(BaseEstimator, TransformerMixin):
    """
    Handles initial data preparation: type conversion, column removal, etc.
    """
    def __init__(self, cols_to_remove=None):
        self.cols_to_remove = cols_to_remove

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if self.cols_to_remove:
            existing = [col for col in self.cols_to_remove if col in df.columns]
            df = df.drop(columns=existing)
        return df


class ClientOutlierHandler(BaseEstimator, TransformerMixin):
    """
    Time-series sequences usually strictly require intact periods, so we don't drop or replace outliers with NaNs 
    in this pipeline step (outliers were detected in Data_Preparation but not altered)
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).copy()


class ClientImputer(BaseEstimator, TransformerMixin):
    """
    Time series targets shouldn't be implicitly mean/median imputed across the training set. 
    Macro data is imputed via business rules earlier. This just passes the data through.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).copy()


class ClientOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Native categorical casting for LightGBM. Instead of One-Hot Encoding, exploding the feature space, 
    we cast to category type, which LightGBM handles natively and optimally.
    """
    def __init__(self, cols=None):
        if cols is None:
            cols = ["TGL Business Unit", "TGL Business Segment", "TGL Business Subsegment"]
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for c in self.cols:
            if c in df.columns:
                df[c] = df[c].astype('category')
        return df


class ClientFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Leakage-aware time-series feature engineer for panel / hierarchical forecasting.

    Main design choices
    -------------------
    1. fit() stores the last `max_window_` rows per bottom-level series.
    2. transform() prepends that history to the new fold so lag/rolling features for the
       first validation/test periods are computed safely.
    3. Row order is always restored before returning, so it is safe inside sklearn Pipeline.
    4. Raw contemporaneous target columns can be dropped to avoid target leakage.

    Added feature families
    ----------------------
    - Calendar / cyclical seasonality
    - Lags and rolling statistics
    - Momentum and rolling trend slopes
    - Intermittency / sparsity features
    - Hierarchical parent-level lag/share features
    - Cross-series features (e.g. ASP from revenue / orders)
    - Macro lags / rolling means / YoY / anomaly signals
    """

    def __init__(
        self,
        targets=None,
        subsegment_col="TGL Business Subsegment",
        period_col="Anon Period",
        parent_levels=None,
        lags=None,
        rolling_windows=None,
        slope_windows=None,
        intermittency_windows=None,
        key_macro_cols=None,
        macro_lags=None,
        orders_col="Orders cons. (anon)",
        revenue_col="Revenue cons. (anon)",
        add_calendar_features=False,
        add_target_features=False,
        add_hierarchy_features=False,
        add_cross_features=False,
        add_macro_features=False,
        drop_raw_targets=True,
        max_window=None,
        eps=1e-8,
    ):
        self.targets = targets if targets is not None else [
            "Orders cons. (anon)",
            "Revenue cons. (anon)",
        ]
        self.subsegment_col = subsegment_col
        self.period_col = period_col
        self.parent_levels = parent_levels
        self.lags = lags if lags is not None else [1, 2, 3, 6, 12, 24]
        self.rolling_windows = rolling_windows if rolling_windows is not None else [3, 6, 12]
        self.slope_windows = slope_windows if slope_windows is not None else [3, 6, 12]
        self.intermittency_windows = intermittency_windows if intermittency_windows is not None else [6, 12]
        self.key_macro_cols = key_macro_cols if key_macro_cols is not None else []
        self.macro_lags = macro_lags if macro_lags is not None else [1, 3, 6, 12]
        self.orders_col = orders_col
        self.revenue_col = revenue_col
        self.add_calendar_features = add_calendar_features
        self.add_target_features = add_target_features
        self.add_hierarchy_features = add_hierarchy_features
        self.add_cross_features = add_cross_features
        self.add_macro_features = add_macro_features
        self.drop_raw_targets = drop_raw_targets
        self.max_window = max_window
        self.eps = eps

    def _normalize_parent_levels(self):
        """
        Returns parent levels as a list of lists.
        Examples:
            None -> []
            ['Country', 'Cluster'] -> [['Country'], ['Cluster']]
            [['Country', 'Region'], 'Business Line'] -> [['Country', 'Region'], ['Business Line']]
        """
        if self.parent_levels is None:
            return []

        out = []
        for lvl in self.parent_levels:
            if isinstance(lvl, (list, tuple)):
                out.append(list(lvl))
            else:
                out.append([lvl])
        return out

    def _required_history(self):
        """
        Compute the minimum history length needed for all configured features.
        """
        candidates = [1]

        if self.lags:
            candidates.append(max(self.lags))

        if self.rolling_windows:
            candidates.append(max(self.rolling_windows))

        if self.slope_windows:
            candidates.append(max(self.slope_windows))

        if self.intermittency_windows:
            candidates.append(max(self.intermittency_windows))

        if self.macro_lags:
            candidates.append(max(self.macro_lags))

        # YoY based features compare lag_1 with lag_13
        candidates.append(13)

        return max(candidates)

    @staticmethod
    def _safe_divide(num, den, eps=1e-8):
        den_adj = np.where(np.abs(den) < eps, np.nan, den)
        return num / den_adj

    @staticmethod
    def _rolling_slope(arr):
        """
        Slope of x against t = 0,1,...,n-1.
        Used inside rolling(...).apply(..., raw=True)
        """
        arr = np.asarray(arr, dtype=float)
        mask = np.isfinite(arr)
        arr = arr[mask]

        n = len(arr)
        if n < 2:
            return np.nan

        t = np.arange(n, dtype=float)
        t_mean = t.mean()
        x_mean = arr.mean()

        denom = np.sum((t - t_mean) ** 2)
        if denom == 0:
            return np.nan

        slope = np.sum((t - t_mean) * (arr - x_mean)) / denom
        return slope

    @staticmethod
    def _months_since_last_nonzero_from_lag(series):
        """
        Input series should already be leakage-safe, e.g. Lag_1.
        Returns 0 if last observed value is nonzero, 1 if one step since last nonzero, etc.
        """
        vals = series.to_numpy(dtype=float)
        out = np.full(len(vals), np.nan, dtype=float)

        last_nonzero_idx = None
        for i, v in enumerate(vals):
            if np.isfinite(v) and v != 0:
                last_nonzero_idx = i

            if last_nonzero_idx is None:
                out[i] = np.nan
            else:
                out[i] = i - last_nonzero_idx

        return pd.Series(out, index=series.index)

    def _add_calendar(self, df):
        if not self.add_calendar_features:
            return df

        period = df[self.period_col]
        month = ((period - 1) % 12) + 1
        quarter = ((month - 1) // 3) + 1

        df["Month"] = month
        df["Quarter"] = quarter
        df["Period_Index"] = period

        # cyclical seasonality
        df["Month_sin"] = np.sin(2 * np.pi * month / 12.0)
        df["Month_cos"] = np.cos(2 * np.pi * month / 12.0)
        df["Quarter_sin"] = np.sin(2 * np.pi * quarter / 4.0)
        df["Quarter_cos"] = np.cos(2 * np.pi * quarter / 4.0)

        return df

    def _add_target_lag_roll_features(self, df, target):
        """
        All target-derived features are based on past information only.
        """
        grp = df.groupby(self.subsegment_col, sort=False)[target]

        # Plain lags
        for lag in self.lags:
            df[f"{target}_Lag_{lag}"] = grp.shift(lag)

        # Ensure lag_1 exists
        lag1_col = f"{target}_Lag_1"
        if lag1_col not in df.columns:
            df[lag1_col] = grp.shift(1)

        # Rolling stats based on lag_1 to avoid current-period leakage
        lag1_grp = df.groupby(self.subsegment_col, sort=False)[lag1_col]

        for w in self.rolling_windows:
            df[f"{target}_Rolling_Mean_{w}"] = lag1_grp.transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
            df[f"{target}_Rolling_Std_{w}"] = lag1_grp.transform(
                lambda s: s.rolling(w, min_periods=2).std()
            )
            df[f"{target}_Rolling_Min_{w}"] = lag1_grp.transform(
                lambda s: s.rolling(w, min_periods=1).min()
            )
            df[f"{target}_Rolling_Max_{w}"] = lag1_grp.transform(
                lambda s: s.rolling(w, min_periods=1).max()
            )

        # Slope / local trend on lag_1
        for w in self.slope_windows:
            df[f"{target}_Trend_Slope_{w}"] = lag1_grp.transform(
                lambda s: s.rolling(w, min_periods=2).apply(self._rolling_slope, raw=True)
            )

        # Momentum
        if (
            f"{target}_Rolling_Mean_3" in df.columns
            and f"{target}_Rolling_Mean_12" in df.columns
        ):
            df[f"{target}_Momentum_3_12"] = (
                df[f"{target}_Rolling_Mean_3"] - df[f"{target}_Rolling_Mean_12"]
            )

        if (
            f"{target}_Rolling_Mean_6" in df.columns
            and f"{target}_Rolling_Mean_12" in df.columns
        ):
            df[f"{target}_Momentum_6_12"] = (
                df[f"{target}_Rolling_Mean_6"] - df[f"{target}_Rolling_Mean_12"]
            )

        # YoY features
        if f"{target}_Lag_12" in df.columns:
            df[f"{target}_YoY_Diff"] = df[lag1_col] - df[f"{target}_Lag_12"]

        if f"{target}_Lag_12" in df.columns:
            df[f"{target}_YoY_Ratio"] = self._safe_divide(
                df[lag1_col], df[f"{target}_Lag_12"], eps=self.eps
            )

        # Volatility / coefficient of variation
        if f"{target}_Rolling_Mean_12" in df.columns and f"{target}_Rolling_Std_12" in df.columns:
            df[f"{target}_CV_12"] = self._safe_divide(
                df[f"{target}_Rolling_Std_12"],
                df[f"{target}_Rolling_Mean_12"],
                eps=self.eps,
            )

        # Intermittency / sparsity based on lag_1
        df[f"{target}_Months_Since_Last_Nonzero"] = (
            df.groupby(self.subsegment_col, sort=False)[lag1_col]
              .transform(self._months_since_last_nonzero_from_lag)
        )

        for w in self.intermittency_windows:
            df[f"{target}_Nonzero_Count_{w}"] = lag1_grp.transform(
                lambda s: s.rolling(w, min_periods=1).apply(
                    lambda x: np.sum(np.isfinite(x) & (x != 0)), raw=True
                )
            )
            df[f"{target}_Zero_Share_{w}"] = lag1_grp.transform(
                lambda s: s.rolling(w, min_periods=1).apply(
                    lambda x: np.mean(np.isfinite(x) & (x == 0)), raw=True
                )
            )

        return df

    def _add_hierarchy_features_for_target(self, df, target):
        """
        Parent-level features using aggregate-then-merge approach.

        Correctly computes hierarchy lags by:
        1. Aggregating to unique parent-period series
        2. Computing lags/rolling on that deduplicated series
        3. Merging features back to child rows

        This avoids the bug where shift() operates row-by-row instead of period-by-period when multiple children share a parent.
        """
        if not self.add_hierarchy_features:
            return df

        parent_levels = self._normalize_parent_levels()
        if not parent_levels:
            return df

        for level_cols in parent_levels:
            required = level_cols + [self.period_col, self.subsegment_col]
            if not all(c in df.columns for c in required):
                continue

            level_name = "__".join(level_cols)
            parent_keys = level_cols + [self.period_col]
            parent_current_col = f"{target}_Parent_Current_{level_name}"

            # 1) Build unique parent-period series by aggregating children
            parent_ts = (
                df.groupby(parent_keys, dropna=False, as_index=False)[target]
                  .sum()
                  .sort_values(parent_keys)
                  .reset_index(drop=True)
            )
            parent_ts = parent_ts.rename(columns={target: parent_current_col})

            # 2) Compute time-series features on deduplicated parent series
            grp = parent_ts.groupby(level_cols, dropna=False, sort=False)[parent_current_col]

            parent_ts[f"{target}_Parent_Lag_1_{level_name}"] = grp.shift(1)
            parent_ts[f"{target}_Parent_Lag_12_{level_name}"] = grp.shift(12)

            for w in self.rolling_windows:
                parent_ts[f"{target}_Parent_Rolling_Mean_{w}_{level_name}"] = (
                    grp.transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
                )
                parent_ts[f"{target}_Parent_Rolling_Std_{w}_{level_name}"] = (
                    grp.transform(lambda s: s.shift(1).rolling(w, min_periods=2).std())
                )

            parent_ts[f"{target}_Parent_YoY_Diff_{level_name}"] = (
                parent_ts[f"{target}_Parent_Lag_1_{level_name}"]
                - parent_ts[f"{target}_Parent_Lag_12_{level_name}"]
            )
            parent_ts[f"{target}_Parent_YoY_Ratio_{level_name}"] = self._safe_divide(
                parent_ts[f"{target}_Parent_Lag_1_{level_name}"],
                parent_ts[f"{target}_Parent_Lag_12_{level_name}"],
                eps=self.eps,
            )

            # 3) Merge parent features back to child rows
            feature_cols = [
                parent_current_col,
                f"{target}_Parent_Lag_1_{level_name}",
                f"{target}_Parent_Lag_12_{level_name}",
                f"{target}_Parent_YoY_Diff_{level_name}",
                f"{target}_Parent_YoY_Ratio_{level_name}",
            ] + [
                f"{target}_Parent_Rolling_Mean_{w}_{level_name}"
                for w in self.rolling_windows
            ] + [
                f"{target}_Parent_Rolling_Std_{w}_{level_name}"
                for w in self.rolling_windows
            ]

            df = df.merge(
                parent_ts[parent_keys + feature_cols],
                on=parent_keys,
                how="left",
                validate="many_to_one",
            )

            # 4) Child share of parent using lagged values only
            parent_lag1_col = f"{target}_Parent_Lag_1_{level_name}"
            parent_lag12_col = f"{target}_Parent_Lag_12_{level_name}"

            target_lag1 = df.get(f"{target}_Lag_1")
            if target_lag1 is not None:
                df[f"{target}_Share_of_Parent_Lag_1_{level_name}"] = self._safe_divide(
                    target_lag1, df[parent_lag1_col], eps=self.eps
                )

            if f"{target}_Lag_12" in df.columns:
                df[f"{target}_Share_of_Parent_Lag_12_{level_name}"] = self._safe_divide(
                    df[f"{target}_Lag_12"], df[parent_lag12_col], eps=self.eps
                )

            if (
                f"{target}_Share_of_Parent_Lag_1_{level_name}" in df.columns
                and f"{target}_Share_of_Parent_Lag_12_{level_name}" in df.columns
            ):
                df[f"{target}_Share_of_Parent_YoY_Diff_{level_name}"] = (
                    df[f"{target}_Share_of_Parent_Lag_1_{level_name}"]
                    - df[f"{target}_Share_of_Parent_Lag_12_{level_name}"]
                )

            # Remove leakage-prone current-period parent total
            df.drop(columns=[parent_current_col], inplace=True)

        return df


    def _add_cross_features(self, df):
        """
        Example cross-series features using orders and revenue.
        """
        if not self.add_cross_features:
            return df

        req_cols = [
            f"{self.orders_col}_Lag_1",
            f"{self.revenue_col}_Lag_1",
        ]
        if not all(c in df.columns for c in req_cols):
            return df

        # Average selling price from lagged signals
        df["ASP_Lag_1"] = self._safe_divide(
            df[f"{self.revenue_col}_Lag_1"],
            df[f"{self.orders_col}_Lag_1"],
            eps=self.eps,
        )

        if (
            f"{self.revenue_col}_Lag_12" in df.columns
            and f"{self.orders_col}_Lag_12" in df.columns
        ):
            df["ASP_Lag_12"] = self._safe_divide(
                df[f"{self.revenue_col}_Lag_12"],
                df[f"{self.orders_col}_Lag_12"],
                eps=self.eps,
            )
            df["ASP_YoY_Diff"] = df["ASP_Lag_1"] - df["ASP_Lag_12"]

        if (
            f"{self.revenue_col}_Rolling_Mean_3" in df.columns
            and f"{self.orders_col}_Rolling_Mean_3" in df.columns
        ):
            df["ASP_Rolling_Mean_3"] = self._safe_divide(
                df[f"{self.revenue_col}_Rolling_Mean_3"],
                df[f"{self.orders_col}_Rolling_Mean_3"],
                eps=self.eps,
            )

        if (
            f"{self.revenue_col}_Rolling_Mean_12" in df.columns
            and f"{self.orders_col}_Rolling_Mean_12" in df.columns
        ):
            df["ASP_Rolling_Mean_12"] = self._safe_divide(
                df[f"{self.revenue_col}_Rolling_Mean_12"],
                df[f"{self.orders_col}_Rolling_Mean_12"],
                eps=self.eps,
            )

        if (
            f"{self.revenue_col}_Momentum_3_12" in df.columns
            and f"{self.orders_col}_Momentum_3_12" in df.columns
        ):
            df["Revenue_vs_Orders_Momentum_3_12"] = (
                df[f"{self.revenue_col}_Momentum_3_12"]
                - df[f"{self.orders_col}_Momentum_3_12"]
            )

        return df

    def _add_macro_features(self, df):
        """
        Macro features are generated at period level and merged back.
        To stay conservative and operationally safe, only lagged / rolling / YoY macro signals
        are kept. Current-period macro values are not used as features here.
        """
        if not self.add_macro_features:
            return df

        available_macros = [c for c in self.key_macro_cols if c in df.columns]
        if not available_macros:
            return df

        period_macro = (
            df[[self.period_col] + available_macros]
            .groupby(self.period_col, as_index=False)
            .first()
            .sort_values(self.period_col)
            .copy()
        )

        for col in available_macros:
            for lag in self.macro_lags:
                period_macro[f"{col}_Lag_{lag}"] = period_macro[col].shift(lag)

            lag1_col = f"{col}_Lag_1"
            lag1_series = period_macro[lag1_col]

            for w in [3, 12]:
                period_macro[f"{col}_Rolling_Mean_{w}"] = lag1_series.rolling(w, min_periods=1).mean()
                period_macro[f"{col}_Rolling_Std_{w}"] = lag1_series.rolling(w, min_periods=2).std()

            if f"{col}_Lag_12" in period_macro.columns:
                period_macro[f"{col}_YoY_Diff"] = (
                    period_macro[lag1_col] - period_macro[f"{col}_Lag_12"]
                )
                period_macro[f"{col}_YoY_Ratio"] = self._safe_divide(
                    period_macro[lag1_col],
                    period_macro[f"{col}_Lag_12"],
                    eps=self.eps,
                )

            if (
                f"{col}_Rolling_Mean_3" in period_macro.columns
                and f"{col}_Rolling_Mean_12" in period_macro.columns
            ):
                period_macro[f"{col}_Momentum_3_12"] = (
                    period_macro[f"{col}_Rolling_Mean_3"]
                    - period_macro[f"{col}_Rolling_Mean_12"]
                )

            if (
                f"{col}_Rolling_Mean_12" in period_macro.columns
                and f"{col}_Rolling_Std_12" in period_macro.columns
            ):
                period_macro[f"{col}_Anomaly_12"] = self._safe_divide(
                    period_macro[lag1_col] - period_macro[f"{col}_Rolling_Mean_12"],
                    period_macro[f"{col}_Rolling_Std_12"],
                    eps=self.eps,
                )

        macro_feature_cols = [
            c for c in period_macro.columns
            if c != self.period_col and c not in available_macros
        ]

        df = df.merge(
            period_macro[[self.period_col] + macro_feature_cols],
            on=self.period_col,
            how="left",
            validate="many_to_one",
        )

        return df

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()

        required = {self.subsegment_col, self.period_col}
        if not required.issubset(df.columns):
            self.history_ = pd.DataFrame()
            self.max_window_ = self.max_window if self.max_window is not None else self._required_history()
            return self

        self.max_window_ = self.max_window if self.max_window is not None else self._required_history()

        df = df.sort_values([self.subsegment_col, self.period_col]).copy()
        self.history_ = (
            df.groupby(self.subsegment_col, group_keys=False)
              .tail(self.max_window_)
              .copy()
        )
        return self

    def transform(self, X):
        check_is_fitted(self, ["history_", "max_window_"])

        df_new = pd.DataFrame(X).copy()
        original_index = df_new.index

        # Keep original row order for sklearn safety
        df_new["_row_id"] = np.arange(len(df_new))
        df_new["_is_history"] = False

        if self.history_ is not None and not self.history_.empty:
            hist = self.history_.copy()
            hist["_row_id"] = -1
            hist["_is_history"] = True
            df = pd.concat([hist, df_new], ignore_index=True)
        else:
            df = df_new.copy()

        required = {self.subsegment_col, self.period_col}
        if required.issubset(df.columns):
            df = df.sort_values(
                [self.subsegment_col, self.period_col, "_is_history", "_row_id"],
                kind="mergesort",
            ).copy()

            df = self._add_calendar(df)

            if self.add_target_features:
                for target in self.targets:
                    if target in df.columns:
                        df = self._add_target_lag_roll_features(df, target)
                        df = self._add_hierarchy_features_for_target(df, target)

            df = self._add_cross_features(df)
            df = self._add_macro_features(df)

        # Keep only rows belonging to the actual input X
        df = df.loc[~df["_is_history"]].copy()

        # Drop raw contemporaneous target columns to avoid leakage
        if self.drop_raw_targets:
            cols_to_drop = [c for c in self.targets if c in df.columns]
            if cols_to_drop:
                df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

        # Restore exact original row order
        df = df.sort_values("_row_id", kind="mergesort").copy()
        df.drop(columns=["_row_id", "_is_history"], inplace=True)
        df.index = original_index

        return df

class ClientFeatureSelection(BaseEstimator, TransformerMixin):
    """
    Kept as a generic skeleton. Emptied out the heavy voting logic previously
    used for Churn classification since forecasting models (LightGBM) will do 
    their own strict native split/feature selection internally.
    """
    def __init__(self, features_to_keep=None):
        self.features_to_keep = features_to_keep

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if self.features_to_keep:
            available = [c for c in self.features_to_keep if c in df.columns]
            return df[available]
        return df

# FEATURE SELECTION

class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, horizon=1, lag_top_k=2, corr_threshold=0.9, min_votes=1, rf_cum_threshold=0.90):
        self.horizon = horizon
        self.lag_top_k = lag_top_k
        self.corr_threshold = corr_threshold
        self.min_votes = min_votes
        self.rf_cum_threshold = rf_cum_threshold
        self.regex = re.compile(r"_(?:Lag|Rolling|Trend|Momentum)_(\d+)")

    def fit(self, X, y):
        self.input_features_ = X.columns.tolist()
        self.removal_log_ = {} 

        # phase 1 Temporal Leak Check
        cols_after_p1 = [c for c in X.columns if self._is_safe(c)]
        self.removal_log_['Phase 1: Leakage'] = list(set(self.input_features_) - set(cols_after_p1))
        
        # phase 2 & 3 Intra-Family + Correlation Pruning
        selected_phase3 = self._prune_features(X[cols_after_p1], y)
        self.removal_log_['Phase 2 & 3: Correlation'] = list(set(cols_after_p1) - set(selected_phase3))
        
        # phase 4 Global Voting
        X_reduced = X[selected_phase3].fillna(X[selected_phase3].median())
        
        # Lasso + RF
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_reduced)
        # @elias: y_train (Revenue in EUR, values in millions) must be scaled before
        # Lasso — otherwise the huge y scale inflates the optimal alpha to ~4M, which
        # zeros out almost all features. RobustScaler on y normalises the penalty space
        # so LassoCV selects a reasonable alpha. RF is scale-invariant so no change there.
        y_scaler = RobustScaler()
        y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
        
        # @elias: Changed cv=5 (standard K-Fold) to TimeSeriesSplit — standard K-Fold
        # shuffles data randomly, which leaks future into past for time series.
        # TimeSeriesSplit always trains on past, validates on future (expanding window).
        lasso = LassoCV(cv=TimeSeriesSplit(n_splits=5)).fit(X_scaled, y_scaled)
        lasso_support = X_reduced.columns[lasso.coef_ != 0]
        
        rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42).fit(X_reduced, y)
        # @bruna: Changed from fixed top-N (rf_top_n=50) to cumulative importance
        # threshold. A hard cap of 50 is arbitrary and fragile: if RF concentrates
        # importance in 15 features, top-50 passes 35 near-zero-importance features
        # that add noise; if the signal is spread across 80, top-50 drops useful ones.
        # Instead, we keep the fewest features whose cumulative importance reaches
        # rf_cum_threshold (default 90%) of the total. This adapts to the actual
        # importance profile of each fold. A floor of 10 prevents over-pruning when
        # importance is extremely concentrated, and a ceiling of N_features prevents
        # selecting more features than available.
        importances_sorted = np.sort(rf.feature_importances_)[::-1]
        cumulative = np.cumsum(importances_sorted)
        
        n_for_threshold = np.searchsorted(cumulative, self.rf_cum_threshold * cumulative[-1]) + 1
        n_rf = np.clip(n_for_threshold, 10, len(X_reduced.columns))
        rf_top_indices = np.argsort(rf.feature_importances_)[-n_rf:]
        rf_support = X_reduced.columns[rf_top_indices]
        
        votes = pd.Series(list(lasso_support) + list(rf_support)).value_counts()
        voted_features = votes[votes >= self.min_votes].index.tolist()
        
        self.removal_log_['Phase 4: Voting'] = list(set(selected_phase3) - set(voted_features))
        
        # phase 5 Structural Inclusion
        structural = [c for c in cols_after_p1 if any(kw in c for kw in ["Month", "Quarter", "Parent", "Share"])]
        
        # @elias: list(set(...)) destroys column order — same data, different Python
        # sessions can produce different orderings, which breaks reproducibility and
        # can silently reorder features entering a model. Fixed by filtering cols_after_p1
        # (which has a stable order) instead of converting a set to a list.
        combined = set(voted_features) | set(structural)
        self.selected_features_ = [c for c in cols_after_p1 if c in combined]
        
        saved_by_structure = set(structural) - set(voted_features)
        if saved_by_structure:
            self.removal_log_['Structural Rescue'] = list(saved_by_structure)

        return self

    # details about FS
    def get_selection_report(self):
        # @elias: get_selection_report() used "variables removed" for every phase,
        # including Structural Rescue — which adds features back, not removes them.
        # The output previously showed "Structural Rescue: 63 variables removed"
        # which is the opposite of what happened. Fixed with phase-specific labels.
        PHASE_LABELS = {
            'Phase 1: Leakage': 'removed',
            'Phase 2 & 3: Correlation': 'removed',
            'Phase 4: Voting': 'removed',
            'Structural Rescue': 'rescued (added back)',
        }

        print("="*60)
        print("Feature Selection Report")
        print("="*60)
        for phase, features in self.removal_log_.items():
            count = len(features)
            label = PHASE_LABELS.get(phase, 'removed')
            print(f"\n> {phase}: {count} variables {label}")
            if count > 0:
                preview = ", ".join(sorted(features)[:10])
                print(f"  Examples: {preview}..." if count > 10 else f"  List: {preview}")

        print("\n" + "="*60)
        print(f"FINAL RESULT: {len(self.selected_features_)} variables selected.")
        print("="*60)

    def transform(self, X):
        return X[self.selected_features_]

    def _is_safe(self, col):
        if any(kw in col for kw in ["Month", "Quarter", "TGL", "Period"]): return True
        match = self.regex.search(col)
        lag = int(match.group(1)) if match else 12
        if "_Lag_" not in col and any(kw in col for kw in ["Rolling", "Trend", "Momentum"]):
            lag = 1
        return lag >= self.horizon

    def _prune_features(self, X, y):
        from collections import defaultdict

        lag_cols = [c for c in X.columns if "_Lag_" in c]

        # @elias: Added NaN-safe Spearman. Macro lag features have no NaN (macro data
        # goes back to 2010, well before training period 1 = Jan 2022). However,
        # Revenue and Orders lag features have structural NaN in early periods because
        # the revenue history only starts at period 1 — e.g. Revenue_Lag_12 is NaN
        # for periods 1–12, Orders_Lag_6 is NaN for periods 1–6.
        # Passing those to spearmanr() directly produces NaN correlations that
        # silently distort the top-K ranking. Fixed with non-NaN row alignment.
        def safe_spearman(col):
            mask = X[col].notna() & pd.Series(y, index=X.index).notna()
            if mask.sum() < 10:
                return 0.0
            return abs(spearmanr(X[col][mask], pd.Series(y, index=X.index)[mask])[0])

        corrs = {c: safe_spearman(c) for c in lag_cols}

        # @elias: Changed from global top-K to per-family top-K.
        # Previous code took the best lag_top_k lags across ALL lag columns globally,
        # which could select e.g. Revenue_Lag_1/2/3 and discard every Orders and Macro
        # lag entirely. Now we group by variable name (prefix before "_Lag_") and keep
        # the best lag_top_k within each group, preserving diversity across families.
        families = defaultdict(list)
        for c in lag_cols:
            family = c.split("_Lag_")[0]
            families[family].append(c)

        # @elias: Pairwise Pearson pruning is now done WITHIN each lag family separately,
        # then separately for non-lag features. Previously all families were merged into
        # a single correlation matrix, which caused cross-family drops: Orders_Lag_3 and
        # Revenue_Lag_3 had Pearson=0.953 > 0.95, so Revenue_Lag_3 was eliminated simply
        # because Orders columns appear first in X.columns.
        # The fix: lags from different variable families are never pruned against each other.
        top_lags = []
        for family, cols in families.items():
            sorted_cols = sorted(cols, key=lambda c: corrs.get(c, 0), reverse=True)
            candidates = sorted_cols[:self.lag_top_k]
            # Intra-family Pearson pruning only
            if len(candidates) > 1:
                X_fam = X[candidates].fillna(X[candidates].median())
                fam_corr = X_fam.corr().abs()
                upper_fam = fam_corr.where(np.triu(np.ones(fam_corr.shape), k=1).astype(bool))
                fam_drop = {c for c in upper_fam.columns if any(upper_fam[c] > self.corr_threshold)}
                candidates = [c for c in candidates if c not in fam_drop]
            top_lags.extend(candidates)

        # Pairwise Pearson pruning on non-lag features independently (not mixed with lags)
        other_cols = [c for c in X.columns if "_Lag_" not in c]
        X_other = X[other_cols].fillna(X[other_cols].median())
        other_corr = X_other.corr().abs()
        upper_other = other_corr.where(np.triu(np.ones(other_corr.shape), k=1).astype(bool))
        other_drop = {c for c in upper_other.columns if any(upper_other[c] > self.corr_threshold)}
        selected_other = [c for c in other_cols if c not in other_drop]

        return top_lags + selected_other