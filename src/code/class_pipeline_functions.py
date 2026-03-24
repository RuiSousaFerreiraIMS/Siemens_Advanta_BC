import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

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
        Parent-level features:
        - parent lag 1 / lag 12
        - parent rolling mean 3 / 12
        - subsegment share of parent using lagged values
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

            # Current-period parent total is computed internally only;
            # it must not be kept as a feature to avoid leakage.
            parent_curr = df.groupby(level_cols + [self.period_col], sort=False)[target].transform("sum")
            parent_curr_col = f"{target}_ParentCurrent_{level_name}"
            df[parent_curr_col] = parent_curr

            parent_grp = df.groupby(level_cols, sort=False)[parent_curr_col]

            df[f"{target}_Parent_Lag_1_{level_name}"] = parent_grp.shift(1)
            df[f"{target}_Parent_Lag_12_{level_name}"] = parent_grp.shift(12)

            parent_lag1_col = f"{target}_Parent_Lag_1_{level_name}"
            parent_lag12_col = f"{target}_Parent_Lag_12_{level_name}"

            parent_lag1_grp = df.groupby(level_cols, sort=False)[parent_lag1_col]

            df[f"{target}_Parent_Rolling_Mean_3_{level_name}"] = parent_lag1_grp.transform(
                lambda s: s.rolling(3, min_periods=1).mean()
            )
            df[f"{target}_Parent_Rolling_Mean_12_{level_name}"] = parent_lag1_grp.transform(
                lambda s: s.rolling(12, min_periods=1).mean()
            )

            # Share of parent using lagged values only
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
            df.drop(columns=[parent_curr_col], inplace=True)

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