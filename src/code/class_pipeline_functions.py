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
    Generates time-series feature engineering variables (lags, rolling stats, calendar features, macro transformations).

    To prevent data leakage when used in a scikit-learn pipeline:
    - fit() memorizes the last 'max_window' periods from the training set.
    - transform() prepends this history to compute lags and rolling stats without data loss for the first periods of the 
    validation/test set.
    """
    def __init__(self, targets=None, lags=None, rolling_windows=None, max_window=12, key_macro_cols=None):
        self.targets = targets if targets else ['Orders cons. (anon)', 'Revenue cons. (anon)']
        self.lags = lags if lags else [1, 2, 3, 6, 12]
        self.rolling_windows = rolling_windows if rolling_windows else [3, 6, 12]
        self.max_window = max_window
        self.key_macro_cols = key_macro_cols if key_macro_cols else []
        self.history_ = None

    def fit(self, X, y=None):
        """
        Learns the tail of the training data to be appended to validation sequences during transformation.
        """
        df = pd.DataFrame(X).copy()
        
        # Save the last `max_window` periods for each subsegment if available
        if 'TGL Business Subsegment' in df.columns and 'Anon Period' in df.columns:
            df.sort_values(by=['TGL Business Subsegment', 'Anon Period'], inplace=True)
            self.history_ = df.groupby('TGL Business Subsegment').tail(self.max_window).copy()
        else:
            self.history_ = pd.DataFrame()
            
        return self

    def transform(self, X):
        """
        Applies time series logic. 
        Uses history from train (if available) to compute lags for the first evaluation periods without data loss.
        """
        check_is_fitted(self, "history_")
        df_new = pd.DataFrame(X).copy()
        
        # Prepend training history if needed securely
        if self.history_ is not None and not self.history_.empty:
            self.history_['_is_history'] = True
            df_new['_is_history'] = False
            df = pd.concat([self.history_, df_new], ignore_index=True)
            has_history = True
        else:
            df = df_new
            has_history = False

        # Ensure correct sorting for calculation
        if 'TGL Business Subsegment' in df.columns and 'Anon Period' in df.columns:
            df.sort_values(by=['TGL Business Subsegment', 'Anon Period'], inplace=True)

            # 1. Calendar Features
            df['Month'] = ((df['Anon Period'] - 1) % 12) + 1
            df['Quarter'] = ((df['Month'] - 1) // 3) + 1
            df['Period_Index'] = df['Anon Period']

            # 2. Lags & Rolling
            for target in self.targets:
                if target in df.columns:
                    for lag in self.lags:
                        df[f'{target}_Lag_{lag}'] = df.groupby('TGL Business Subsegment')[target].shift(lag)

                    # Lag 1 must exist for rolling calculation to avoid current period leakage
                    if f'{target}_Lag_1' not in df.columns:
                        df[f'{target}_Lag_1'] = df.groupby('TGL Business Subsegment')[target].shift(1)

                    for window in self.rolling_windows:
                        df[f'{target}_Rolling_Mean_{window}'] = df.groupby('TGL Business Subsegment')[f'{target}_Lag_1'].transform(
                            lambda x: x.rolling(window, min_periods=1).mean()
                        )
                        if window >= 3:
                            df[f'{target}_Rolling_Std_{window}'] = df.groupby('TGL Business Subsegment')[f'{target}_Lag_1'].transform(
                                lambda x: x.rolling(window, min_periods=2).std()
                            )

            # 3. Macro Lags (if key_macro_cols available)
            available_macros = [c for c in self.key_macro_cols if c in df.columns]
            if available_macros:
                for col in available_macros:
                    # Sort unique periods to compute sequence features properly without duplication
                    period_df = df[['Anon Period', col]].drop_duplicates().sort_values('Anon Period')
                    period_df[f'{col}_Lag_1'] = period_df[col].shift(1)
                    period_df[f'{col}_Lag_3'] = period_df[col].shift(3)
                    period_df[f'{col}_MoM_Diff'] = period_df[col] - period_df[f'{col}_Lag_1']
                    
                    # Merge back onto the main df
                    df = df.merge(period_df.drop(columns=[col]), on='Anon Period', how='left')

        # Filter back strictly to the originally passed rows (usually validation set length)
        if has_history:
            df = df[df['_is_history'] == False].drop(columns=['_is_history'])
            df.index = df_new.index  # Restore to exactly mirror the input X_test

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