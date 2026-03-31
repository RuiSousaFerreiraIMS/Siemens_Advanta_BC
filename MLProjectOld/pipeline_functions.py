


################################################################################
# Feature Selection
################################################################################

# Explanation of Leakage Prevention:
#       0) Function call: pass the preprocessor_pipe (which contains the Majority Voter) into RandomizedSearchCV:
#       1) Splitting: The search CV splits the data into Train and Validation folds.
#       2) Fitting: It calls .fit() on your pipeline using only the Train fold.
#       3) Voting: The custom MajorityVoteSelectorTransformer runs inside the pipeline. It sees only the Train fold. It calculates votes and selects features based only on that fold.
#       4) Transformation: It transforms the Validation fold based on the features selected in step 3.
#       ==> Leakage Free: Since the Validation fold was never used to decide which features to keep, there is no leakage.

# Explanation of why it is not a problem for the final refit that different features might have been selected in different folds:
#       0) Final refit is called on best hyperparameters found during CV.
#       1) The MajorityVoteSelectorTransformer sees the entire training data during final refit.
#       2) It calculates votes and selects features based on the entire training data. (This is done without hp-tuning now because the hps are fixed.)
#       3) It transforms the entire training data based on the features selected in step 2.
#       ==> No Problem: Although different folds might have selected different features during CV, the final refit uses the entire training data to select only one final set of features (which might vary from previous features selected in the folds but thats not a problem).


class MajorityVoteSelectorTransformer(BaseEstimator, TransformerMixin):
    """
    Runs multiple feature selectors and keeps a feature if at least `min_votes` selectors agree.
    """

    def __init__(self, selectors=None, min_votes=2, verbose=False):
        """
        args:
            selectors: list of sklearn feature selector objects.
            min_votes: int, minimum number of selectors that must agree to keep a feature.
        """
        self.selectors = selectors
        self.min_votes = min_votes
        self.verbose = verbose
        self.fitted_selectors_ = []
        self.support_mask_ = None
        self.feature_names_in_ = None
        self.votes_ = None

    def fit(self, X, y=None):
        # Validate inputs
        if not self.selectors:
            raise ValueError("You must provide a list of selectors.")

        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)  # store input feature names if available for the get_feature_names_out method

        self.fitted_selectors_ = []
        votes = np.zeros(X.shape[1])

        # Loop through each selector, fit it, and tally votes
        for selector in self.selectors:
            # Clone to ensure we don't modify the original objects
            sel = clone(selector)
            sel.fit(X, y)
            self.fitted_selectors_.append(sel)

            # Get boolean mask of selected features (True means that they are selected)
            votes += sel.get_support().astype(int)

        # Create the final mask: True if votes >= threshold
        self.votes_ = votes.copy()
        self.support_mask_ = votes >= self.min_votes

        if self.verbose:
            _print_section("MajorityVoteSelectorTransformer report")
            n_total = X.shape[1]
            n_keep = int(self.support_mask_.sum())
            print(f"Features kept: {n_keep}/{n_total} (min_votes={self.min_votes})")

            # simple visualization: how many features got 0/1/2/.. votes
            vc = pd.Series(votes).value_counts().sort_index()
            rep = pd.DataFrame({"votes": vc.index, "n_features": vc.values})
            _maybe_display(rep, max_rows=20)

            plt.figure()
            plt.title("Vote distribution (how many selectors agreed)")
            plt.bar(rep["votes"].astype(str), rep["n_features"])
            plt.xlabel("Votes")
            plt.ylabel("Number of features")
            plt.tight_layout()
            plt.show()

        return self

    def transform(self, X):
        if self.support_mask_ is None:
            raise NotFittedError("This MajorityVoteSelectorTransformer instance is not fitted yet.")

        # If X is a DataFrame, keep column names for better debugging. Otherwise return numpy array
        if hasattr(X, "loc"):
            return X.loc[:, self.support_mask_]

        return X[:, self.support_mask_]

    def get_feature_names_out(self, input_features=None):
        """This method is called by sklearn when set_output(transform='pandas') is on (like in our debug-transformer)"""
        # If we stored names during fit, use them as default
        if input_features is None and self.feature_names_in_ is not None:
            input_features = self.feature_names_in_

        # If input_features is still None, sklearn generates x0, x1...
        if input_features is None:
            # If no names provided, generate generic indices
            return np.array([f"x{i}" for i in range(len(self.support_mask_))])[self.support_mask_]

        return np.array(input_features)[self.support_mask_]


################################################################################

class SpearmanRelevancyRedundancySelector(BaseEstimator, SelectorMixin):
    """
    Selects features based on:
      1. Relevance: High Spearman correlation with the target.
      2. Non-Redundancy: Low Spearman correlation with already selected features (drops redundant variable with lower correlation to target).

    More detailed (Algorithm: “Maximum Relevance, Minimum Redundancy (mRMR)-style pruning”):
    1. Sort features by relevance.
    2. Start with an empty list of selected features.
    3. For each feature (in order of relevance):
    4. Compare it with all already-selected features.
    5. If its |corr| with any selected feature > threshold, skip it.
    6. Otherwise, keep the feature.

    Parameters:
    ----------
    relevance_threshold : float
        Minimum absolute Spearman correlation with target to consider a feature 'relevant'.
    redundancy_threshold : float
        Maximum absolute Spearman correlation allowed between a new feature and already selected features.
    """

    def __init__(
        self,
        relevance_threshold=0.1,
        redundancy_threshold=0.85,
        verbose=False,
        verbose_top_n=15,
        verbose_plot=True,
    ):
        self.relevance_threshold = relevance_threshold
        self.redundancy_threshold = redundancy_threshold
        self.verbose = verbose
        self.verbose_top_n = verbose_top_n
        self.verbose_plot = verbose_plot

        self.selected_indices_ = None
        self.feature_names_in_ = None

    def fit(self, X, y):
        # 1. Input Validation and Feature Name Capture
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array([str(c) for c in X.columns])

        # Convert to Numpy for speed, ensure y is correct shape
        X_arr, y_arr = check_X_y(X, y, dtype=None)
        n_features = X_arr.shape[1]

        # 1) Relevance Filtering (Filter out weak features first)
        relevance_scores = []
        for i in range(n_features):
            # Calculate spearman with target
            corr, _ = spearmanr(X_arr[:, i], y_arr)
            relevance_scores.append(abs(corr))

        self.relevance_scores_ = np.array(relevance_scores)
        relevance_indices = np.where(self.relevance_scores_ > self.relevance_threshold)[0]
        self.relevance_pass_indices_ = relevance_indices

        # Sort candidates by relevance (Best feature first (descending) -> argsort gives ascending, so we take [::-1])
        sorted_candidates = relevance_indices[np.argsort(self.relevance_scores_[relevance_indices])[::-1]]

        # 2) Eliminate Redundant Features (Remove the one with lower relevance)
        selected_indices = []

        # Optimization: Create a DataFrame of just the candidates for fast matrix corr
        if len(sorted_candidates) > 0:
            X_candidates = pd.DataFrame(X_arr[:, sorted_candidates])  # pandas for the correlation matrix as it handles Spearman efficiently
            corr_matrix = X_candidates.corr(method="spearman").abs().values

            # Map: corr_matrix index [i] corresponds to sorted_candidates[i]
            kept_local_indices = []

            for i in range(len(sorted_candidates)):
                # Always keep the single most relevant feature (i=0) because there is no possible redundant feature here yet
                if i == 0:
                    kept_local_indices.append(i)
                    continue

                # Check correlation with al features that passed relevance threshold (corr_matrix[i, kept_local_indices] gives array of corrs)
                current_corrs = corr_matrix[i, kept_local_indices]

                # If the max correlation with any selected feature is too high, drop it
                if np.max(current_corrs) < self.redundancy_threshold:
                    kept_local_indices.append(i)

            # Convert local kept indices back to original feature indices
            selected_indices = sorted_candidates[kept_local_indices]

        self.selected_indices_ = np.array(selected_indices)

        # Verbose report
        if self.verbose:
            _print_section("SpearmanRelevancyRedundancySelector report")

            total = n_features
            passed = len(self.relevance_pass_indices_)
            kept = len(self.selected_indices_)

            print(f"Thresholds: relevance >= {self.relevance_threshold} | redundancy < {self.redundancy_threshold}")
            print(f"Total features: {total}")
            print(f"Passed relevance filter: {passed}")
            print(f"Kept after redundancy pruning: {kept}")

            # Build a readable table for top features by relevance
            names = (
                self.feature_names_in_
                if self.feature_names_in_ is not None
                else np.array([f"x{i}" for i in range(n_features)])
            )

            mask = self._get_support_mask()

            rep = pd.DataFrame(
                {
                    "feature": names,
                    "abs_spearman_with_target": self.relevance_scores_,
                    "selected": mask,
                }
            ).sort_values("abs_spearman_with_target", ascending=False)

            print("\nTop features by relevance (highest |Spearman| first):")
            _maybe_display(rep.head(self.verbose_top_n), max_rows=self.verbose_top_n)

            if self.verbose_plot:
                top = rep.head(self.verbose_top_n)
                if len(top) > 0:
                    plt.figure()
                    plt.title("Top feature relevance (|Spearman with target|)")
                    plt.barh(top["feature"][::-1], top["abs_spearman_with_target"][::-1])
                    plt.xlabel("|Spearman correlation|")
                    plt.tight_layout()
                    plt.show()

        return self

    def _get_support_mask(self):
        """
        Required by SelectorMixin. Returns boolean mask of selected features.
        """
        check_is_fitted(self, "selected_indices_")
        n_features = len(self.relevance_scores_)
        mask = np.zeros(n_features, dtype=bool)
        mask[self.selected_indices_] = True
        return mask

    def get_feature_names_out(self, input_features=None):
        """
        Ensures proper feature names are passed to the next step.
        """
        if input_features is not None:
            names = np.array(input_features)
        elif self.feature_names_in_ is not None:
            names = self.feature_names_in_
        else:
            names = np.array([f"x{i}" for i in range(len(self.relevance_scores_))])

        return names[self._get_support_mask()]


################################################################################

class MutualInfoThresholdSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        threshold=0.01,
        n_neighbors=10,
        random_state=42,
        verbose=False,
        verbose_top_n=15,
        verbose_plot=True,
    ):
        """
        threshold: Minimum MI score required to keep a feature.
        n_neighbors: Parameter for the internal MI calculation.
        """
        self.threshold = threshold
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_top_n = verbose_top_n
        self.verbose_plot = verbose_plot

        self.mask_ = None
        self.feature_names_in_ = None

    def _encode_if_needed(self, X):
        """
        MI needs numeric input. If X contains non-numeric columns, we one-hot encode (pandas get_dummies)
        and remember the resulting columns for consistent transform().

        This makes your quick notebook snippet work directly on X_fe (which still has strings).
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array([str(c) for c in X.columns])

            # if already all numeric, keep as-is
            if all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
                X_num = X.copy()
                self.enc_columns_ = list(X_num.columns)
                self.discrete_mask_ = np.array([False] * X_num.shape[1])
                return X_num

            # else: encode categoricals
            cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
            X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False, dummy_na=True)

            self.enc_columns_ = list(X_enc.columns)

            # discrete mask: columns created from categoricals are discrete; numeric originals treated as continuous
            discrete_mask = np.array([False] * X_enc.shape[1])
            for c in cat_cols:
                prefix = f"{c}_"
                for j, colname in enumerate(self.enc_columns_):
                    if colname.startswith(prefix):
                        discrete_mask[j] = True
            self.discrete_mask_ = discrete_mask
            return X_enc

        # numpy array input (assume already numeric)
        self.feature_names_in_ = None
        self.enc_columns_ = None
        self.discrete_mask_ = None
        return X

    def fit(self, X, y):
        # Encode if needed (so your direct notebook snippet works)
        X_enc = self._encode_if_needed(X)

        # Calculate Mutual Information Scores
        if isinstance(X_enc, pd.DataFrame):
            X_mat = X_enc.values
            feature_names = np.array(self.enc_columns_, dtype=object)
        else:
            X_mat = X_enc
            feature_names = None

        # discrete_features only if we computed it (mixed data)
        if getattr(self, "discrete_mask_", None) is not None:
            self.scores_ = mutual_info_regression(
                X_mat,
                y,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
                discrete_features=self.discrete_mask_,
            )
        else:
            self.scores_ = mutual_info_regression(
                X_mat,
                y,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
            )

        # 3. Create Mask based on Threshold
        self.mask_ = self.scores_ > self.threshold

        # store names for downstream / verbose
        self.feature_names_enc_ = feature_names

        # Verbose report
        if self.verbose:
            _print_section("MutualInfoThresholdSelector report")

            n_total = len(self.scores_)
            n_kept = int(np.sum(self.mask_))

            print(f"Threshold: MI >= {self.threshold}")
            print(f"Total features: {n_total}")
            print(f"Kept features : {n_kept}")

            names = (
                self.feature_names_enc_
                if self.feature_names_enc_ is not None
                else np.array([f"x{i}" for i in range(n_total)])
            )

            rep = pd.DataFrame(
                {
                    "feature": names,
                    "mutual_information": self.scores_,
                    "selected": self.mask_,
                }
            ).sort_values("mutual_information", ascending=False)

            print("\nTop features by mutual information:")
            _maybe_display(rep.head(self.verbose_top_n), max_rows=self.verbose_top_n)

            if self.verbose_plot:
                top = rep.head(self.verbose_top_n)
                if len(top) > 0:
                    plt.figure()
                    plt.title("Top feature importance (Mutual Information)")
                    plt.barh(top["feature"][::-1], top["mutual_information"][::-1])
                    plt.xlabel("Mutual information")
                    plt.tight_layout()
                    plt.show()

        return self

    def transform(self, X):
        if self.mask_ is None:
            raise NotFittedError("Selector not fitted.")

        # Transform must mirror fit encoding
        if isinstance(X, pd.DataFrame):
            # if fit used encoding (enc_columns_ exists), apply it; else assume numeric DF
            if getattr(self, "enc_columns_", None) is not None:
                # encode using same logic as fit (same cat columns inference)
                cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
                X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False, dummy_na=True)

                # align to training columns
                X_enc = X_enc.reindex(columns=self.enc_columns_, fill_value=0)

                return X_enc.loc[:, self.mask_]

            # numeric-only DF path
            return X.loc[:, self.mask_]

        # numpy path
        return X[:, self.mask_]

    def get_support(self):
        return self.mask_

    def get_feature_names_out(self, input_features=None):
        if getattr(self, "feature_names_enc_", None) is not None:
            return np.array(self.feature_names_enc_, dtype=object)[self.mask_]
        if input_features is None:
            return np.array([f"x{i}" for i in range(len(self.mask_))], dtype=object)[self.mask_]
        return np.array(input_features, dtype=object)[self.mask_]


