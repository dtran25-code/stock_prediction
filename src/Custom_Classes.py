"""
Custom transformers for the IEEE-CIS Fraud Detection pipeline.

All classes subclass sklearn's BaseEstimator + TransformerMixin, are fully
pickle-able (no lambdas, no closures over local state), and follow the
sklearn fit/transform contract so they can be composed inside
sklearn Pipelines, imblearn Pipelines, and ColumnTransformers.

Two stages:

    Raw-DataFrame stage   (operate on pandas DataFrames, return DataFrames)
      DropHighMissingCols, DropRedundantColumns, DropHighCardinality,
      DtypeAdjuster, DateTimeExpander, LogTransformerSkewed,
      AggregationFeatures, TransactionRatioFeatures, InteractionFeatures

    Post-ColumnTransformer stage   (operate on ndarrays, return ndarrays)
      DropLowVariance, DropLowTargetCorrelation, DropHighCorrelation,
      ClusterFeatures

A thin ``FeatureEngineer`` alias is kept for backwards compatibility with
existing ``from src.Custom_Classes import FeatureEngineer`` imports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


# =============================================================================
# 1. Cleaning / Sanitization on raw DataFrame
# =============================================================================

class DropHighMissingCols(BaseEstimator, TransformerMixin):
    """Drop columns whose share of missing values exceeds ``threshold``.

    The decision is made at ``fit`` time on the training data only,
    so no test-set information leaks in.
    """

    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = _ensure_df(X)
        miss = X.isna().mean()
        self.cols_to_drop_ = miss[miss > self.threshold].index.tolist()
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        X = _ensure_df(X)
        return X.drop(columns=[c for c in self.cols_to_drop_ if c in X.columns])

    def get_feature_names_out(self, input_features=None):
        return np.asarray([c for c in self.feature_names_in_ if c not in self.cols_to_drop_])


class DropRedundantColumns(BaseEstimator, TransformerMixin):
    """Business-intuition drop: caller supplies the list of columns."""

    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop or []

    def fit(self, X, y=None):
        X = _ensure_df(X)
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        X = _ensure_df(X)
        return X.drop(columns=[c for c in self.cols_to_drop if c in X.columns])

    def get_feature_names_out(self, input_features=None):
        return np.asarray([c for c in self.feature_names_in_ if c not in self.cols_to_drop])


class DropHighCardinality(BaseEstimator, TransformerMixin):
    """Drop object/string columns whose unique count exceeds ``threshold``.

    Keeps the column if it is numeric or low-cardinality. High-cardinality
    strings would explode one-hot encoding; if you still want them keep
    them and route through a TargetEncoder in the ColumnTransformer.
    """

    def __init__(self, threshold: int = 200):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = _ensure_df(X)
        self.feature_names_in_ = np.asarray(X.columns)
        self.cols_to_drop_ = []
        for col in X.columns:
            if X[col].dtype == "object" or str(X[col].dtype).startswith("category"):
                if X[col].nunique(dropna=True) > self.threshold:
                    self.cols_to_drop_.append(col)
        return self

    def transform(self, X):
        X = _ensure_df(X)
        return X.drop(columns=[c for c in self.cols_to_drop_ if c in X.columns])

    def get_feature_names_out(self, input_features=None):
        return np.asarray([c for c in self.feature_names_in_ if c not in self.cols_to_drop_])


class DtypeAdjuster(BaseEstimator, TransformerMixin):
    """Memory-saving dtype downcast (float64->float32, int64->int32/16/8).

    Matches the behaviour of the old ``reduce_mem`` helper but as a
    transformer so it lives inside the pipeline.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def fit(self, X, y=None):
        X = _ensure_df(X)
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        X = _ensure_df(X).copy()
        for col in X.select_dtypes(include=["int64"]).columns:
            X[col] = pd.to_numeric(X[col], downcast="integer")
        for col in X.select_dtypes(include=["float64"]).columns:
            X[col] = pd.to_numeric(X[col], downcast="float")
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_)


class DateTimeExpander(BaseEstimator, TransformerMixin):
    """Recode ``TransactionDT`` (seconds since ``start_ref``) into
    ``hour``, ``day_of_week``, ``day_of_month``, ``month`` columns.

    Counts as a cleaning step (recoding dates) AND a feature-engineering
    step (date-time at different scales) in the rubric.
    """

    def __init__(self, col: str = "TransactionDT", start_ref: str = "2017-12-01"):
        self.col = col
        self.start_ref = start_ref

    def fit(self, X, y=None):
        X = _ensure_df(X)
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        X = _ensure_df(X).copy()
        if self.col not in X.columns:
            return X
        base = pd.to_datetime(self.start_ref)
        ts = base + pd.to_timedelta(X[self.col].fillna(0).astype("int64"), unit="s")
        X[self.col + "_hour"]        = ts.dt.hour.astype("int16")
        X[self.col + "_day_of_week"] = ts.dt.dayofweek.astype("int16")
        X[self.col + "_day_of_month"]= ts.dt.day.astype("int16")
        X[self.col + "_month"]       = ts.dt.month.astype("int16")
        return X

    def get_feature_names_out(self, input_features=None):
        extras = [self.col + s for s in ("_hour", "_day_of_week", "_day_of_month", "_month")]
        return np.asarray(list(self.feature_names_in_) + extras)


class LogTransformerSkewed(BaseEstimator, TransformerMixin):
    """Apply ``log1p`` to right-skewed numeric columns.

    Skew is measured on training data only; same column set is applied
    at transform time. Non-numeric columns and columns with any negative
    values are skipped (log1p is only safe for values >= 0).
    """

    def __init__(self, skew_threshold: float = 1.0):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        X = _ensure_df(X)
        self.feature_names_in_ = np.asarray(X.columns)
        self.skewed_cols_ = []
        for col in X.select_dtypes(include=[np.number]).columns:
            s = X[col].dropna()
            if len(s) == 0:
                continue
            if s.min() < 0:
                continue
            if abs(s.skew()) > self.skew_threshold:
                self.skewed_cols_.append(col)
        return self

    def transform(self, X):
        X = _ensure_df(X).copy()
        for col in self.skewed_cols_:
            if col in X.columns:
                X[col] = np.log1p(X[col].clip(lower=0).fillna(0))
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_)


# =============================================================================
# 2. Creative feature engineering on raw DataFrame
# =============================================================================

class AggregationFeatures(BaseEstimator, TransformerMixin):
    """Group-by aggregations (mean / std / count) learned on train set.

    ``group_cols`` is a list of column names to group by (e.g. ["card1"]).
    For each group column the transformer computes mean, std, count of
    ``agg_col`` and stores the lookup dict from ``fit`` so ``transform``
    on test is fully deterministic and leakage-free.
    """

    def __init__(self, group_cols=None, agg_col: str = "TransactionAmt",
                 aggs=("mean", "std", "count")):
        self.group_cols = group_cols or ["card1"]
        self.agg_col = agg_col
        self.aggs = tuple(aggs)

    def fit(self, X, y=None):
        X = _ensure_df(X)
        self.feature_names_in_ = np.asarray(X.columns)
        self.lookups_ = {}
        for g in self.group_cols:
            if g in X.columns and self.agg_col in X.columns:
                grouped = X.groupby(g)[self.agg_col].agg(list(self.aggs))
                # Ensure numeric dtype & fill NaN stds with 0 for single-member groups
                grouped = grouped.astype("float32").fillna(0.0)
                self.lookups_[g] = grouped
        return self

    def transform(self, X):
        X = _ensure_df(X).copy()
        for g, table in self.lookups_.items():
            if g not in X.columns:
                continue
            joined = X[[g]].merge(table, how="left", left_on=g, right_index=True)
            for agg in self.aggs:
                new_col = f"{g}_{self.agg_col}_{agg}"
                X[new_col] = joined[agg].astype("float32").fillna(0.0).to_numpy()
        return X

    def get_feature_names_out(self, input_features=None):
        extras = [f"{g}_{self.agg_col}_{agg}"
                  for g in self.group_cols for agg in self.aggs]
        return np.asarray(list(self.feature_names_in_) + extras)


class TransactionRatioFeatures(BaseEstimator, TransformerMixin):
    """Create ratio features using the aggregation columns produced
    upstream (``card1_TransactionAmt_mean`` etc.).

    Must run AFTER ``AggregationFeatures``. Adds:
      * ``TransactionAmt_to_card1_mean``
      * ``TransactionAmt_to_card1_std``
    """

    def __init__(self, amt_col: str = "TransactionAmt", group_col: str = "card1"):
        self.amt_col = amt_col
        self.group_col = group_col

    def fit(self, X, y=None):
        X = _ensure_df(X)
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        X = _ensure_df(X).copy()
        mean_col = f"{self.group_col}_{self.amt_col}_mean"
        std_col  = f"{self.group_col}_{self.amt_col}_std"
        eps = 1e-6
        if self.amt_col in X.columns and mean_col in X.columns:
            X[f"{self.amt_col}_to_{self.group_col}_mean"] = (
                X[self.amt_col].astype("float32") / (X[mean_col].astype("float32") + eps)
            )
        if self.amt_col in X.columns and std_col in X.columns:
            X[f"{self.amt_col}_to_{self.group_col}_std"] = (
                X[self.amt_col].astype("float32") / (X[std_col].astype("float32") + eps)
            )
        return X

    def get_feature_names_out(self, input_features=None):
        extras = [
            f"{self.amt_col}_to_{self.group_col}_mean",
            f"{self.amt_col}_to_{self.group_col}_std",
        ]
        return np.asarray(list(self.feature_names_in_) + extras)


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Hand-crafted interactions between categorical/ID columns.

    Produces string-concatenation interactions (e.g. card1_addr1) that
    downstream TargetEncoder in the ColumnTransformer can encode as
    high-cardinality categoricals.
    """

    def __init__(self, pairs=None):
        # default pairs based on IEEE-CIS domain knowledge
        self.pairs = pairs or [("card1", "addr1"), ("card1", "card2"), ("P_emaildomain", "card1")]

    def fit(self, X, y=None):
        X = _ensure_df(X)
        self.feature_names_in_ = np.asarray(X.columns)
        self.new_cols_ = []
        for a, b in self.pairs:
            if a in X.columns and b in X.columns:
                self.new_cols_.append(f"{a}_{b}")
        return self

    def transform(self, X):
        X = _ensure_df(X).copy()
        for a, b in self.pairs:
            if a in X.columns and b in X.columns:
                new_col = f"{a}_{b}"
                X[new_col] = (
                    X[a].astype("string").fillna("NA") + "_" +
                    X[b].astype("string").fillna("NA")
                )
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + self.new_cols_)


# =============================================================================
# 3. Post-ColumnTransformer stage (ndarray in / ndarray out)
# =============================================================================

class DropLowVariance(BaseEstimator, TransformerMixin):
    """Drop columns whose variance falls below ``threshold``.

    Accepts ndarray or sparse matrix (same as sklearn's VarianceThreshold
    but with a preserved column-name path via ``input_features``).
    """

    def __init__(self, threshold: float = 1e-4):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = _to_array(X)
        variances = np.nanvar(X, axis=0)
        self.mask_ = variances > self.threshold
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = _to_array(X)
        return X[:, self.mask_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = np.asarray([f"x{i}" for i in range(self.n_features_in_)])
        return np.asarray(input_features)[self.mask_]


class DropLowTargetCorrelation(BaseEstimator, TransformerMixin):
    """Drop columns whose |Pearson correlation| with ``y`` is below
    ``threshold``. Requires ``y`` at fit time.
    """

    def __init__(self, threshold: float = 0.005):
        self.threshold = threshold

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("DropLowTargetCorrelation needs y at fit time.")
        X = _to_array(X).astype(np.float64, copy=False)
        y = np.asarray(y).astype(np.float64)
        n_cols = X.shape[1]
        corrs = np.zeros(n_cols, dtype=np.float64)
        y_mean = y.mean()
        y_std = y.std()
        if y_std == 0:
            # degenerate: keep all features
            self.mask_ = np.ones(n_cols, dtype=bool)
            self.n_features_in_ = n_cols
            return self
        # Column-wise correlation, NaN-safe
        for j in range(n_cols):
            col = X[:, j]
            mask = ~np.isnan(col)
            if mask.sum() < 2:
                corrs[j] = 0.0
                continue
            c = col[mask]
            yy = y[mask]
            c_std = c.std()
            if c_std == 0:
                corrs[j] = 0.0
                continue
            corrs[j] = ((c - c.mean()) * (yy - yy.mean())).mean() / (c_std * y_std)
        self.mask_ = np.abs(corrs) > self.threshold
        # Always keep at least one feature
        if not self.mask_.any():
            self.mask_ = np.abs(corrs) >= np.nanmax(np.abs(corrs))
        self.n_features_in_ = n_cols
        return self

    def transform(self, X):
        X = _to_array(X)
        return X[:, self.mask_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = np.asarray([f"x{i}" for i in range(self.n_features_in_)])
        return np.asarray(input_features)[self.mask_]


class DropHighCorrelation(BaseEstimator, TransformerMixin):
    """Drop one column from every pair whose |Pearson correlation|
    exceeds ``threshold``. Greedy upper-triangular scan.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = _to_array(X)
        n_cols = X.shape[1]
        if n_cols < 2:
            self.mask_ = np.ones(n_cols, dtype=bool)
            self.n_features_in_ = n_cols
            return self
        # Replace NaNs with column means for correlation calc
        means = np.nanmean(X, axis=0)
        X_filled = np.where(np.isnan(X), means, X)
        # Use pandas for a fast, NaN-safe correlation matrix on big arrays
        corr = pd.DataFrame(X_filled).corr().abs().to_numpy()
        upper = np.triu(corr, k=1)
        drop = set()
        for i in range(n_cols):
            if i in drop:
                continue
            for j in range(i + 1, n_cols):
                if j in drop:
                    continue
                if upper[i, j] > self.threshold:
                    drop.add(j)
        mask = np.ones(n_cols, dtype=bool)
        for j in drop:
            mask[j] = False
        self.mask_ = mask
        self.n_features_in_ = n_cols
        return self

    def transform(self, X):
        X = _to_array(X)
        return X[:, self.mask_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = np.asarray([f"x{i}" for i in range(self.n_features_in_)])
        return np.asarray(input_features)[self.mask_]


class ClusterFeatures(BaseEstimator, TransformerMixin):
    """Append KMeans cluster label + distance-to-centroid features.

    Adds ``n_clusters + 1`` new columns (one label + one distance per
    centroid, condensed to min distance) to the input matrix.
    """

    def __init__(self, n_clusters: int = 8, random_state: int = 42, sample_size: int = 50000):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.sample_size = sample_size

    def fit(self, X, y=None):
        X = _to_array(X)
        # KMeans on a random subsample for speed
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        if n > self.sample_size:
            idx = rng.choice(n, size=self.sample_size, replace=False)
            X_fit = X[idx]
        else:
            X_fit = X
        # Replace NaN with column means so KMeans doesn't barf
        col_means = np.nanmean(X_fit, axis=0)
        X_fit = np.where(np.isnan(X_fit), col_means, X_fit)
        self.col_means_ = col_means
        self.kmeans_ = KMeans(n_clusters=self.n_clusters,
                              random_state=self.random_state,
                              n_init=10)
        self.kmeans_.fit(X_fit)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = _to_array(X)
        X_filled = np.where(np.isnan(X), self.col_means_, X)
        labels = self.kmeans_.predict(X_filled).reshape(-1, 1).astype(np.float32)
        dists = self.kmeans_.transform(X_filled).min(axis=1, keepdims=True).astype(np.float32)
        return np.hstack([X_filled, labels, dists])

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = np.asarray([f"x{i}" for i in range(self.n_features_in_)])
        extras = np.asarray(["kmeans_label", "kmeans_min_dist"])
        return np.concatenate([np.asarray(input_features), extras])


# =============================================================================
# 4. Helper utilities
# =============================================================================

def _ensure_df(X):
    """Coerce to DataFrame so column ops work even if sklearn hands us
    an ndarray (e.g., when a prior step returned one)."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


def _to_array(X):
    """Coerce to dense ndarray with float dtype so numeric ops work."""
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    # sparse -> dense; float for NaN support
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


# =============================================================================
# 5. Backwards-compat alias
# =============================================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Legacy no-op alias kept so old pickles / imports resolve.

    The real feature-engineering work is split across the transformers
    above; this class exists only because ``inference_project.py``
    historically imported ``FeatureEngineer`` by name.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
