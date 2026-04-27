# ════════════════════════════════════════════════════════════════════════════
# CELL A — one-time package install (run once, then comment out)
# Paste as a new code cell right after the existing import cell (Cell 3).
# ════════════════════════════════════════════════════════════════════════════
%pip install -q imbalanced-learn shap category_encoders joblib boto3 sagemaker


# ════════════════════════════════════════════════════════════════════════════
# CELL B — extended imports for Stages A–F (pipelines, FE, SHAP, deployment)
# Paste directly after Cell A.
# ════════════════════════════════════════════════════════════════════════════

# ── Standard-library helpers ──────────────────────────────────────────────────
import os, sys, json, tarfile
from pathlib import Path
from datetime import datetime

# ── Pipeline scaffolding (sklearn + imblearn) ─────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline            # keep sklearn version for simple chains
from imblearn.pipeline import Pipeline as ImbPipeline          # REQUIRED when a resampling step is in the chain
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# ── Additional preprocessing / feature-engineering building blocks ────────────
from sklearn.preprocessing import (
    OneHotEncoder, FunctionTransformer, PowerTransformer, MinMaxScaler,
)
from sklearn.feature_selection import (
    mutual_info_classif, VarianceThreshold, SelectFromModel,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Target / frequency encoders for high-cardinality categoricals (DeviceInfo, id_31, …)
try:
    from category_encoders import TargetEncoder, CountEncoder
    HAS_CATEGORY_ENCODERS = True
except ImportError:
    HAS_CATEGORY_ENCODERS = False
    print("category_encoders not installed — run the pip cell above.")

# ── Extended metrics for the 4-metric rubric table ────────────────────────────
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)

# ── Explainability + model persistence ────────────────────────────────────────
import shap
import joblib

# ── AWS / SageMaker deployment (Stage E) ──────────────────────────────────────
# These imports are safe even without live credentials; the deploy cells
# themselves will fail-fast if boto3 can't authenticate.
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
try:
    from sagemaker import get_execution_role
except ImportError:
    get_execution_role = None  # only available inside a SageMaker notebook instance

# ── Make the project's `src/` package importable so the fitted pipeline can
#    pickle / unpickle its custom transformer classes (needed by both the
#    Streamlit app and the SageMaker inference container). ─────────────────────
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Extended imports loaded.")
print(f"  scikit-learn:  available")
print(f"  imbalanced-learn (imblearn): loaded")
print(f"  shap:          {shap.__version__}")
print(f"  joblib:        {joblib.__version__}")
print(f"  boto3:         {boto3.__version__}")
print(f"  sagemaker:     {sagemaker.__version__}")
print(f"  category_encoders available: {HAS_CATEGORY_ENCODERS}")
