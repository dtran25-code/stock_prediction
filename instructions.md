# Instructions for ClaudeCode — Final Project Revision

**From:** CoWork (CEO)
**To:** ClaudeCode (CTO)
**Subject:** Rework `Project 4.ipynb` to close rubric gaps before final submission
**Deliverable folder:** `C:\Machine Learning class Notebooks\Project`

---

## 1. Context

This project is the final deliverable for the IEEE-CIS Fraud Detection problem. The current notebook (`Project 4.ipynb`) already contains a substantial first pass — EDA, manual cleaning, eight baseline models across three feature views, partial grid search, and written analysis. However, it does not yet line up with the Detailed Grading Rubric in several high-value areas. Your job is to bring the notebook, the Streamlit app, and the supporting artifacts into full rubric compliance and to produce a separate Executive Summary document.

The three source-of-truth files are in the project root:
- `Project_Spring_2026 (Updated).pdf` — project description, milestones, deliverables
- `Detailed Grading Rubric.pdf` — how the 100 points are allocated
- `Executive Summary.pdf` — template for the separate exec-summary deliverable

All CSV data files (`train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`) are in the project root. `sample_submission.csv` is present but should NOT be used for training — it is only the Kaggle submission template.

---

## 2. Authoritative Gap List (what's missing vs. rubric)

The rubric totals 100 points. Below is what the current notebook has, what's missing, and the estimated points at risk. Address every item marked MISSING or PARTIAL.

### 2.1 Data Cleaning & Feature Engineering must live inside a sklearn Pipeline

The rubric's opening instruction is non-negotiable:

> "to avoid data leakage, cleaning and feature engineering should ideally be performed within a pipeline."

The current notebook performs cleaning, encoding, imputation, scaling, feature selection, and oversampling as **loose imperative steps on dataframes**. This is the single biggest structural issue and jeopardizes **~30 points** (cleaning + feature engineering + resampling + model eval). You must refactor these into sklearn `Pipeline` / `ColumnTransformer` / `imblearn.pipeline.Pipeline` objects that are fit only on training folds.

**Required structure:**

```
imblearn.pipeline.Pipeline([
    ('preproc',   ColumnTransformer([...])),   # cleaning + FE as transformers
    ('resample',  SMOTE() or RandomOverSampler(...)),
    ('select',    SelectKBest(...)),            # or RFE, VIF filter, etc.
    ('clf',       <one of the models>),
])
```

Write custom transformers (subclass `BaseEstimator, TransformerMixin`) where sklearn doesn't ship one — e.g., `DropHighMissingCols`, `DropLowVariance`, `DropHighCorrelation`, `TargetEncoderHighCardinality`, `DateTimeExpander`, `TransactionRatioFeatures`.

### 2.2 Section-by-section gap table

| Rubric section | Pts | Status | What to do |
|---|---|---|---|
| General Analysis of Business Problem | 2.5 | ✅ Present (Cell 0) | Tighten wording if needed. |
| Data Collection — imports | 1.25 | ✅ Present (Cell 3) | Add `imblearn`, `shap`, `joblib`, `category_encoders` imports. |
| Data Collection — load/merge | 1.25 | ✅ Present (Cell 4) | Keep the `reduce_mem` helper and left-join on `TransactionID`. |
| **Data Cleaning (≥5 pipeline steps)** | **10** | ⚠️ PARTIAL — steps exist but NOT in pipeline | Convert all 5+ steps into transformer classes inside the pipeline: (1) handle missing values, (2) drop duplicates/outliers, (3) recode text/date fields, (4) adjust dtypes, (5) encode categoricals, (6) log-transform skewed numerics. Each must be a `.fit`/`.transform` step. |
| **Feature Engineering (≥10 pipeline steps)** | **20** | ❌ MISSING — ~5 ad-hoc steps, none in pipeline | Implement all 10 inside the pipeline, grouped per rubric: **Sanitization** (drop high-missing, drop low-variance, drop low target-corr, drop high-cardinality, drop redundant), **Creative** (interaction features, datetime expansion to hour/day/month, clustering via KMeans, aggregations like mean TransactionAmt per card1, ratio features like `TransactionAmt / card1_mean`), **Final Selection** (VIF/correlation filter, SelectKBest or mutual info, PCA on a tail of low-signal columns). |
| Data Visualization (≥5 plots) | 5 | ✅ Present (Cells 5–9) | Verify at least one univariate, one bivariate, and one multivariate plot. Add a target-variable bar plot if not already there. |
| **Resampling in pipeline** | **2** | ❌ MISSING — currently manual `resample()` | Use `imblearn.pipeline.Pipeline` with `SMOTE` (or `RandomOverSampler` for speed) as an explicit step. Remove the manual `X_train_bal` block (Cell 16). |
| 4 diverse models | 8 | ✅ Present (8 in notebook) | Keep the most diverse 4 as *primary* pipelines (Logistic Reg, Random Forest, Gradient Boosting, KNN or Naive Bayes) and present the other 4 as *extended comparison*. Each primary model must be its own full pipeline. |
| Scoring metric chosen | 1 | ✅ ROC-AUC | Keep ROC-AUC as primary; justify in markdown (imbalanced data → PR-AUC also reported). |
| Train/test split | 1 | ✅ Present | Keep stratified 80/20. |
| **K-fold CV results + box plot** | **2** | ❌ MISSING | Run `cross_val_score` with `StratifiedKFold(n_splits=5)` on each of the 4 primary pipelines, store per-fold scores, and render a **box plot** comparing the 4 models. |
| 4 test metrics reported | 4 | ✅ Present | Keep accuracy, precision, recall, ROC-AUC, PR-AUC in the consolidated table. |
| Best model selection + overfit check | 2 | ⚠️ PARTIAL | After picking the best pipeline, explicitly print train ROC-AUC vs test ROC-AUC. If gap > 0.05, discuss the mitigation already in place (regularization, max_depth, SMOTE). |
| **Grid search — ≥4 params** | **5** | ⚠️ PARTIAL — only 3 params in `rf_grid` | Expand grids so every tuned pipeline varies at least **4 hyperparameters**. For RF use e.g. `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`. For GB add `learning_rate`, `n_estimators`, `max_depth`, `subsample`. Keep `StratifiedKFold(n_splits=3)` + `scoring='roc_auc'`. |
| **Save fine-tuned pipeline** | **1.25** | ❌ MISSING | `joblib.dump(best_pipeline, 'fine_tuned_pipeline.joblib')`. Also produce `fine_tuned_pipeline.tar.gz` for the Streamlit loader. |
| Fine-tuned test metrics | 2.5 | ✅ Present (Cell 48) | Keep, but run on the *pipeline* (not the bare estimator). |
| Rank feature importance | 2.5 | ✅ Present (Cells 33–34, 52) | Keep; make sure the features come out of the fitted pipeline's `get_feature_names_out()`. |
| **Local explainability (SHAP)** | **2.5** | ❌ MISSING | Use `shap.TreeExplainer` (for RF/GB) or `shap.Explainer` with KernelExplainer fallback. Produce at least one waterfall plot for a single fraud case and one summary beeswarm plot. |
| **Save SHAP explainer** | **1.25** | ❌ MISSING | `joblib.dump(explainer, 'explainer_fraud.shap')`. |
| **AWS deployment** | **5** | ❌ MISSING in notebook | Add a section that packages the pipeline + `inference_project.py` into `model.tar.gz`, uploads to S3, and shows the `sagemaker.sklearn.model.SKLearnModel` deployment call. If live AWS is not available in this environment, provide the code blocks with a markdown note explaining they were executed in class. |
| **Streamlit web app (prediction + SHAP)** | **10** | ⚠️ PARTIAL — `StreamlitApp_Project.py` exists but inputs are only 3 features | Fix `MODEL_INFO["keys"]` and `MODEL_INFO["inputs"]` so the form collects the ~top-15 feature inputs the fine-tuned pipeline expects. Ensure the SHAP waterfall renders for class 1 (fraud). Add a fallback path that runs the pipeline locally if the SageMaker endpoint is unreachable, so the app is demoable without AWS. |
| **Executive Summary** (separate doc) | **10** | ❌ MISSING as a deliverable file | Produce a standalone `Executive Summary.docx` or `Executive Summary.pdf` following the `Executive Summary.pdf` template (objective, key results, how it would be used, key drivers, business impact, risks, recommendation). No code, no jargon, visuals encouraged. |

**Sum of at-risk points from MISSING/PARTIAL above: ~55 pts** — prioritize these.

---

## 3. Execution Plan (recommended order)

Work through these in order; each stage is independently testable.

### Stage A — Refactor into pipelines (highest leverage)
1. Create a new module `src/custom_classes.py` (new folder `src/` next to the notebook) that defines the custom transformers listed in §2.1. Keep each class pickleable — no lambdas, no closures over local state.
2. In the notebook, replace Cells 11–19 with a single "Pipeline Construction" section that builds:
   - a `ColumnTransformer` for numeric vs. categorical preprocessing,
   - an `imblearn.pipeline.Pipeline` wrapping cleaning → FE → resampling → feature selection → classifier.
3. Delete the manual oversampling cell (Cell 16) and the manual imputer/scaler cell (Cell 18) — they are now inside the pipeline.

### Stage B — Model training & K-fold
4. Define **four primary pipelines** (one per classifier). Each is its own `imblearn.pipeline.Pipeline` with an identical preproc prefix and a different final estimator.
5. For each primary pipeline, run `cross_val_score` with `StratifiedKFold(5)` on the training set only, record the 5 scores, and draw a **box plot** comparing the four distributions. This is the missing "K-fold validation results (2 pts)" item.
6. Fit each pipeline on full train, score on test across **accuracy, precision, recall, ROC-AUC, PR-AUC**, and consolidate into one results DataFrame.

### Stage C — Grid search & finalization
7. Pick the best primary pipeline by test ROC-AUC. Print train vs. test ROC-AUC to show the overfitting check.
8. Build a **GridSearchCV** over that pipeline with **≥4 hyperparameters** varied. Use `scoring='roc_auc'`, `cv=StratifiedKFold(3)`, `n_jobs=-1`.
9. Refit on the full training set, evaluate on test, and `joblib.dump` the fitted pipeline to `fine_tuned_pipeline.joblib`. Also tar-gzip it to `fine_tuned_pipeline.tar.gz` so `StreamlitApp_Project.py`'s existing loader works unchanged.

### Stage D — Explainability
10. Build a SHAP explainer on the preprocessed training sample (after the pipeline's preproc prefix, before the classifier). Use `TreeExplainer` for tree ensembles, else `Explainer(model.predict_proba, background)`.
11. Produce: (a) beeswarm summary over ~2,000 test transactions, (b) at least two waterfall plots — one confirmed fraud, one confirmed legit.
12. `joblib.dump` the explainer to `explainer_fraud.shap`.

### Stage E — Deployment + Streamlit
13. Fix `StreamlitApp_Project.py`:
    - Change `MODEL_INFO["keys"]` / `MODEL_INFO["inputs"]` to match the top-15 features from the fine-tuned pipeline.
    - Add a `USE_LOCAL_PIPELINE = True` fallback path that `joblib.load`s `fine_tuned_pipeline.joblib` when AWS creds aren't set, so the app runs without SageMaker.
    - Keep the SHAP waterfall block; verify it addresses class 1 (fraud).
14. Add a new notebook section "AWS SageMaker Deployment" that shows the packaging / upload / `SKLearnModel.deploy(...)` calls. If AWS isn't reachable in this environment, wrap it in a markdown header noting it was executed during the class demo and leave the code runnable.

### Stage F — Executive Summary deliverable
15. Generate a separate `Executive Summary.docx` (use the `docx` skill) following the template in `Executive Summary.pdf`. Pull numbers from the notebook's final results. Keep it 1–3 pages, prose-led, with at least one visual (e.g., risk-tier bar chart, feature-importance bar, confusion-matrix snapshot).

### Stage G — Final polish
16. Re-run the entire notebook top-to-bottom to confirm it executes cleanly.
17. Add a short "Reproducibility" markdown cell near the top listing Python / sklearn / imblearn / shap versions and the random seed.

---

## 4. Concrete File Manifest (what should exist at the end)

Everything below goes in `C:\Machine Learning class Notebooks\Project\`:

- `Project 4.ipynb` — fully re-run, pipeline-based, all rubric sections labeled with matching markdown headers.
- `src/custom_classes.py` — custom transformer classes used in the pipeline.
- `src/__init__.py` — empty, so `src` is importable.
- `fine_tuned_pipeline.joblib` — the saved fine-tuned pipeline.
- `fine_tuned_pipeline.tar.gz` — tarball for the Streamlit loader.
- `explainer_fraud.shap` — the saved SHAP explainer.
- `StreamlitApp_Project.py` — updated to match the final feature set and support local fallback.
- `inference_project.py` — keep as-is (already correct for SageMaker).
- `Executive Summary.docx` — new deliverable, template-compliant.
- `requirements.txt` — new file listing pinned versions for `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `shap`, `joblib`, `streamlit`, `sagemaker`, `boto3`, `category_encoders`, `matplotlib`, `seaborn`.

---

## 5. Hard Constraints

- **No data leakage.** Everything that touches test data must go through a pipeline that was fit on train. Do not `fit_transform` on test.
- **Random seed = 42** everywhere (`train_test_split`, `KFold`, `RandomForest`, `SMOTE`, etc.).
- **Handle memory.** The merged dataset is large; keep the `reduce_mem` dtype downcast, cap majority-class sampling if needed, and subsample for SHAP (e.g. 2,000 rows).
- **Do not remove the existing markdown narrative** in Cells 53–57 — they already satisfy rubric sections 8–10 and read well. Edit in place if you must, but preserve the substance.
- **Headers must match rubric sections verbatim** so the grader can check off each item: "General Analysis of the Business Problem", "Data Collection", "Data Cleaning", "Feature Engineering", "Data Visualization", "Models", "Finalize Model", "Deployment", "Conclusion — Executive Summary".

---

## 6. Answers to Your Open Questions

1. **AWS / SageMaker — live deployment, debug-first.**
   The existing deployment code in `StreamlitApp_Project.py` + `inference_project.py` is not currently working. Do **not** rewrite from scratch — read those two files, understand the intended shape (S3 → tarball → `SKLearnModel.deploy` → endpoint called by Streamlit), then debug. Credentials come from the user via `.streamlit/secrets.toml` under the `[aws_credentials]` section (keys already referenced in the app: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_BUCKET`, `AWS_ENDPOINT`). Ask Chan directly for those plus the IAM execution role ARN — don't hardcode anything.

2. **Streamlit — local only.**
   `streamlit run StreamlitApp_Project.py` against the saved `fine_tuned_pipeline.joblib` and the SageMaker endpoint is the target. No Streamlit Community Cloud deploy, no GitHub push, no Git LFS concerns.

3. **Executive Summary — `.docx`.**
   Use the `docx` skill. Chan can export to PDF themselves afterward if they want to lock it.

## 7. Module Naming — Match Existing Imports

`inference_project.py` already imports `from src.Custom_Classes import FeatureEngineer` and `StreamlitApp_Project.py` references `from src.Custom_Classes import DropHighMissingCols, TransactionFeatureEngineer, DropHighCorrelation`. **Use that exact path and filename: `src/Custom_Classes.py`** (CamelCase with underscore) — not `src/custom_classes.py` as loosely referenced in §2.1 / §4 above. Add an empty `src/__init__.py`.

## 8. Missing Imports — Add These Two New Cells

The current imports in Cell 3 of `Project 4.ipynb` are enough for the first-pass notebook but are missing everything required for Stages A–F. Two new cells have been staged in `new_import_cells.py` in the project root — paste them in right after the existing Cell 3.

**What they add (and why):**
- `BaseEstimator`, `TransformerMixin`, `ColumnTransformer` — required to write the custom pipeline transformers (Stage A).
- `imblearn.pipeline.Pipeline`, `SMOTE`, `RandomOverSampler`, `RandomUnderSampler` — required so the resampling step can live *inside* the pipeline (Stage A, the 2-pt rubric item).
- `OneHotEncoder`, `FunctionTransformer`, `PowerTransformer`, `MinMaxScaler` — FE building blocks for Stage A.
- `VarianceThreshold`, `mutual_info_classif`, `SelectFromModel`, `KMeans`, `PCA` — the Final-Selection rubric items (collinearity drop, MI, clustering, dimension reduction).
- `category_encoders.TargetEncoder`, `CountEncoder` — for the high-cardinality fix called out in the Suggestions cell.
- `accuracy_score`, `precision_score`, `recall_score`, `f1_score` — the notebook currently only imports `roc_auc_score` and `average_precision_score` explicitly, but the rubric asks for 4 metrics on test.
- `shap`, `joblib` — SHAP explainability + pipeline persistence (Stages C–D, ~5 pts).
- `boto3`, `sagemaker`, `SKLearnModel`, `get_execution_role`, `tarfile` — Stage E deployment.
- `sys.path.insert(0, PROJECT_ROOT)` — makes `src/Custom_Classes.py` importable so the fitted pipeline can pickle its custom transformers (critical — the pipeline won't unpickle inside SageMaker or Streamlit without this).

Cell A is a one-time `%pip install` for `imbalanced-learn shap category_encoders joblib boto3 sagemaker`. Cell B is the consolidated import block. Both are in `new_import_cells.py`; read that file and paste the two blocks in as new code cells.

---

Proceed end-to-end and report back with the updated notebook + the file manifest in §4.
