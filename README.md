# MedPredict

MedPredict is an educational AI/ML platform that demonstrates disease-related risk prediction using publicly available healthcare datasets. The planned system provides machine-learning predictions, model evaluation metrics, and SHAP-based explanations through a web interface and REST API.

> **Educational use only.** Model outputs are dataset-specific estimates, not diagnoses, treatment guidance, or medical advice. They must not be used to diagnose, treat, or prevent disease or to make real medical decisions. Consult a qualified healthcare professional for medical advice.

## Project Status

Phases 1-15 are complete. Diabetes EDA is in [notebooks/diabetes_eda.ipynb](notebooks/diabetes_eda.ipynb); baseline training, grouped-CV comparison, holdout evaluation, tuning, SHAP examples, and artifact persistence are in [notebooks/diabetes_training.ipynb](notebooks/diabetes_training.ipynb). The responsive React frontend includes a model overview, performance dashboard, dynamic diabetes assessment, dedicated result view, and prediction history. Metrics and explainability reports are saved under `reports/`. The tuned XGBoost pipeline is the educational demo artifact, selected using grouped-CV PR-AUC/recall trade-offs with SHAP support; this is not a clinical model selection or validation. The same holdout was inspected during baseline comparison, so its tuned-model metrics are exploratory rather than untouched external validation. Model and preprocessor artifacts are local under `backend/trained_models/`; versioned metadata records the dataset hash, schema, target, rationale, and metrics. FastAPI validates diabetes prediction requests from the persisted 46-feature schema and returns a probability estimate with SHAP contributors. Prediction history stores outcome metadata only; submitted health inputs are never stored. The official UCI archive and its unchanged source files are present locally, with documentation in [data/raw/diabetes/DATASET.md](data/raw/diabetes/DATASET.md). Heart and liver modules remain later work.

## Architecture

The intended request flow is:

```text
React + Vite frontend
        | HTTP / JSON
        v
FastAPI routes -> Pydantic validation -> prediction service
                                         |             |
                                         v             v
                                  disease model    explanation service
                                         |             |
                                         +------ SQLite history/metadata
```

Disease implementations will live behind separate model configurations and services, so adding a module does not require changing the whole application. SQLite will store prediction history and model metadata; raw health inputs will not be persisted by default. The frontend will obtain assessment fields from each model's declared feature configuration rather than maintaining a second, hand-written feature list.

## Phase 1 Dataset And Task

The first module is based on the **Diabetes 130-US Hospitals for Years 1999-2008** dataset from the UCI Machine Learning Repository. UCI describes 101,766 hospital encounters involving patients diagnosed with diabetes and 47 features; it contains missing values and includes demographic and encounter information. Dataset documentation and limitations are recorded in [data/raw/diabetes/DATASET.md](data/raw/diabetes/DATASET.md).

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- License: CC BY 4.0; attribution is required.
- Original target: `readmitted`, with `NO`, `>30`, and `<30` categories.
- Planned task: estimate whether a recorded encounter is followed by readmission within 30 days. `<30` is positive; `NO` and `>30` are grouped as negative. This label definition and its limitations will be explicit in dataset documentation.
- This dataset does **not** support a claim that the model predicts whether an individual has diabetes.

The data contains repeated encounters for some patients. The training design will exclude identifier fields from predictors and use patient-aware splitting when feasible so encounters from one patient do not appear on both sides of evaluation. Metrics will describe only this dataset and evaluation setup, not clinical effectiveness.

## Planned ML Pipeline

```text
Documented raw data -> validation and profiling -> train/test split
 -> training-only sklearn preprocessing -> cross-validation and model comparison
 -> bounded hyperparameter search -> held-out evaluation
 -> SHAP explanations -> serialized model + metadata -> FastAPI prediction service
```

Cleaning choices, missing-value handling, categorical encoding, feature selection, and any class-imbalance strategy will be justified and fitted without test-data leakage. SMOTE will not be applied unless analysis supports it, and if used it will run only inside training folds. Evaluation will include accuracy, precision, recall, F1, ROC-AUC, PR-AUC where appropriate, and confusion matrices; model choice will consider trade-offs rather than accuracy alone. No results will be reported until they have been generated by the reproducible pipeline.

## Run Backend

From the repository root, install dependencies and start the API:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload
```

The current endpoints are `GET /api/health`, `GET /api/models`, `GET /api/models/diabetes`, `POST /api/predict/diabetes`, `GET /api/history`, `GET /api/history/{prediction_id}`, and `DELETE /api/history/{prediction_id}`. The prediction request schema uses actual feature names, numeric constraints, and categorical values from the fitted model metadata. History stores predictions and model metadata only, not input features. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Repository Structure

```text
backend/
  app/{api,schemas,services,db,ml/{diabetes,heart,liver}}/
  trained_models/
  tests/
frontend/
  src/{components,pages,services,hooks,utils}/
data/
  raw/{diabetes,heart,liver}/
  processed/
  external/
notebooks/
reports/figures/
```

`data/raw/` is reserved for unchanged source files, which are ignored by Git; dataset documentation remains tracked. Processed data, notebooks, model artifacts, APIs, and the UI will be added in their respective phases.

## Development Phases

1. Project structure (complete)
2. Download and document datasets (diabetes complete)
3. Diabetes EDA (complete)
4. Diabetes preprocessing (complete)
5. Train multiple diabetes models (complete)
6. Compare models (complete)
7. Tune selected models (complete)
8. Add SHAP explainability (complete)
9. Save selected model and metadata (complete)
10. Build FastAPI backend (complete)
11. Add prediction API (complete)
12. Build React frontend (complete)
13. Implement prediction result UI (complete)
14. Add prediction history (complete; metadata only)
15. Build model performance dashboard (complete)
16. Add heart disease module
17. Add liver disease module
18. Complete testing
19. Polish responsive UI/UX
20. Complete README and deployment preparation

## Disclaimer

This project is for education and portfolio demonstration only. Dataset-specific model probabilities are not medical diagnoses or medical advice, and this project does not establish clinical effectiveness. Do not use it for real medical decisions.