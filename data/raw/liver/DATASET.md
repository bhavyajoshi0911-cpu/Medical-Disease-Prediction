# Liver Disease Dataset

## Dataset Name

Indian Liver Patient Dataset (ILPD).

## Source

- Repository: UCI Machine Learning Repository
- Dataset page: https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset
- Original archive: https://archive.ics.uci.edu/static/public/225/ilpd+indian+liver+patient+dataset.zip
- License: Creative Commons Attribution 4.0 International (CC BY 4.0). Attribution is required.
- Planned downloaded archive: `ilpd-indian-liver-patient-dataset.zip`
- Planned extracted source file: `indian_liver_patient_dataset.csv`

## Size And Structure

- Rows: 583 patient records.
- Feature count: 10 clinical and laboratory attributes plus a target column.
- Target: `Dataset` with values `1` (patient has liver disease) and `2` (no liver disease). This is a binary disease label, not a medical diagnosis.
- Data type: numeric fields and a small number of categorical/encoded medical indicators.
- The dataset is small and historically used in medical ML examples; it is not a substitute for a current clinical validation study.

## Feature Descriptions

The clinical features are typically used in a standard binary classification task:

| Feature | Description |
| --- | --- |
| `Age` | Patient age in years. |
| `Gender` | Biological sex label from the source data. |
| `Total_Bilirubin` | Total bilirubin concentration. |
| `Direct_Bilirubin` | Direct bilirubin concentration. |
| `Alkaline_Phosphotase` | ALP level. |
| `Alamine_Aminotransferase` | ALT level. |
| `Aspartate_Aminotransferase` | AST level. |
| `Total_Proteins` | Total protein level. |
| `Albumin` | Albumin level. |
| `Albumin_and_Globulin_Ratio` | Albumin-to-globulin ratio. |
| `Dataset` | Target label for liver disease status. |

## Target And Prediction Task

The source target is `Dataset`:

| Source value | Meaning | Planned binary label |
| --- | --- | --- |
| `1` | Liver disease present | Positive |
| `2` | Liver disease absent | Negative |

The planned task is to estimate whether a patient is likely to have liver disease based on available clinical indicators. This is a dataset-specific educational demo and not a medical diagnostic system.

## Missing Values

The dataset may contain missing values in lab-derived attributes; they must be treated explicitly in the preprocessing pipeline. Do not silently drop rows or assume the missingness is random. Missing-value handling must be fitted only on training data.

## Known Limitations

- The dataset is small and historically sourced; it does not represent the full population or current clinical practice.
- A binary label from one historical dataset cannot establish disease prevalence, diagnostic performance, or treatment benefit.
- Model performance depends on the exact target definition, sample selection, and preprocessing decisions.
- Clinical variables can be correlated and partially redundant; feature engineering must be defended and kept transparent.
- Results on this dataset do not establish clinical effectiveness or real-world deployment readiness.

## Preprocessing Plan

No preprocessing has been applied to the source files yet. The planned pipeline is:

1. Preserve the original downloaded CSV unchanged.
2. Validate column names, data types, target encoding, and missingness patterns.
3. Define a consistent binary target for the educational task.
4. Encode or transform the categorical `Gender` field with training-only fitting logic.
5. Impute missing numeric values using a transformer fitted only on training data.
6. Train a model with a sklearn `Pipeline` to avoid leakage between train and evaluation sets.
7. Compare candidate models using cross-validation and a documented priority metric.
8. Save model metadata, artifact paths, and explanatory outputs before exposure in the API.

No processed dataset, fitted preprocessor, or model is included in this phase.

## Citation

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Indian Liver Patient Dataset. https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset
