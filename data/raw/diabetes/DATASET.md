# Diabetes 130-US Hospitals Dataset

## Dataset Name

Diabetes 130-US Hospitals for Years 1999-2008.

## Source

- Repository: UCI Machine Learning Repository
- Dataset page: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- Original archive: https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip
- License: Creative Commons Attribution 4.0 International (CC BY 4.0). Attribution is required.
- Downloaded archive: `diabetes-130-us-hospitals-1999-2008.zip`
- Extracted source files: `source/diabetic_data.csv` and `source/IDS_mapping.csv`

## Size And Structure

- Rows: 101,766 hospital encounters.
- UCI feature count: 47.
- CSV columns: 50, consisting of the 47 features, two identifiers (`encounter_id`, `patient_nbr`), and the original target (`readmitted`).
- Unique patients: 71,518. Multiple encounters for a patient are present.
- Duplicate encounter identifiers observed: 0.
- Feature types: categorical and integer, as described by UCI. The raw CSV contains coded and categorical values; types must be validated during preprocessing.

The row and column counts above were checked against the downloaded CSV. The UCI feature count excludes identifiers and the target.

## Feature Descriptions

The 47 feature columns are grouped here by their documented role. Identifier columns are listed separately and are not model predictors.

| Feature group | Columns | Description |
| --- | --- | --- |
| Demographics | `race`, `gender`, `age`, `weight` | Race, gender, age band in ten-year intervals, and weight in pounds. |
| Encounter codes | `admission_type_id`, `discharge_disposition_id`, `admission_source_id` | Coded encounter type, discharge destination/status, and admission source. `IDS_mapping.csv` supplies the source dictionaries. |
| Encounter context | `time_in_hospital`, `payer_code`, `medical_specialty` | Length of stay in days, payer code, and admitting physician specialty. |
| Utilization and counts | `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses` | Counts of procedures, medications, diagnoses, and prior-year outpatient, emergency, and inpatient visits, plus laboratory procedures. |
| Diagnosis codes | `diag_1`, `diag_2`, `diag_3` | Primary, secondary, and additional diagnosis codes recorded for the encounter. |
| Test results | `max_glu_serum`, `A1Cresult` | Categorical blood glucose serum and HbA1c result bands, including a `None` category where no result is represented. |
| Diabetes medication status | `metformin`, `repaglinide`, `nateglinide`, `chlorpropamide`, `glimepiride`, `acetohexamide`, `glipizide`, `glyburide`, `tolbutamide`, `pioglitazone`, `rosiglitazone`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`, `examide`, `citoglipton`, `insulin`, `glyburide-metformin`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone` | Per-encounter medication status, represented using categories such as `No`, `Up`, `Down`, and `Steady`. |
| Treatment change indicators | `change`, `diabetesMed` | Whether diabetes medication changed during the encounter and whether diabetes medication was prescribed. |

Identifiers: `encounter_id` uniquely identifies an encounter; `patient_nbr` identifies a patient and is repeated across encounters. They must not be used as predictive features. `patient_nbr` may be used only to keep a patient's encounters together when splitting data.

## Target And Prediction Task

The source target is `readmitted`, with values `NO`, `>30`, and `<30`:

| Source value | Meaning | Planned binary label |
| --- | --- | --- |
| `<30` | Readmitted within 30 days | Positive: 30-day readmission |
| `>30` | Readmitted after 30 days | Negative for the within-30-day task |
| `NO` | No readmission recorded | Negative for the within-30-day task |

Observed source counts:

| Source value | Encounters | Share |
| --- | ---: | ---: |
| `<30` | 11,357 | 11.16% |
| `>30` | 35,545 | 34.93% |
| `NO` | 54,864 | 53.91% |

The planned task is **30-day hospital readmission prediction for an encounter involving a patient diagnosed with diabetes**. The binary positive class is `<30`; `>30` and `NO` are grouped as negative. This is a label transformation of the source target, not a diabetes diagnosis. It also combines no readmission with readmission after 30 days, which must be considered when interpreting model results.

## Missing Values

The CSV uses `?` to represent missing values. The following counts were measured from the downloaded file:

| Column | Missing rows | Share of rows |
| --- | ---: | ---: |
| `weight` | 98,569 | 96.86% |
| `medical_specialty` | 49,949 | 49.08% |
| `payer_code` | 40,256 | 39.56% |
| `race` | 2,273 | 2.23% |
| `diag_3` | 1,423 | 1.40% |
| `diag_2` | 358 | 0.35% |
| `diag_1` | 21 | 0.02% |

Do not treat `?` as an ordinary category or silently drop incomplete rows. In particular, the very high missingness in `weight` requires an explicit modeling decision.

## Known Limitations

- The data covers inpatient encounters from 130 US hospitals and integrated delivery networks during 1999-2008; it does not represent current care or the general population.
- Included encounters already concern patients diagnosed with diabetes and meet selection conditions including a 1-14 day stay, laboratory testing, and administered medication. The dataset cannot support diabetes diagnosis or population screening claims.
- It is observational historical data with repeated encounters per patient. Random row splitting could place the same patient's records in both training and evaluation sets; use patient-aware splitting and report its limitations.
- Demographic fields include race, gender, and age band. Their presence creates privacy and fairness considerations and does not imply the model is fair across groups.
- Missingness is substantial and may not be random. Missing-value handling can affect both performance and which records are represented.
- `readmitted` is a hospital-record outcome. Readmissions outside the source's observable records may not be captured, and the binary task merges `>30` with `NO`.
- Features describe the hospitalization, including length of stay, discharge disposition, procedures, and medication activity. A prediction-time definition is required before using them; variables unavailable at the intended prediction time must be excluded to avoid outcome leakage.
- Results on this dataset do not establish clinical effectiveness, transportability, or benefit to patients.

## Preprocessing Plan

No preprocessing has been applied to the source files. The following steps are planned for a separate processed dataset and training pipeline:

1. Keep the downloaded archive and extracted CSV files unchanged; preserve `?` in raw data.
2. Validate expected columns, row counts, target values, identifier uniqueness, and documented ID mappings.
3. Convert `?` to missing values in the processing pipeline and report missingness before choosing imputation, exclusion, or feature removal. Do not remove outliers solely because they are extreme.
4. Exclude `weight` from the initial predictor set because 96.86% of its values are missing; revisit only with a documented justification.
5. Exclude `encounter_id` and `patient_nbr` from predictors. Use `patient_nbr` only for patient-grouped splitting.
6. Treat categorical codes and grouped values as categorical, not continuous measurements. Apply mappings only where the source dictionary defines them.
7. Define the binary target as described above and document the class imbalance.
8. Fit imputers, encoders, scaling, feature selection, and any resampling only on training folds using an sklearn `Pipeline`/`ColumnTransformer`. Do not apply SMOTE unless training-only analysis supports it.
9. Define which features are available at prediction time before training; exclude any feature that would not be known at that point.

No processed dataset, fitted preprocessor, or model is included in this phase.

## Citation

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999-2008* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

Dataset documentation and license: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008