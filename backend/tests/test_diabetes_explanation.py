import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from backend.app.ml.diabetes.explanation import explain_pipeline, top_local_contributions
from backend.app.ml.diabetes.preprocessing import NUMERIC_FEATURES, build_diabetes_preprocessor


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    row_count = 80
    features = pd.DataFrame(
        {
            name: [((row * (column_index + 3)) % 13) for row in range(row_count)]
            for column_index, name in enumerate(NUMERIC_FEATURES)
        }
    )
    features["gender"] = ["female" if row % 2 else "male" for row in range(row_count)]
    target = pd.Series([row % 2 for row in range(row_count)], name="readmitted_30d")
    return features, target


@pytest.mark.parametrize("classifier_kind", ["logistic", "xgboost"])
def test_explanation_has_aligned_shap_values_and_local_directions(
    sample_data: tuple[pd.DataFrame, pd.Series],
    classifier_kind: str,
) -> None:
    features, target = sample_data
    preprocessor = build_diabetes_preprocessor(features)
    if classifier_kind == "logistic":
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
        expected_scale = "log-odds"
    else:
        classifier = XGBClassifier(
            n_estimators=4,
            max_depth=2,
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
            tree_method="hist",
        )
        expected_scale = "raw margin"
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    ).fit(features.iloc[:60], target.iloc[:60])

    result = explain_pipeline(
        model,
        features.iloc[:40],
        features.iloc[60:65],
        background_size=20,
    )
    local = top_local_contributions(result, max_features=5)

    assert result.shap_values.values.shape[0] == 5
    assert result.shap_values.values.shape[1] == len(result.shap_values.feature_names)
    assert result.positive_probabilities.between(0, 1).all()
    assert result.output_scale == expected_scale
    assert len(result.global_importance) == result.shap_values.values.shape[1]
    assert local["direction"].str.contains("<30 model output").all()


def test_explanation_rejects_schema_mismatch(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = sample_data
    model = Pipeline(
        [
            ("preprocessor", build_diabetes_preprocessor(features)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    ).fit(features.iloc[:60], target.iloc[:60])

    with pytest.raises(ValueError, match="columns do not match"):
        explain_pipeline(model, features.iloc[:20], features.iloc[60:65].drop(columns="gender"))