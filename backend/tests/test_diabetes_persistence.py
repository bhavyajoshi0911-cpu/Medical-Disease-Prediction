import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.app.ml.diabetes.persistence import save_diabetes_artifacts
from backend.app.ml.diabetes.preprocessing import NUMERIC_FEATURES, build_diabetes_preprocessor


def test_saved_pipeline_preprocessor_and_metadata_round_trip(tmp_path) -> None:
    row_count = 60
    features = pd.DataFrame(
        {
            name: [((row * (index + 3)) % 13) for row in range(row_count)]
            for index, name in enumerate(NUMERIC_FEATURES)
        }
    )
    features["gender"] = ["female" if row % 2 else "male" for row in range(row_count)]
    target = pd.Series([row % 2 for row in range(row_count)])
    pipeline = Pipeline(
        [
            ("preprocessor", build_diabetes_preprocessor(features)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    ).fit(features, target)
    expected_probabilities = pipeline.predict_proba(features.iloc[:5])

    metadata = save_diabetes_artifacts(
        pipeline,
        tmp_path,
        model_version="test-version",
        source_sha256="a" * 64,
        metrics={"cv": {"pr_auc_mean": np.float64(0.25)}},
        selection_rationale="Test fixture only; not a clinical selection.",
    )

    restored_model = joblib.load(tmp_path / "diabetes_model.joblib")
    restored_preprocessor = joblib.load(tmp_path / "diabetes_preprocessor.joblib")
    stored_metadata = json.loads((tmp_path / "model_metadata.json").read_text(encoding="utf-8"))
    assert np.allclose(restored_model.predict_proba(features.iloc[:5]), expected_probabilities)
    assert restored_preprocessor.get_feature_names_out().tolist() == pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out().tolist()
    assert stored_metadata == metadata
    assert metadata["metrics"]["cv"]["pr_auc_mean"] == 0.25
    assert metadata["educational_use_only"] is True
    schema = {field["name"]: field for field in metadata["feature_schema"]}
    assert schema["time_in_hospital"]["minimum"] == 1
    assert schema["time_in_hospital"]["maximum"] == 14
    assert schema["gender"]["type"] == "category"
    assert schema["gender"]["categories"] == ["female", "male"]