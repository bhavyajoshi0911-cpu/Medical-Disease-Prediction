import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.app.ml.liver.persistence import save_liver_artifacts
from backend.app.ml.liver.preprocessing import build_liver_preprocessor


def test_saved_pipeline_preprocessor_and_metadata_round_trip(tmp_path) -> None:
    row_count = 60
    features = pd.DataFrame(
        {
            "Age": [((row * 3) % 13) for row in range(row_count)],
            "Gender": ["Female" if row % 2 else "Male" for row in range(row_count)],
            "Total_Bilirubin": [((row * 5) % 7) + 0.5 for row in range(row_count)],
            "Direct_Bilirubin": [((row * 2) % 5) + 0.2 for row in range(row_count)],
            "Alkaline_Phosphotase": [((row * 7) % 11) + 50 for row in range(row_count)],
            "Alamine_Aminotransferase": [((row * 11) % 13) + 20 for row in range(row_count)],
            "Aspartate_Aminotransferase": [((row * 13) % 15) + 25 for row in range(row_count)],
            "Total_Proteins": [((row * 17) % 9) + 5 for row in range(row_count)],
            "Albumin": [((row * 19) % 6) + 2 for row in range(row_count)],
            "Albumin_and_Globulin_Ratio": [((row * 23) % 5) + 1.0 for row in range(row_count)],
        }
    )
    target = pd.Series([row % 2 for row in range(row_count)])
    pipeline = Pipeline(
        [
            ("preprocessor", build_liver_preprocessor(features)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    ).fit(features, target)
    expected_probabilities = pipeline.predict_proba(features.iloc[:5])

    metadata = save_liver_artifacts(
        pipeline,
        tmp_path,
        model_version="test-version",
        source_sha256="a" * 64,
        metrics={"cv": {"pr_auc_mean": np.float64(0.25)}},
        selection_rationale="Test fixture only; not a clinical selection.",
    )

    restored_model = joblib.load(tmp_path / "liver_model.joblib")
    restored_preprocessor = joblib.load(tmp_path / "liver_preprocessor.joblib")
    stored_metadata = json.loads((tmp_path / "liver_metadata.json").read_text(encoding="utf-8"))
    assert np.allclose(restored_model.predict_proba(features.iloc[:5]), expected_probabilities)
    assert restored_preprocessor.get_feature_names_out().tolist() == pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out().tolist()
    assert stored_metadata == metadata
    assert metadata["metrics"]["cv"]["pr_auc_mean"] == 0.25
    assert metadata["educational_use_only"] is True
    schema = {field["name"]: field for field in metadata["feature_schema"]}
    assert schema["Age"]["type"] == "integer"
    assert schema["Gender"]["type"] == "category"
    assert schema["Gender"]["categories"] == ["Female", "Male"]
