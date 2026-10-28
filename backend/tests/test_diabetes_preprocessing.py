from pathlib import Path

import pandas as pd
import pytest

from backend.app.ml.diabetes.preprocessing import (
    EXCLUDED_FEATURES,
    IDENTIFIER_COLUMNS,
    build_diabetes_preprocessor,
    encode_readmission_target,
    load_diabetes_dataset,
    split_by_patient,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIABETES_CSV = PROJECT_ROOT / "data" / "raw" / "diabetes" / "source" / "diabetic_data.csv"


def test_readmission_target_mapping_and_validation() -> None:
    labels = pd.Series(["NO", ">30", "<30"])

    encoded = encode_readmission_target(labels)

    assert encoded.tolist() == [0, 0, 1]
    assert encoded.name == "readmitted_30d"
    with pytest.raises(ValueError, match="Unexpected or missing"):
        encode_readmission_target(pd.Series(["unknown"]))


@pytest.fixture(scope="module")
def diabetes_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if not DIABETES_CSV.is_file():
        pytest.skip("Download the documented UCI source dataset to run integration checks")
    return load_diabetes_dataset(DIABETES_CSV)


def test_loader_preserves_categories_and_excludes_identifiers_and_weight(
    diabetes_data: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    predictors, target, groups = diabetes_data

    assert len(predictors) == 101766
    assert target.value_counts().to_dict() == {0: 90409, 1: 11357}
    assert set(IDENTIFIER_COLUMNS).isdisjoint(predictors.columns)
    assert set(EXCLUDED_FEATURES).isdisjoint(predictors.columns)
    assert predictors["max_glu_serum"].eq("None").any()
    assert groups.nunique() == 71518


def test_patient_split_has_no_group_overlap_and_preserves_alignment(
    diabetes_data: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    predictors, target, groups = diabetes_data

    split = split_by_patient(predictors, target, groups)

    assert set(split.groups_train).isdisjoint(split.groups_test)
    assert split.X_train.index.equals(split.y_train.index)
    assert split.X_test.index.equals(split.y_test.index)
    assert 0.08 < split.y_test.mean() < 0.15


def test_preprocessor_is_unfitted_until_training_data_is_supplied(
    diabetes_data: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    predictors, target, groups = diabetes_data
    split = split_by_patient(predictors, target, groups)
    preprocessor = build_diabetes_preprocessor(split.X_train)

    assert not hasattr(preprocessor, "transformers_")
    transformed_train = preprocessor.fit_transform(split.X_train.iloc[:2000])
    transformed_test = preprocessor.transform(split.X_test.iloc[:200])

    assert transformed_train.shape[0] == 2000
    assert transformed_test.shape[0] == 200
    assert transformed_train.shape[1] == transformed_test.shape[1]