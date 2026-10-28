import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from backend.app.ml.liver.preprocessing import NUMERIC_FEATURES
from backend.app.ml.liver.training import (
    build_candidate_estimators,
    train_candidate_models,
)


def test_baseline_registry_contains_five_imbalance_aware_candidates() -> None:
    training_target = pd.Series([0] * 16 + [1] * 2)

    estimators = build_candidate_estimators(training_target, random_state=13)

    assert set(estimators) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "linear_svm",
        "xgboost",
    }
    assert isinstance(estimators["logistic_regression"], LogisticRegression)
    assert isinstance(estimators["decision_tree"], DecisionTreeClassifier)
    assert isinstance(estimators["random_forest"], RandomForestClassifier)
    assert isinstance(estimators["linear_svm"], LinearSVC)
    assert isinstance(estimators["xgboost"], XGBClassifier)
    assert estimators["xgboost"].get_params()["scale_pos_weight"] == 8.0


def test_candidate_registry_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="both binary classes"):
        build_candidate_estimators(pd.Series([0, 0, 0]))


def test_all_candidate_pipelines_fit_and_predict() -> None:
    row_count = 40
    training_features = pd.DataFrame(
        {
            feature: [((row * (index + 3)) % 11) for row in range(row_count)]
            for index, feature in enumerate(NUMERIC_FEATURES)
        }
    )
    training_features["Gender"] = ["Female" if row % 2 else "Male" for row in range(row_count)]
    training_target = pd.Series([row % 2 for row in range(row_count)])

    training_run = train_candidate_models(training_features, training_target)

    assert len(training_run.models) == 5
    assert set(training_run.fit_seconds) == set(training_run.models)
    assert all(seconds >= 0 for seconds in training_run.fit_seconds.values())
    assert all(
        pipeline.predict(training_features).shape == (row_count,)
        for pipeline in training_run.models.values()
    )
