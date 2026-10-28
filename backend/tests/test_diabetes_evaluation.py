import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.app.ml.diabetes.evaluation import compare_candidate_models
from backend.app.ml.diabetes.training import CandidateTrainingRun


def test_comparison_reports_cv_and_holdout_metrics_without_ranking() -> None:
    training_features = pd.DataFrame(
        {"signal": np.tile([0.0, 1.0], 60)},
        index=pd.RangeIndex(0, 120),
    )
    training_target = pd.Series(np.tile([0, 1], 60), index=training_features.index)
    training_groups = pd.Series(training_features.index, index=training_features.index)
    holdout_features = pd.DataFrame(
        {"signal": np.tile([0.0, 1.0], 20)},
        index=pd.RangeIndex(120, 160),
    )
    holdout_target = pd.Series(np.tile([0, 1], 20), index=holdout_features.index)
    holdout_groups = pd.Series(holdout_features.index, index=holdout_features.index)
    model = Pipeline([("classifier", LogisticRegression())]).fit(
        training_features,
        training_target,
    )
    training_run = CandidateTrainingRun(
        models={"logistic_regression": model},
        fit_seconds={"logistic_regression": 0.1},
    )

    result = compare_candidate_models(
        training_run,
        training_features,
        training_target,
        training_groups,
        holdout_features,
        holdout_target,
        holdout_groups,
        cv_splits=3,
    )

    assert result.holdout_metrics.loc["logistic_regression", "accuracy"] > 0.9
    assert result.holdout_metrics.loc["logistic_regression", "roc_auc"] > 0.9
    assert result.confusion_matrices["logistic_regression"].shape == (2, 2)
    assert "roc_auc" in result.cross_validation_metrics.index.get_level_values("metric")
    assert result.score_types["logistic_regression"] == "probability"


def test_comparison_rejects_patient_overlap() -> None:
    features = pd.DataFrame({"signal": [0, 1, 0, 1]}, index=range(4))
    target = pd.Series([0, 1, 0, 1], index=features.index)
    groups = pd.Series([1, 2, 3, 4], index=features.index)
    model = Pipeline([("classifier", LogisticRegression())]).fit(features, target)
    training_run = CandidateTrainingRun({"logistic_regression": model}, {"logistic_regression": 0.1})

    with pytest.raises(ValueError, match="overlapping patients"):
        compare_candidate_models(
            training_run,
            features.iloc[:2],
            target.iloc[:2],
            groups.iloc[:2],
            features.iloc[2:],
            target.iloc[2:],
            pd.Series([1, 4], index=features.index[2:]),
            cv_splits=2,
        )