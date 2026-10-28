import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.app.ml.diabetes.training import CandidateTrainingRun
from backend.app.ml.diabetes.tuning import tune_selected_candidates


def test_tuning_uses_grouped_cv_and_returns_refitted_search() -> None:
    features = pd.DataFrame(
        {"signal": [index % 2 for index in range(120)]},
        index=pd.RangeIndex(120),
    )
    target = pd.Series([index % 2 for index in range(120)], index=features.index)
    groups = pd.Series(features.index, index=features.index)
    model = Pipeline([("classifier", LogisticRegression(max_iter=1000))]).fit(features, target)
    run = CandidateTrainingRun({"logistic_regression": model}, {"logistic_regression": 0.1})

    tuning = tune_selected_candidates(
        run,
        features,
        target,
        groups,
        candidate_names=("logistic_regression",),
        cv_splits=3,
        n_iter=2,
    )

    search = tuning.searches["logistic_regression"]
    assert search.best_estimator_ is not None
    assert search.n_splits_ == 3
    assert len(search.cv_results_["params"]) == 2
    assert tuning.summary.loc["logistic_regression", "search_objective"] == "average_precision"


def test_tuning_rejects_unknown_candidate() -> None:
    features = pd.DataFrame({"signal": [0, 1, 0, 1]})
    target = pd.Series([0, 1, 0, 1])
    groups = pd.Series([1, 2, 3, 4])
    run = CandidateTrainingRun({}, {})

    with pytest.raises(ValueError, match="Unsupported candidates"):
        tune_selected_candidates(
            run,
            features,
            target,
            groups,
            candidate_names=("unknown",),
        )