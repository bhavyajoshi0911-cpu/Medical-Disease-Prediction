"""Bounded patient-grouped hyperparameter searches for selected candidates."""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from backend.app.ml.diabetes.training import CandidateTrainingRun

TUNABLE_CANDIDATES = ("logistic_regression", "xgboost")
PARAMETER_DISTRIBUTIONS: dict[str, dict[str, list[object]]] = {
    "logistic_regression": {
        "classifier__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        "classifier__class_weight": [
            "balanced",
            {0: 1.0, 1: 4.0},
            {0: 1.0, 1: 8.0},
            {0: 1.0, 1: 12.0},
        ],
    },
    "xgboost": {
        "classifier__n_estimators": [80, 120, 160],
        "classifier__max_depth": [3, 4, 5],
        "classifier__learning_rate": [0.03, 0.06, 0.1],
        "classifier__subsample": [0.75, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.85, 1.0],
        "classifier__scale_pos_weight": [4.0, 8.0, 12.0],
    },
}


class CandidateTuningRun(NamedTuple):
    searches: dict[str, RandomizedSearchCV]
    summary: pd.DataFrame


def tune_selected_candidates(
    training_run: CandidateTrainingRun,
    training_features: pd.DataFrame,
    training_target: pd.Series,
    training_groups: pd.Series,
    *,
    candidate_names: tuple[str, ...] = TUNABLE_CANDIDATES,
    cv_splits: int = 3,
    n_iter: int = 4,
    random_state: int = 42,
) -> CandidateTuningRun:
    """Tune selected estimators using grouped CV on training rows only.

    Average precision (PR-AUC) is the per-candidate search objective because
    the positive class is uncommon. Results compare parameter configurations
    within each named candidate; this function does not compare candidates or
    inspect a holdout partition.
    """
    if not (
        training_features.index.equals(training_target.index)
        and training_target.index.equals(training_groups.index)
    ):
        raise ValueError("Training features, target, and groups must have aligned indexes")
    if not candidate_names:
        raise ValueError("At least one candidate must be selected for tuning")
    unknown = set(candidate_names) - set(PARAMETER_DISTRIBUTIONS)
    unavailable = set(candidate_names) - set(training_run.models)
    if unknown or unavailable:
        raise ValueError(
            f"Unsupported candidates: {sorted(unknown | unavailable)}; "
            f"supported: {sorted(PARAMETER_DISTRIBUTIONS)}"
        )
    if cv_splits < 2 or n_iter < 1:
        raise ValueError("cv_splits must be at least 2 and n_iter must be positive")

    splitter = StratifiedGroupKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    searches: dict[str, RandomizedSearchCV] = {}
    summary_rows: list[dict[str, object]] = []

    for candidate_name in candidate_names:
        search = RandomizedSearchCV(
            estimator=training_run.models[candidate_name],
            param_distributions=PARAMETER_DISTRIBUTIONS[candidate_name],
            n_iter=n_iter,
            scoring="average_precision",
            cv=splitter,
            refit=True,
            n_jobs=1,
            random_state=random_state,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(training_features, training_target, groups=training_groups)
        searches[candidate_name] = search
        best_index = int(search.best_index_)
        summary_rows.append(
            {
                "candidate": candidate_name,
                "search_objective": "average_precision",
                "best_cv_mean": float(search.cv_results_["mean_test_score"][best_index]),
                "best_cv_std": float(search.cv_results_["std_test_score"][best_index]),
                "best_params": search.best_params_,
                "configurations_tested": len(search.cv_results_["params"]),
            }
        )

    return CandidateTuningRun(searches, pd.DataFrame(summary_rows).set_index("candidate"))