"""Consistent baseline comparison for the diabetes readmission candidates."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from backend.app.ml.diabetes.training import CandidateTrainingRun

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


class CandidateComparison(NamedTuple):
    holdout_metrics: pd.DataFrame
    cross_validation_metrics: pd.DataFrame
    confusion_matrices: dict[str, np.ndarray]
    score_types: dict[str, str]


def _positive_class_scores(model: object, features: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return probability or decision scores for positive class 1."""
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        raise ValueError("Fitted candidate must expose positive class 1")
    positive_index = classes.index(1)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        return np.asarray(probabilities)[:, positive_index], "probability"
    if hasattr(model, "decision_function"):
        decision_scores = np.asarray(model.decision_function(features))
        if decision_scores.ndim == 1:
            if positive_index == 0:
                decision_scores = -decision_scores
        else:
            decision_scores = decision_scores[:, positive_index]
        return decision_scores, "decision score"
    raise TypeError("Candidate must provide predict_proba or decision_function")


def compare_candidate_models(
    training_run: CandidateTrainingRun,
    training_features: pd.DataFrame,
    training_target: pd.Series,
    training_groups: pd.Series,
    holdout_features: pd.DataFrame,
    holdout_target: pd.Series,
    holdout_groups: pd.Series,
    *,
    cv_splits: int = 3,
    random_state: int = 42,
) -> CandidateComparison:
    """Cross-validate training data and evaluate fitted candidates on holdout.

    Cross-validation uses patient-grouped folds. The supplied fitted pipelines
    are evaluated on the untouched holdout once; no model ranking or selection
    is performed by this function.
    """
    aligned = (
        training_features.index.equals(training_target.index)
        and training_target.index.equals(training_groups.index)
        and holdout_features.index.equals(holdout_target.index)
        and holdout_target.index.equals(holdout_groups.index)
    )
    if not aligned:
        raise ValueError("Features, targets, and patient groups must have aligned indexes")
    if set(training_groups).intersection(holdout_groups):
        raise ValueError("Training and holdout partitions contain overlapping patients")
    if set(training_target.unique()) != {0, 1} or set(holdout_target.unique()) != {0, 1}:
        raise ValueError("Training and holdout targets must each contain classes 0 and 1")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")

    splitter = StratifiedGroupKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    cv_records: list[dict[str, object]] = []
    holdout_records: list[dict[str, object]] = []
    confusion_matrices: dict[str, np.ndarray] = {}
    score_types: dict[str, str] = {}

    for name, model in training_run.models.items():
        cv_result = cross_validate(
            model,
            training_features,
            training_target,
            groups=training_groups,
            cv=splitter,
            scoring=SCORING,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        for metric in SCORING:
            fold_scores = cv_result[f"test_{metric}"]
            cv_records.append(
                {
                    "model": name,
                    "metric": metric,
                    "mean": float(np.mean(fold_scores)),
                    "std": float(np.std(fold_scores, ddof=1)),
                }
            )

        predictions = model.predict(holdout_features)
        positive_scores, score_type = _positive_class_scores(model, holdout_features)
        score_types[name] = score_type
        confusion_matrices[name] = confusion_matrix(
            holdout_target,
            predictions,
            labels=[0, 1],
        )
        holdout_records.append(
            {
                "model": name,
                "accuracy": accuracy_score(holdout_target, predictions),
                "precision": precision_score(holdout_target, predictions, zero_division=0),
                "recall": recall_score(holdout_target, predictions, zero_division=0),
                "f1": f1_score(holdout_target, predictions, zero_division=0),
                "roc_auc": roc_auc_score(holdout_target, positive_scores),
                "pr_auc": average_precision_score(holdout_target, positive_scores),
                "score_type": score_type,
                "training_seconds": training_run.fit_seconds[name],
            }
        )

    holdout_metrics = pd.DataFrame(holdout_records).set_index("model")
    cross_validation_metrics = pd.DataFrame(cv_records).set_index(["model", "metric"])
    return CandidateComparison(
        holdout_metrics,
        cross_validation_metrics,
        confusion_matrices,
        score_types,
    )