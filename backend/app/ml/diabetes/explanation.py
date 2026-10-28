"""SHAP explanations for the tuned diabetes readmission candidates."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
import shap
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


class PipelineExplanation(NamedTuple):
    shap_values: shap.Explanation
    positive_probabilities: pd.Series
    global_importance: pd.Series
    output_scale: str


def explain_pipeline(
    model: Pipeline,
    background_features: pd.DataFrame,
    examples: pd.DataFrame,
    *,
    background_size: int = 100,
    random_state: int = 42,
) -> PipelineExplanation:
    """Explain positive-class model outputs for raw, schema-matched examples.

    SHAP values are reported on each estimator's raw model-output scale:
    Logistic Regression log-odds or XGBoost margin. They are not probability
    point changes and do not describe causal effects.
    """
    if not isinstance(model, Pipeline) or not {"preprocessor", "classifier"}.issubset(model.named_steps):
        raise ValueError("Expected a fitted pipeline with preprocessor and classifier steps")
    if background_features.empty or examples.empty:
        raise ValueError("Background data and examples must both be non-empty")
    if background_size < 1:
        raise ValueError("background_size must be positive")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    expected_features = list(preprocessor.feature_names_in_)
    if list(background_features.columns) != expected_features:
        raise ValueError("Background feature columns do not match the fitted preprocessor")
    if list(examples.columns) != expected_features:
        raise ValueError("Example feature columns do not match the fitted preprocessor")

    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ValueError("Fitted classifier must contain positive class 1")
    positive_index = classes.index(1)

    if len(background_features) > background_size:
        background = background_features.sample(
            n=background_size,
            random_state=random_state,
        )
    else:
        background = background_features

    transformed_background = preprocessor.transform(background)
    transformed_examples = preprocessor.transform(examples)
    feature_names = list(preprocessor.get_feature_names_out())
    if transformed_examples.shape[1] != len(feature_names):
        raise ValueError("Transformed feature count does not match preprocessor feature names")

    if isinstance(classifier, LogisticRegression):
        explainer = shap.LinearExplainer(classifier, transformed_background)
        output_scale = "log-odds"
    elif isinstance(classifier, XGBClassifier):
        explainer = shap.TreeExplainer(classifier)
        output_scale = "raw margin"
    else:
        raise TypeError(
            "SHAP explanation currently supports LogisticRegression and XGBClassifier"
        )

    raw_explanation = explainer(transformed_examples)
    values = np.asarray(raw_explanation.values)
    base_values = np.asarray(raw_explanation.base_values)
    if values.ndim == 3:
        values = values[:, :, positive_index]
        if base_values.ndim == 2:
            base_values = base_values[:, positive_index]
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")

    display_data = (
        transformed_examples.toarray()
        if sparse.issparse(transformed_examples)
        else np.asarray(transformed_examples)
    )
    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=display_data,
        feature_names=feature_names,
    )
    global_importance = pd.Series(
        np.mean(np.abs(values), axis=0),
        index=feature_names,
        name="mean_absolute_shap_value",
    ).sort_values(ascending=False)
    probabilities = model.predict_proba(examples)[:, positive_index]

    return PipelineExplanation(
        explanation,
        pd.Series(probabilities, index=examples.index, name="probability_readmitted_within_30d"),
        global_importance,
        output_scale,
    )


def top_local_contributions(
    result: PipelineExplanation,
    *,
    row_index: int = 0,
    max_features: int = 10,
) -> pd.DataFrame:
    """Return signed encoded-feature contributions for one explained row."""
    if not 0 <= row_index < result.shap_values.values.shape[0]:
        raise IndexError("row_index is outside the explained batch")
    if max_features < 1:
        raise ValueError("max_features must be positive")

    values = result.shap_values.values[row_index]
    feature_names = list(result.shap_values.feature_names)
    encoded_values = result.shap_values.data[row_index]
    contributions = pd.DataFrame(
        {
            "feature": feature_names,
            "encoded_value": encoded_values,
            "contribution": values,
        }
    )
    contributions["direction"] = np.where(
        contributions["contribution"] >= 0,
        "toward <30 model output",
        "away from <30 model output",
    )
    return (
        contributions.assign(
            absolute_contribution=lambda frame: frame["contribution"].abs()
        )
        .sort_values("absolute_contribution", ascending=False)
        .drop(columns="absolute_contribution")
        .head(max_features)
        .reset_index(drop=True)
    )