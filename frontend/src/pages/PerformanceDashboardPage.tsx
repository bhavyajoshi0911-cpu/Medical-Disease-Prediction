import { useEffect, useState } from "react";
import { ArrowRight, BarChart3, Gauge, ShieldAlert } from "lucide-react";
import { Link } from "react-router-dom";
import { getDiabetesModel } from "../services/api";
import type { ModelInfo } from "../types";

function formatNumber(value: number | undefined, digits = 3): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "—";
  }
  return value.toFixed(digits);
}

function formatPercent(value: number | undefined, digits = 1): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "—";
  }
  return `${(value * 100).toFixed(digits)}%`;
}

export function PerformanceDashboardPage() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getDiabetesModel().then(setModel).catch((reason: Error) => setError(reason.message));
  }, []);

  const tuning = model?.metrics.tuning_cv_best_average_precision as
    | (ModelInfo["metrics"]["tuning_cv_best_average_precision"] & {
        selection_note?: string;
      })
    | undefined;
  const holdout = model?.metrics.post_selection_holdout as
    | (ModelInfo["metrics"]["post_selection_holdout"] & {
        confusion_matrix_labels_0_1?: number[][];
      })
    | undefined;
  const confusion = holdout?.confusion_matrix_labels_0_1 ?? [
    [0, 0],
    [0, 0],
  ];
  const summaryCards = [
    {
      label: "Grouped CV PR-AUC",
      value: formatNumber(tuning?.mean),
      detail: tuning ? `${tuning.folds}-fold CV estimate` : "loading metadata",
      tone: "mint",
    },
    {
      label: "Holdout PR-AUC",
      value: formatNumber(holdout?.pr_auc),
      detail: holdout ? "Selection-phase metric" : "loading metadata",
      tone: "gold",
    },
    {
      label: "Holdout ROC-AUC",
      value: formatNumber(holdout?.roc_auc),
      detail: holdout ? "Discrimination summary" : "loading metadata",
      tone: "coral",
    },
    {
      label: "Accuracy",
      value: formatPercent(holdout?.accuracy),
      detail: holdout ? "Overall classification rate" : "loading metadata",
      tone: "mint",
    },
    {
      label: "Recall",
      value: formatPercent(holdout?.recall),
      detail: holdout ? "Positive class capture" : "loading metadata",
      tone: "coral",
    },
    {
      label: "F1 score",
      value: formatPercent(holdout?.f1),
      detail: holdout ? `Precision ${formatPercent(holdout.precision)}` : "loading metadata",
      tone: "gold",
    },
  ];

  return (
    <div className="page-stack page-enter">
      <section className="page-heading">
        <div>
          <p className="eyebrow">MODEL WORKSPACE <span>·</span> PHASE 15</p>
          <h1>Model performance dashboard</h1>
          <p className="heading-subtitle">Dataset-specific evaluation metrics for the deployed diabetes readmission model.</p>
        </div>
        <Link className="button button-primary" to="/assessment/diabetes">Start assessment <ArrowRight size={17} /></Link>
      </section>

      <div className="notice-bar" role="note">
        <ShieldAlert size={18} />
        <p><strong>Exploratory model evaluation.</strong> Results reflect this dataset and setup only. They are not clinical evidence and should not be used for medical decisions.</p>
      </div>

      {error && <div className="inline-error" role="alert">{error}</div>}

      <section className="dashboard-summary-grid" aria-label="Model performance summary">
        {summaryCards.map((card) => (
          <article key={card.label} className="metric-card">
            <div className={`metric-icon metric-icon-${card.tone}`}>
              <Gauge size={18} />
            </div>
            <p className="metric-label">{card.label}</p>
            <strong className="metric-value">{card.value}</strong>
            <span className="metric-foot">{card.detail}</span>
          </article>
        ))}
      </section>

      <section className="lower-grid">
        <article className="panel dashboard-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">EVALUATION SNAPSHOT</p>
              <h2>{model?.algorithm ?? "Loading model"}</h2>
            </div>
            <span className={`availability ${model?.available ? "availability-ready" : "availability-pending"}`}><span />{model?.available ? "Available" : "Unavailable"}</span>
          </div>

          <div className="model-details">
            <div><span>Model version</span><strong>{model?.model_version ?? "—"}</strong></div>
            <div><span>Training date</span><strong>{model ? new Date(model.training_date).toLocaleDateString() : "—"}</strong></div>
            <div><span>Input features</span><strong>{model?.features.length ?? "—"}</strong></div>
          </div>

          <div className="metric-context">
            <div className="metric-context-row"><span><BarChart3 size={15} /> Selection search</span><strong>{tuning ? `${tuning.configurations_tested} configurations` : "—"}</strong></div>
            <div className="metric-context-row"><span><Gauge size={15} /> CV average precision</span><strong>{tuning ? `${formatNumber(tuning.mean)} ± ${formatNumber(tuning.std)}` : "—"}</strong></div>
            <p>{tuning?.selection_note ?? "Model metrics are being loaded from the saved metadata artifact."}</p>
          </div>
        </article>

        <article className="panel dashboard-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">CONFUSION MATRIX</p>
              <h2>Holdout outcome counts</h2>
            </div>
            <BarChart3 size={19} className="panel-heading-icon" />
          </div>

          <div className="dashboard-matrix" aria-label="Holdout confusion matrix">
            <div className="matrix-cell">{confusion[0]?.[0] ?? 0}</div>
            <div className="matrix-cell">{confusion[0]?.[1] ?? 0}</div>
            <div className="matrix-cell">{confusion[1]?.[0] ?? 0}</div>
            <div className="matrix-cell">{confusion[1]?.[1] ?? 0}</div>
          </div>

          <div className="matrix-label"><span>Actual 0</span><span>Actual 1</span></div>
          <div className="matrix-label"><span>Predicted 0</span><span>Predicted 1</span></div>

          <p className="dashboard-footnote">{holdout?.partition_note ?? "The holdout partition was inspected during model selection, so these values are exploratory rather than untouched validation."}</p>
        </article>
      </section>

      <section className="phase-strip">
        <div className="phase-strip-mark">MP</div>
        <div><strong>Model metrics are dataset-specific</strong><span>They describe the educational diabetes readmission setup, not clinical performance.</span></div>
        <Link to="/assessment/diabetes" aria-label="Return to the assessment form"><ArrowRight size={19} /></Link>
      </section>
    </div>
  );
}
