import { useEffect, useRef, useState } from "react";
import { ArrowDown, ArrowLeft, ArrowRight, CircleAlert, FileText, Gauge, LoaderCircle, ShieldAlert } from "lucide-react";
import { Link, useLocation, useParams } from "react-router-dom";
import { getDiabetesModel, getPredictionHistoryItem } from "../services/api";
import type { ModelInfo, PredictionHistoryItem, PredictionResult } from "../types";

type ResultState = {
  result?: PredictionResult;
  model?: ModelInfo;
};

export function ResultPage() {
  const location = useLocation();
  const { id } = useParams();
  const explanationRef = useRef<HTMLElement | null>(null);
  const state = location.state as ResultState | null;
  const [savedRecord, setSavedRecord] = useState<PredictionHistoryItem | null>(null);
  const [savedModel, setSavedModel] = useState<ModelInfo | null>(state?.model ?? null);
  const [historyLoading, setHistoryLoading] = useState(!state?.result);
  const [historyError, setHistoryError] = useState("");

  useEffect(() => {
    if (state?.result) return;
    let active = true;
    const recordId = Number(id);
    if (!Number.isSafeInteger(recordId) || recordId < 1) {
      setHistoryError("This result is only available in the browser session where the assessment was completed.");
      setHistoryLoading(false);
      return () => { active = false; };
    }
    setHistoryLoading(true);
    Promise.all([getPredictionHistoryItem(recordId), getDiabetesModel()])
      .then(([record, modelInfo]) => {
        if (!active) return;
        setSavedRecord(record);
        setSavedModel(modelInfo);
      })
      .catch((reason: Error) => {
        if (active) setHistoryError(reason.message);
      })
      .finally(() => {
        if (active) setHistoryLoading(false);
      });
    return () => { active = false; };
  }, [id, state?.result]);

  const liveResult = state?.result;
  const model = state?.model ?? savedModel;
  const probability = liveResult?.probability ?? savedRecord?.probability;
  const modelName = liveResult?.model ?? savedRecord?.model_name;
  const modelVersion = liveResult?.model_version ?? savedRecord?.model_version;
  const predictionTarget = liveResult?.prediction_target ?? savedRecord?.prediction_target;
  const diseaseName = model?.disease ?? liveResult?.disease ?? "diabetes";
  const isLiver = diseaseName === "liver";
  const topFeatures = liveResult?.top_features ?? [];
  const disclaimer = liveResult?.disclaimer ?? "This saved record contains only prediction metadata. Feature inputs and per-assessment explanations were not retained.";

  if (historyLoading) {
    return <div className="loading-panel page-enter"><LoaderCircle className="spin" size={19} /> Loading saved assessment</div>;
  }

  if (probability === undefined || !modelName || !modelVersion || !predictionTarget) {
    return (
      <div className="page-stack page-enter result-missing">
        <p className="eyebrow">ASSESSMENT RESULT</p>
        <CircleAlert size={25} />
        <h1>Result unavailable</h1>
        <p>{historyError || "This result is not available in the current browser session."}</p>
        <div className="result-missing-actions">
          <Link className="button button-quiet" to="/history">View history</Link>
          <Link className="button button-primary" to={isLiver ? "/assessment/liver" : "/assessment/diabetes"}>Start a new assessment <ArrowRight size={16} /></Link>
        </div>
      </div>
    );
  }

  const probabilityPercent = Math.round(probability * 1000) / 10;
  const topMagnitude = Math.max(...topFeatures.map((feature) => Math.abs(feature.contribution)), 0.001);
  const cvMetrics = model?.metrics.tuning_cv_best_average_precision;
  const holdoutMetrics = model?.metrics.post_selection_holdout;

  return (
    <div className="page-stack page-enter result-page">
      <section className="page-heading result-heading">
        <div>
          <Link to={isLiver ? "/assessment/liver" : "/assessment/diabetes"} className="back-link"><ArrowLeft size={15} /> Assessment</Link>
          <p className="eyebrow">ASSESSMENT RESULT <span>·</span> {isLiver ? "LIVER CLINICAL" : "DIABETES ENCOUNTER"}</p>
          <h1>Model estimate</h1>
          <p className="heading-subtitle">{isLiver ? "Dataset-specific liver disease risk estimate." : "Dataset-specific prediction for 30-day hospital readmission."}</p>
        </div>
        <div className="result-heading-actions">
          <Link to="/history" className="button button-quiet">View history</Link>
          <button
            type="button"
            className="button button-quiet"
            onClick={() => explanationRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
          >
            View explanation <ArrowDown size={15} />
          </button>
          <Link to={isLiver ? "/assessment/liver" : "/assessment/diabetes"} className="button button-primary">New assessment <ArrowRight size={16} /></Link>
        </div>
      </section>

      <div className="notice-bar result-notice" role="note">
        <ShieldAlert size={18} />
        <p><strong>Educational estimate only.</strong> This is not a medical diagnosis, treatment recommendation, or medical advice. Consult a qualified healthcare professional for medical decisions.</p>
      </div>

      <section className="result-layout">
        <div className="result-main-column">
          <article className="result-summary-card">
            <div className="result-summary-top">
              <div>
                <p className="eyebrow">PREDICTION TARGET</p>
                <h2>{isLiver ? "Liver disease status" : "30-day hospital readmission"}</h2>
              </div>
              <span className="result-target-chip">{isLiver ? "Positive label" : "&lt;30 label"}</span>
            </div>
            <div className="result-estimate-row">
              <div>
                <span className="result-estimate-label">Estimated model probability</span>
                <div className="result-probability">{probabilityPercent}<span>%</span></div>
              </div>
              {liveResult ? (
                <div className={`prediction-label ${liveResult.prediction === 1 ? "prediction-positive" : "prediction-negative"}`}>
                  <span className="prediction-indicator" />
                  <span>{liveResult.prediction === 1 ? (isLiver ? "Predicted label: liver disease present" : "Predicted label: within 30 days") : (isLiver ? "Predicted label: no liver disease" : "Predicted label: not within 30 days")}</span>
                </div>
              ) : <div className="prediction-label"><span className="prediction-indicator" />Saved history estimate</div>}
            </div>
            <div className="probability-track result-track" role="meter" aria-label="Estimated model probability" aria-valuemin={0} aria-valuemax={100} aria-valuenow={probabilityPercent}>
              <span style={{ width: `${probabilityPercent}%` }} />
            </div>
            <p className="calibration-note">Model probability calibration has not been assessed. The estimate is based on the selected dataset and model.</p>
            <div className="result-model-facts">
              <div><span>Model</span><strong>{modelName}</strong></div>
              <div><span>Version</span><strong>{modelVersion}</strong></div>
              <div><span>Input features</span><strong>{model?.features.length ?? "—"}</strong></div>
            </div>
          </article>

          <article className="result-explanation-card" id="shap-explanation" ref={explanationRef}>
            <div className="explanation-heading">
              <div className="explanation-heading-icon"><Gauge size={18} /></div>
              <div>
                <p className="eyebrow">EXPLAINABLE AI</p>
                <h2>What influenced this prediction?</h2>
              </div>
              <span className="shap-tag">SHAP</span>
            </div>
            <p className="explanation-intro">These features influenced the model output. Positive and negative directions refer to the model’s raw output, not changes in probability or causes.</p>
            <div className="contribution-chart" role="list" aria-label="Top model feature contributions">
              {topFeatures.map((feature) => {
                const positive = feature.contribution >= 0;
                const width = Math.max(3, Math.abs(feature.contribution) / topMagnitude * 100);
                return (
                  <div className="contribution-row" role="listitem" key={feature.feature}>
                    <div className="contribution-row-top">
                      <span className="contribution-name">{feature.feature.replaceAll("_", " ")}</span>
                      <span className={`contribution-value ${positive ? "contribution-positive" : "contribution-negative"}`}>
                        {positive ? "+" : ""}{feature.contribution.toFixed(3)}
                      </span>
                    </div>
                    <div className="contribution-track"><span className={positive ? "bar-positive" : "bar-negative"} style={{ width: `${width}%` }} /></div>
                    <span className="contribution-direction">{positive ? "Toward the <30 model output" : "Away from the <30 model output"}</span>
                  </div>
                );
              })}
              {topFeatures.length === 0 && (
                <p className="saved-explanation-note">Feature-level explanations are not retained in history because submitted health inputs are not stored.</p>
              )}
            </div>
            <div className="explanation-caveat"><CircleAlert size={16} /><span>SHAP describes how the model used features; it does not establish that any feature caused an outcome.</span></div>
          </article>
        </div>

        <aside className="result-side-column">
          <article className="result-info-card">
            <div className="side-card-heading"><FileText size={17} /><h2>Model information</h2></div>
            <dl className="result-info-list">
              <div><dt>Dataset</dt><dd>{model?.dataset.name ?? "Diabetes 130-US Hospitals"}</dd></div>
              <div><dt>Prediction target</dt><dd>{predictionTarget}</dd></div>
              <div><dt>Training inputs</dt><dd>{model?.features.length ?? "—"} dataset features</dd></div>
            </dl>
            <a className="text-link" href={model?.dataset.source ?? "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008"} target="_blank" rel="noreferrer">Dataset documentation <ArrowRight size={15} /></a>
          </article>

          <article className="result-info-card evaluation-card">
            <div className="side-card-heading"><Gauge size={17} /><h2>Evaluation context</h2></div>
            <div className="evaluation-metric"><span>Grouped CV PR-AUC</span><strong>{cvMetrics ? `${cvMetrics.mean.toFixed(3)} ± ${cvMetrics.std.toFixed(3)}` : "—"}</strong><small>{cvMetrics?.folds ?? "—"} folds · training search</small></div>
            {holdoutMetrics && <div className="evaluation-metric"><span>Holdout recall</span><strong>{(holdoutMetrics.recall * 100).toFixed(1)}%</strong><small>{holdoutMetrics.partition_note}</small></div>}
            <p className="evaluation-caveat">These metrics describe this project’s dataset and evaluation setup. They do not establish clinical effectiveness.</p>
          </article>

          <div className="result-disclaimer-card">
            <strong>For education only</strong>
            <p>{disclaimer}</p>
          </div>
        </aside>
      </section>
    </div>
  );
}