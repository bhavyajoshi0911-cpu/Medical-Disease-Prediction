import { useEffect, useState } from "react";
import { ArrowDownRight, ArrowRight, ArrowUpRight, Database, FileHeart, Gauge, ShieldAlert } from "lucide-react";
import { Link } from "react-router-dom";
import { getDiabetesModel } from "../services/api";
import type { ModelInfo } from "../types";

export function OverviewPage() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getDiabetesModel().then(setModel).catch((reason: Error) => setError(reason.message));
  }, []);

  const cv = model?.metrics.tuning_cv_best_average_precision;
  const holdout = model?.metrics.post_selection_holdout;

  return (
    <div className="page-stack page-enter">
      <section className="page-heading">
        <div>
          <p className="eyebrow">MODEL WORKSPACE <span>·</span> PHASE 12</p>
          <h1>Readmission overview</h1>
          <p className="heading-subtitle">Explore a dataset-specific estimate for 30-day hospital readmission.</p>
        </div>
        <Link className="button button-primary" to="/assessment/diabetes">Start assessment <ArrowRight size={17} /></Link>
      </section>

      <section className="overview-hero" aria-labelledby="active-model-title">
        <div className="hero-copy">
          <div className="hero-kicker"><span className="hero-kicker-dot" /> ACTIVE MODULE</div>
          <h2 id="active-model-title">Diabetes patient<br className="desktop-break" /> readmission</h2>
          <p>Prediction target: readmission within 30 days of a hospital encounter. This model does not predict whether someone has diabetes.</p>
          <Link to="/assessment/diabetes" className="hero-link">Open assessment <ArrowRight size={16} /></Link>
        </div>
        <div className="hero-metric" aria-label="30-day hospital readmission target">
          <div className="hero-metric-top"><span>OUTCOME WINDOW</span><span className="hero-index">01 / 03</span></div>
          <div className="hero-number">30<span>days</span></div>
          <div className="hero-rule"><span /></div>
          <div className="hero-metric-bottom"><span>Hospital encounter</span><span>UCI · 1999—2008</span></div>
        </div>
      </section>

      <div className="notice-bar" role="note">
        <ShieldAlert size={18} />
        <p><strong>Educational use only.</strong> Model estimates are not diagnoses or medical advice. Consult a qualified healthcare professional for real medical decisions.</p>
      </div>

      {error && <div className="inline-error" role="alert">{error}</div>}

      <section className="metric-grid" aria-label="Model facts">
        <article className="metric-card">
          <div className="metric-icon metric-icon-mint"><Database size={18} /></div>
          <p className="metric-label">Training dataset</p>
          <strong className="metric-value metric-value-text">101,766 <small>encounters</small></strong>
          <span className="metric-foot">130 US hospitals · 1999–2008</span>
        </article>
        <article className="metric-card">
          <div className="metric-icon metric-icon-coral"><FileHeart size={18} /></div>
          <p className="metric-label">Model inputs</p>
          <strong className="metric-value">{model?.features.length ?? "—"}</strong>
          <span className="metric-foot">Dataset-configured features</span>
        </article>
        <article className="metric-card">
          <div className="metric-icon metric-icon-gold"><Gauge size={18} /></div>
          <p className="metric-label">Grouped CV PR-AUC</p>
          <strong className="metric-value">{cv ? cv.mean.toFixed(3) : "—"}<small>{cv ? `± ${cv.std.toFixed(3)}` : ""}</small></strong>
          <span className="metric-foot">{cv ? `${cv.folds}-fold training CV · tuning estimate` : "Loading model metadata"}</span>
        </article>
      </section>

      <section className="lower-grid">
        <article className="panel model-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">DEPLOYED DEMO CANDIDATE</p><h2>{model?.algorithm ?? "Loading model"}</h2></div>
            <span className={`availability ${model?.available ? "availability-ready" : "availability-pending"}`}><span />{model?.available ? "Available" : "Unavailable"}</span>
          </div>
          <div className="model-details">
            <div><span>Version</span><strong>{model?.model_version ?? "—"}</strong></div>
            <div><span>Target label</span><strong>readmitted_30d</strong></div>
            <div><span>Feature encoding</span><strong>{model?.transformed_feature_count.toLocaleString() ?? "—"} columns</strong></div>
          </div>
          <div className="metric-context">
            <div className="metric-context-row"><span><ArrowUpRight size={15} /> Holdout recall</span><strong>{holdout ? `${(holdout.recall * 100).toFixed(1)}%` : "—"}</strong></div>
            <div className="metric-context-row"><span><ArrowDownRight size={15} /> Holdout precision</span><strong>{holdout ? `${(holdout.precision * 100).toFixed(1)}%` : "—"}</strong></div>
            <p>The same holdout was inspected during baseline comparison. These tuned-model results are exploratory, not untouched external validation.</p>
          </div>
        </article>

        <article className="panel dataset-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">SOURCE</p><h2>UCI hospital encounters</h2></div>
            <Database size={19} className="panel-heading-icon" />
          </div>
          <p className="dataset-description">Records from hospitalized patients already diagnosed with diabetes. The target is a recorded readmission outcome, not a diabetes diagnosis.</p>
          <div className="dataset-target"><span>POSITIVE LABEL</span><code>&lt;30</code><span>Readmitted within 30 days</span></div>
          <div className="dataset-target dataset-target-muted"><span>NEGATIVE LABELS</span><code>NO · &gt;30</code><span>No readmission or after 30 days</span></div>
          <a className="text-link" href="https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008" target="_blank" rel="noreferrer">Dataset documentation <ArrowRight size={15} /></a>
        </article>
      </section>

      <section className="phase-strip">
        <div className="phase-strip-mark">MP</div>
        <div><strong>Model output ≠ clinical evidence</strong><span>Metrics and explanations describe this dataset and evaluation setup only.</span></div>
        <Link to="/assessment/diabetes" aria-label="Start a new diabetes readmission assessment"><ArrowRight size={19} /></Link>
      </section>
    </div>
  );
}