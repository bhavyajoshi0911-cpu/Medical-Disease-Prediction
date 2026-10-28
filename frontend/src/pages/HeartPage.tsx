import { useEffect, useState, type FormEvent } from "react";
import { ArrowLeft, ArrowRight, CircleHelp, HeartPulse, LoaderCircle, ShieldAlert } from "lucide-react";
import { Link } from "react-router-dom";
import { getHeartModel, predictHeart } from "../services/api";
import type { FeatureSchema, ModelInfo } from "../types";

const featureLabels: Record<string, { label: string; unit?: string; help?: string }> = {
  age: { label: "Age", unit: "years" },
  sex: { label: "Sex" },
  chest_pain_type: { label: "Chest pain type" },
  resting_bp: { label: "Resting blood pressure", unit: "mmHg" },
  cholesterol: { label: "Cholesterol", unit: "mg/dL" },
  fasting_bs: { label: "Fasting blood sugar" },
  resting_ecg: { label: "Resting ECG" },
  max_hr: { label: "Max heart rate", unit: "bpm" },
  exercise_angina: { label: "Exercise-induced angina" },
  oldpeak: { label: "Oldpeak" },
  st_slope: { label: "ST slope" },
};

function fieldLabel(name: string) {
  return featureLabels[name]?.label ?? name.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function HeartPage() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [modelError, setModelError] = useState("");
  const [values, setValues] = useState<Record<string, string>>({});

  useEffect(() => {
    getHeartModel()
      .then((metadata) => {
        setModel(metadata);
        if (!metadata.available) {
          setModelError("The heart model artifact is not yet deployed in this build.");
        }
      })
      .catch((reason: Error) => setModelError(reason.message));
  }, []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!model?.available) {
      setError("The heart model is not available yet. This module is currently a planned placeholder.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      await predictHeart(model.feature_schema, values);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to generate the prediction right now.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-stack page-enter assessment-page">
      <section className="page-heading assessment-heading">
        <div>
          <Link to="/" className="back-link"><ArrowLeft size={15} /> Overview</Link>
          <p className="eyebrow">HEART MODULE <span>·</span> PLANNED ASSESSMENT</p>
          <h1>Heart disease assessment</h1>
          <p className="heading-subtitle">This module is currently a placeholder awaiting a heart-disease dataset and trained pipeline.</p>
        </div>
        <div className="form-progress"><span>{Object.values(values).filter(Boolean).length}</span><span> / {model?.feature_schema.length ?? "—"} fields</span></div>
      </section>

      <div className="notice-bar assessment-notice" role="note">
        <ShieldAlert size={18} />
        <p><strong>Educational project only.</strong> The heart module is not yet deployed as a live medical model; this page is prepared for the next dataset and pipeline rollout.</p>
      </div>

      {modelError && <div className="inline-error" role="alert">{modelError}</div>}

      <div className="assessment-layout">
        <form className="assessment-form" onSubmit={handleSubmit}>
          {model ? (
            <details className="feature-group" open>
              <summary>
                <span className="group-number">01</span>
                <span className="group-title">Clinical indicators</span>
                <span className="group-count">{model.feature_schema.length} fields</span>
              </summary>
              <div className="field-grid">
                {model.feature_schema.map((field) => <FeatureField key={field.name} field={field} value={values[field.name] ?? ""} onChange={(name, value) => { setValues((current) => ({ ...current, [name]: value })); setError(""); }} />)}
              </div>
            </details>
          ) : (
            <div className="loading-panel"><LoaderCircle className="spin" size={19} /> Loading heart model configuration</div>
          )}

          {error && <div className="inline-error" role="alert">{error}</div>}
          <div className="form-actions">
            <p>Live prediction will be enabled after the heart model artifact and schema are added.</p>
            <button className="button button-primary" type="submit" disabled={loading || !model?.available}>
              {loading ? <><LoaderCircle className="spin" size={17} /> Preparing</> : <>Generate estimate <ArrowRight size={17} /></>}
            </button>
          </div>
        </form>

        <aside className="assessment-aside">
          <div className="aside-sticky">
            <div className="aside-panel target-panel">
              <p className="eyebrow">PREDICTION TARGET</p>
              <h2>Cardiovascular risk</h2>
              <div className="target-row"><span className="target-mark target-positive">Positive</span><span>Likely heart disease presence</span></div>
              <div className="target-row"><span className="target-mark">Negative</span><span>Likely no heart disease</span></div>
            </div>
            <div className="aside-panel method-panel">
              <p className="eyebrow">MODEL</p>
              <strong>{model?.algorithm ?? "Heart disease classifier"}</strong>
              <span>Version {model?.model_version ?? "—"}</span>
              <div className="method-divider" />
              <p>Once the data and pipeline are in place, the same SHAP and result flow used elsewhere in the platform will be applied here.</p>
            </div>
            <p className="aside-footnote"><CircleHelp size={15} /> The current implementation is for dataset and interface preparation only.</p>
          </div>
        </aside>
      </div>
    </div>
  );
}

type FeatureFieldProps = {
  field: FeatureSchema;
  value: string;
  onChange: (name: string, value: string) => void;
};

function FeatureField({ field, value, onChange }: FeatureFieldProps) {
  const info = featureLabels[field.name];
  const required = field.required ?? true;
  const id = `feature-${field.name.replaceAll(/[^a-zA-Z0-9_-]/g, "-")}`;
  const helpId = `${id}-help`;

  return (
    <div className="field-control">
      <label htmlFor={id}>{fieldLabel(field.name)}{required && <span className="required-mark" aria-label="required">*</span>}</label>
      {field.type === "integer" || field.type === "number" ? (
        <div className="numeric-input-wrap">
          <input
            id={id}
            type="number"
            inputMode="decimal"
            step={field.type === "integer" ? 1 : "any"}
            min={field.minimum}
            max={field.maximum}
            value={value}
            onChange={(event) => onChange(field.name, event.target.value)}
            required={required}
            aria-describedby={info?.help || info?.unit ? helpId : undefined}
          />
          {info?.unit && <span>{info.unit}</span>}
        </div>
      ) : (
        <select
          id={id}
          value={value}
          onChange={(event) => onChange(field.name, event.target.value)}
          required={required}
          aria-describedby={info?.help ? helpId : undefined}
        >
          <option value="">Choose a value</option>
          {field.nullable && <option value="__missing__">Not recorded</option>}
          {(field.categories ?? []).map((category) => <option key={category} value={category}>{category}</option>)}
        </select>
      )}
      {(info?.help || field.minimum !== undefined || field.maximum !== undefined) && (
        <span id={helpId} className="field-help">
          {info?.help ?? `Allowed range: ${field.minimum ?? "any"}${field.maximum === undefined ? "+" : `–${field.maximum}`}${info?.unit ? ` ${info.unit}` : ""}.`}
        </span>
      )}
    </div>
  );
}
