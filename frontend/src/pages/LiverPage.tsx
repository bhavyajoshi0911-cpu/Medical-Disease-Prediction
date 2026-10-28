import { useEffect, useState, type FormEvent } from "react";
import { ArrowLeft, ArrowRight, CircleHelp, LoaderCircle, ShieldAlert } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { getLiverModel, predictLiver } from "../services/api";
import type { FeatureSchema, ModelInfo } from "../types";

const featureLabels: Record<string, { label: string; unit?: string; help?: string }> = {
  Age: { label: "Age", unit: "years" },
  Gender: { label: "Gender", help: "Biological sex label as recorded in the source dataset." },
  Total_Bilirubin: { label: "Total bilirubin", unit: "mg/dL" },
  Direct_Bilirubin: { label: "Direct bilirubin", unit: "mg/dL" },
  Alkaline_Phosphotase: { label: "Alkaline phosphatase", unit: "U/L" },
  Alamine_Aminotransferase: { label: "ALT", unit: "U/L" },
  Aspartate_Aminotransferase: { label: "AST", unit: "U/L" },
  Total_Proteins: { label: "Total protein", unit: "g/dL" },
  Albumin: { label: "Albumin", unit: "g/dL" },
  Albumin_and_Globulin_Ratio: { label: "Albumin/globulin ratio" },
};

function fieldLabel(name: string) {
  return featureLabels[name]?.label ?? name.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function groupFeatures(schema: FeatureSchema[]) {
  const groups = new Map<string, FeatureSchema[]>();
  const fields = schema.slice();
  for (const field of fields) {
    const name = field.name.toLowerCase();
    const group = name === "gender" ? "Patient details" : "Clinical laboratory values";
    groups.set(group, [...(groups.get(group) ?? []), field]);
  }
  return [...groups.entries()];
}

export function LiverPage() {
  const navigate = useNavigate();
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [modelError, setModelError] = useState("");

  useEffect(() => {
    getLiverModel()
      .then((metadata) => {
        setModel(metadata);
        if (!metadata.available) {
          setModelError("The liver model artifact is not available yet in this build.");
        }
      })
      .catch((reason: Error) => setModelError(reason.message));
  }, []);

  function updateValue(name: string, value: string) {
    setValues((current) => ({ ...current, [name]: value }));
    setError("");
  }

  async function submitAssessment(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!model?.available) {
      setError("The liver model is unavailable. Please try again later.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const prediction = await predictLiver(model.feature_schema, values);
      const resultId = typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : String(Date.now());
      navigate(`/result/${resultId}`, { state: { result: prediction, model } });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to generate the prediction right now. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  const fieldsEntered = model?.feature_schema.filter((field) => values[field.name] !== undefined && values[field.name] !== "").length ?? 0;

  return (
    <div className="page-stack page-enter assessment-page">
      <section className="page-heading assessment-heading">
        <div>
          <Link to="/" className="back-link"><ArrowLeft size={15} /> Overview</Link>
          <p className="eyebrow">LIVER MODULE <span>·</span> CLINICAL ASSESSMENT</p>
          <h1>Liver disease assessment</h1>
          <p className="heading-subtitle">Enter the clinical and laboratory values used by the ILPD-based liver risk model.</p>
        </div>
        <div className="form-progress"><span>{fieldsEntered}</span><span> / {model?.feature_schema.length ?? "—"} fields</span></div>
      </section>

      <div className="notice-bar assessment-notice" role="note">
        <ShieldAlert size={18} />
        <p><strong>Educational project only.</strong> This model is a dataset-specific educational demo and not a clinical diagnosis or medical advice.</p>
      </div>

      {modelError && <div className="inline-error" role="alert">{modelError}</div>}

      <div className="assessment-layout">
        <form className="assessment-form" onSubmit={submitAssessment}>
          {model ? groupFeatures(model.feature_schema).map(([group, fields], groupIndex) => (
            <details className="feature-group" key={group} open={groupIndex === 0}>
              <summary>
                <span className="group-number">{String(groupIndex + 1).padStart(2, "0")}</span>
                <span className="group-title">{group}</span>
                <span className="group-count">{fields.length} fields</span>
              </summary>
              <div className="field-grid">
                {fields.map((field) => (
                  <FeatureField key={field.name} field={field} value={values[field.name] ?? ""} onChange={updateValue} />
                ))}
              </div>
            </details>
          )) : <div className="loading-panel"><LoaderCircle className="spin" size={19} /> Loading liver model feature configuration</div>}

          {error && <div className="inline-error" role="alert">{error}</div>}
          <div className="form-actions">
            <p>Inputs are used for this request only and are not saved to prediction history.</p>
            <button className="button button-primary" type="submit" disabled={loading || !model?.available}>
              {loading ? <><LoaderCircle className="spin" size={17} /> Estimating</> : <>Generate estimate <ArrowRight size={17} /></>}
            </button>
          </div>
        </form>

        <aside className="assessment-aside">
          <div className="aside-sticky">
            <div className="aside-panel target-panel">
              <p className="eyebrow">PREDICTION TARGET</p>
              <h2>Liver disease presence</h2>
              <div className="target-row"><span className="target-mark target-positive">Positive</span><span>Patient likely has liver disease</span></div>
              <div className="target-row"><span className="target-mark">Negative</span><span>Patient likely does not have liver disease</span></div>
            </div>
            <div className="aside-panel method-panel">
              <p className="eyebrow">MODEL</p>
              <strong>{model?.algorithm ?? "Liver classifier"}</strong>
              <span>Version {model?.model_version ?? "—"}</span>
              <div className="method-divider" />
              <p>These features influence the model output. Contributions are explanatory and not causal.</p>
            </div>
            <p className="aside-footnote"><CircleHelp size={15} /> Only provide data you are authorized to use.</p>
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
