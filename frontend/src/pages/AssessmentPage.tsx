import { useEffect, useState, type FormEvent } from "react";
import { ArrowLeft, ArrowRight, CircleHelp, LoaderCircle, ShieldAlert } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { getDiabetesModel, predictDiabetes } from "../services/api";
import type { FeatureSchema, ModelInfo } from "../types";

const featureLabels: Record<string, { label: string; unit?: string; help?: string }> = {
  race: { label: "Race", help: "As recorded in the source dataset." },
  gender: { label: "Gender", help: "As recorded in the source dataset." },
  age: { label: "Age band", help: "Grouped in ten-year intervals in the source." },
  admission_type_id: { label: "Admission type" },
  discharge_disposition_id: { label: "Discharge disposition" },
  admission_source_id: { label: "Admission source" },
  time_in_hospital: { label: "Time in hospital", unit: "days" },
  payer_code: { label: "Payer code" },
  medical_specialty: { label: "Medical specialty" },
  num_lab_procedures: { label: "Lab procedures", unit: "count" },
  num_procedures: { label: "Procedures", unit: "count" },
  num_medications: { label: "Medications", unit: "count" },
  number_outpatient: { label: "Prior outpatient visits", unit: "count" },
  number_emergency: { label: "Prior emergency visits", unit: "count" },
  number_inpatient: { label: "Prior inpatient visits", unit: "count" },
  number_diagnoses: { label: "Diagnoses recorded", unit: "count" },
  diag_1: { label: "Primary diagnosis code" },
  diag_2: { label: "Secondary diagnosis code" },
  diag_3: { label: "Additional diagnosis code" },
  max_glu_serum: { label: "Max glucose serum result" },
  A1Cresult: { label: "HbA1c result" },
};

const medicationFeatures = new Set([
  "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
  "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
  "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
  "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
  "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
  "change", "diabetesMed",
]);

function fieldLabel(name: string) {
  return featureLabels[name]?.label ?? name.replaceAll("_", " ").replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function featureGroup(name: string, type: FeatureSchema["type"]) {
  if (medicationFeatures.has(name)) return "Medication activity";
  if (["diag_1", "diag_2", "diag_3", "max_glu_serum", "A1Cresult", "num_lab_procedures"].includes(name)) return "Diagnoses and test results";
  if (type === "integer" || type === "number" || name === "payer_code" || name === "medical_specialty") return "Encounter and utilization";
  return "Patient and admission context";
}

function groupFeatures(schema: FeatureSchema[]) {
  const groups = new Map<string, FeatureSchema[]>();
  for (const field of schema) {
    const group = featureGroup(field.name, field.type);
    groups.set(group, [...(groups.get(group) ?? []), field]);
  }
  return [...groups.entries()];
}

export function AssessmentPage() {
  const navigate = useNavigate();
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [modelError, setModelError] = useState("");

  useEffect(() => {
    getDiabetesModel().then((metadata) => {
      setModel(metadata);
      if (!metadata.available) setModelError("The diabetes model artifact is unavailable.");
    }).catch((reason: Error) => setModelError(reason.message));
  }, []);

  function updateValue(name: string, value: string) {
    setValues((current) => ({ ...current, [name]: value }));
    setError("");
  }

  async function submitAssessment(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!model?.available) {
      setError("The diabetes model is unavailable. Please try again later.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const prediction = await predictDiabetes(model.feature_schema, values);
      navigate(`/result/${crypto.randomUUID()}`, { state: { result: prediction, model } });
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
          <p className="eyebrow">DIABETES MODULE <span>·</span> ENCOUNTER ASSESSMENT</p>
          <h1>Model assessment</h1>
          <p className="heading-subtitle">Enter the encounter fields used by the UCI readmission model.</p>
        </div>
        <div className="form-progress"><span>{fieldsEntered}</span><span> / {model?.feature_schema.length ?? "—"} fields</span></div>
      </section>

      <div className="notice-bar assessment-notice" role="note">
        <ShieldAlert size={18} />
        <p><strong>Educational project only.</strong> The target is hospital readmission risk for an encounter involving a patient already diagnosed with diabetes. This is not a diabetes diagnosis or medical advice.</p>
      </div>

      {modelError && <div className="inline-error" role="alert">{modelError}</div>}

      <div className="assessment-layout">
        <form className="assessment-form" onSubmit={submitAssessment}>
          {model ? groupFeatures(model.feature_schema).map(([group, fields], groupIndex) => (
            <details className="feature-group" key={group} open={groupIndex === 0 || groupIndex === 1}>
              <summary>
                <span className="group-number">{String(groupIndex + 1).padStart(2, "0")}</span>
                <span className="group-title">{group}</span>
                <span className="group-count">{fields.length} fields</span>
              </summary>
              <div className="field-grid">
                {fields.map((field) => <FeatureField key={field.name} field={field} value={values[field.name] ?? ""} onChange={updateValue} />)}
              </div>
            </details>
          )) : <div className="loading-panel"><LoaderCircle className="spin" size={19} /> Loading model feature configuration</div>}

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
              <h2>30-day readmission</h2>
              <div className="target-row"><span className="target-mark target-positive">&lt;30</span><span>Readmitted within 30 days</span></div>
              <div className="target-row"><span className="target-mark">NO / &gt;30</span><span>No readmission or after 30 days</span></div>
            </div>
            <div className="aside-panel method-panel">
              <p className="eyebrow">MODEL</p>
              <strong>{model?.algorithm ?? "XGBoost"}</strong>
              <span>Version {model?.model_version ?? "—"}</span>
              <div className="method-divider" />
              <p>These features influenced the model output. Contributions are not causes and are not medical advice.</p>
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
            inputMode="numeric"
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

