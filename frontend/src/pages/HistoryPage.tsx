import { useCallback, useEffect, useState } from "react";
import { ArrowRight, ClipboardList, LoaderCircle, Trash2 } from "lucide-react";
import { Link } from "react-router-dom";
import { deletePredictionHistoryItem, getPredictionHistory } from "../services/api";
import type { PredictionHistoryItem } from "../types";

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function HistoryPage() {
  const [items, setItems] = useState<PredictionHistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [deleting, setDeleting] = useState<number | null>(null);

  const loadHistory = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const history = await getPredictionHistory();
      setItems(history.items);
      setTotal(history.total);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to load prediction history.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void loadHistory(); }, [loadHistory]);

  async function removeItem(id: number) {
    setDeleting(id);
    setError("");
    try {
      await deletePredictionHistoryItem(id);
      setItems((current) => current.filter((item) => item.id !== id));
      setTotal((current) => Math.max(0, current - 1));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to delete this history item.");
    } finally {
      setDeleting(null);
    }
  }

  return (
    <div className="page-stack page-enter history-page">
      <section className="page-heading">
        <div>
          <p className="eyebrow">ASSESSMENTS <span>·</span> PRIVATE TO THIS DEMO</p>
          <h1>Prediction history</h1>
          <p className="heading-subtitle">Saved outcome estimates and model versions. Submitted feature values are not stored.</p>
        </div>
        <Link className="button button-primary" to="/assessment/diabetes">New assessment <ArrowRight size={16} /></Link>
      </section>

      <div className="notice-bar" role="note">
        <ClipboardList size={18} />
        <p><strong>Metadata only.</strong> History stores the disease, target, probability, model, version, and timestamp—never the health inputs used for a request.</p>
      </div>

      {error && <div className="inline-error" role="alert">{error}</div>}

      <section className="history-panel" aria-label="Saved predictions">
        <div className="history-panel-heading">
          <div><span className="eyebrow">RECENT RECORDS</span><h2>{total} {total === 1 ? "assessment" : "assessments"}</h2></div>
          <button className="button button-quiet history-refresh" onClick={() => void loadHistory()} disabled={loading}>
            {loading ? <LoaderCircle className="spin" size={15} /> : <ClipboardList size={15} />} Refresh
          </button>
        </div>

        {loading ? (
          <div className="history-loading"><LoaderCircle className="spin" size={18} /> Loading saved assessments</div>
        ) : items.length === 0 ? (
          <div className="history-empty">
            <div className="history-empty-mark"><ClipboardList size={21} /></div>
            <h3>No saved predictions</h3>
            <p>Completed estimates will appear here. Feature inputs are not retained.</p>
            <Link className="text-link" to="/assessment/diabetes">Start an assessment <ArrowRight size={15} /></Link>
          </div>
        ) : (
          <>
            <div className="history-table-wrap">
              <table className="history-table">
                <thead><tr><th>Date</th><th>Disease</th><th>Prediction target</th><th>Probability</th><th>Model</th><th><span className="sr-only">Actions</span></th></tr></thead>
                <tbody>
                  {items.map((item) => (
                    <tr key={item.id}>
                      <td>{formatDate(item.created_at)}</td>
                      <td><span className="history-disease"><span />{item.disease}</span></td>
                      <td>{item.prediction_target}</td>
                      <td><strong className="history-probability">{(item.probability * 100).toFixed(1)}%</strong></td>
                      <td><span className="history-model">{item.model_name}</span><small>v{item.model_version}</small></td>
                      <td className="history-actions">
                        <Link className="icon-button" to={`/result/${item.id}`} aria-label={`View saved result ${item.id}`} title="View saved result"><ArrowRight size={17} /></Link>
                        <button className="icon-button delete-button" onClick={() => void removeItem(item.id)} disabled={deleting === item.id} aria-label={`Delete saved result ${item.id}`} title="Delete saved result">
                          {deleting === item.id ? <LoaderCircle className="spin" size={16} /> : <Trash2 size={16} />}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="history-footnote">Showing the latest {items.length} of {total}. No assessment features are available from saved history.</p>
          </>
        )}
      </section>
    </div>
  );
}