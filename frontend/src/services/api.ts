import type { FeatureSchema, HealthStatus, ModelInfo, PredictionHistoryItem, PredictionHistoryResponse, PredictionResult } from "../types";

const apiBase = import.meta.env.VITE_API_BASE_URL ?? "";

async function readJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${apiBase}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new Error("Unable to connect to MedPredict. Check that the API is running.");
  }

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(typeof body.detail === "string" ? body.detail : "The request could not be completed.");
  }
  return body as T;
}

export function getHealth(): Promise<HealthStatus> {
  return readJson<HealthStatus>("/api/health");
}

export function getDiabetesModel(): Promise<ModelInfo> {
  return readJson<ModelInfo>("/api/models/diabetes");
}

export function getHeartModel(): Promise<ModelInfo> {
  return readJson<ModelInfo>("/api/models/heart");
}

export function getLiverModel(): Promise<ModelInfo> {
  return readJson<ModelInfo>("/api/models/liver");
}

export function predictDiabetes(
  schema: FeatureSchema[],
  values: Record<string, string>,
): Promise<PredictionResult> {
  const payload: Record<string, number | string | null> = {};
  for (const field of schema) {
    const value = values[field.name] ?? "";
    if (field.type === "integer" || field.type === "number") {
      payload[field.name] = Number(value);
    } else if (value === "__missing__") {
      payload[field.name] = null;
    } else {
      payload[field.name] = value === "" && field.nullable ? null : value;
    }
  }
  return readJson<PredictionResult>("/api/predict/diabetes", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function predictLiver(
  schema: FeatureSchema[],
  values: Record<string, string>,
): Promise<PredictionResult> {
  const payload: Record<string, number | string | null> = {};
  for (const field of schema) {
    const value = values[field.name] ?? "";
    if (field.type === "integer" || field.type === "number") {
      payload[field.name] = Number(value);
    } else if (value === "__missing__") {
      payload[field.name] = null;
    } else {
      payload[field.name] = value === "" && field.nullable ? null : value;
    }
  }
  return readJson<PredictionResult>("/api/predict/liver", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function predictHeart(
  schema: FeatureSchema[],
  values: Record<string, string>,
): Promise<PredictionResult> {
  const payload: Record<string, number | string | null> = {};
  for (const field of schema) {
    const value = values[field.name] ?? "";
    if (field.type === "integer" || field.type === "number") {
      payload[field.name] = Number(value);
    } else if (value === "__missing__") {
      payload[field.name] = null;
    } else {
      payload[field.name] = value === "" && field.nullable ? null : value;
    }
  }
  return readJson<PredictionResult>("/api/predict/heart", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getPredictionHistory(limit = 50, offset = 0): Promise<PredictionHistoryResponse> {
  return readJson<PredictionHistoryResponse>(`/api/history?limit=${limit}&offset=${offset}`);
}

export function getPredictionHistoryItem(id: number): Promise<PredictionHistoryItem> {
  return readJson<PredictionHistoryItem>(`/api/history/${id}`);
}

export async function deletePredictionHistoryItem(id: number): Promise<void> {
  const response = await fetch(`${apiBase}/api/history/${id}`, { method: "DELETE" });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : "Unable to delete this history item.");
  }
}