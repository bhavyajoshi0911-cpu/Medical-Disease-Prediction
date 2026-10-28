export type FeatureSchema = {
  name: string;
  type: "integer" | "number" | "category";
  required?: boolean;
  nullable?: boolean;
  minimum?: number;
  maximum?: number;
  categories?: string[];
};

export type ModelInfo = {
  disease: string;
  available: boolean;
  model_name: string;
  algorithm: string;
  model_version: string;
  training_date: string;
  dataset: { name: string; source: string; sha256: string };
  features: string[];
  feature_schema: FeatureSchema[];
  transformed_feature_count: number;
  target: string;
  metrics: {
    tuning_cv_best_average_precision?: {
      mean: number;
      std: number;
      folds: number;
      configurations_tested: number;
    };
    post_selection_holdout?: {
      accuracy: number;
      precision: number;
      recall: number;
      f1: number;
      roc_auc: number;
      pr_auc: number;
      partition_note: string;
    };
  };
  selection_rationale: string;
  clinical_validation: boolean;
  educational_use_only: boolean;
};

export type FeatureContribution = {
  feature: string;
  contribution: number;
  direction: string;
};

export type PredictionResult = {
  prediction_id: number;
  created_at: string;
  prediction: 0 | 1;
  disease: string;
  prediction_target: string;
  probability: number;
  model: string;
  model_version: string;
  top_features: FeatureContribution[];
  disclaimer: string;
};

export type HealthStatus = {
  status: "ok";
  app_name: string;
  database: "ok";
};

export type PredictionHistoryItem = {
  id: number;
  disease: string;
  prediction_target: string;
  probability: number;
  model_name: string;
  model_version: string;
  created_at: string;
};

export type PredictionHistoryResponse = {
  items: PredictionHistoryItem[];
  total: number;
  limit: number;
  offset: number;
};