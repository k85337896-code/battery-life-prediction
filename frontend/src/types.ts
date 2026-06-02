export type BatteryType = string;

export interface CurvePoint {
  cycle: number;
  specific_capacity: number;
  soh: number;
}

export interface DatasetItem {
  id: number;
  battery_type: BatteryType;
  theoretical_capacity: number;
  rated_capacity: number;
  c_rate: number;
  cycle_life: number;
  current_soh: number;
  capacity_curve: CurvePoint[];
  source: string;
  note: string;
  chemistry?: string;
  dataset_name?: string;
  cell_name?: string;
  label_status?: string;
  training_eligible?: number;
  quality_flags?: string[];
  capacity_baseline?: number;
  additional_features?: Record<string, unknown>;
  created_at: string;
}

export interface PredictionResult {
  predicted_cycle_life: number;
  predicted_remaining_life: number;
  prediction_uncertainty_cycles?: number;
  predicted_life_lower?: number;
  predicted_life_upper?: number;
  soh_at_prediction: number;
  correlation_score: number;
  matched_dataset: DatasetItem;
  top_matches: Array<DatasetItem & { correlation_score: number }>;
  input_curve: CurvePoint[];
  predicted_curve: CurvePoint[];
  selected_model_key?: string;
  selected_model_name?: string;
  model_predicted_life?: number | null;
  xgb_predicted_life?: number | null;
}

export interface HistoryItem {
  id: number;
  predict_time: string;
  battery_type: BatteryType;
  rated_capacity: number;
  predicted_remaining_life: number;
  soh_at_prediction: number;
  matched_dataset_id: number;
  correlation_score: number;
  input_curve: CurvePoint[];
  predicted_curve: CurvePoint[];
}
