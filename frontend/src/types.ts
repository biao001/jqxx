export type Severity = 'none' | 'low' | 'medium' | 'high' | 'critical';

export interface Detection {
  id: string;
  type: string;
  timestamp: string;
  confidence: number;
  severity: Severity;
  source: 'behavior' | 'fatigue' | string;
  bbox?: [number, number, number, number] | number[];
}

export interface DrivingStats {
  score: number;
  status: string;
  focus: number;
  reaction: number;
  compliance: number;
  fatigue: number;
  stability: number;
}

export interface Capability {
  name?: string;
  mode?: 'model' | 'fallback' | 'partial' | 'not_loaded' | string;
  error?: string | null;
  configured?: boolean;
  provider?: string;
  model?: string;
  predictor?: string;
  feature_extractor?: string;
}

export interface BehaviorSummary {
  algorithm_label?: string;
  label: string;
  confidence: number;
  severity: Severity;
  bbox?: [number, number, number, number] | number[] | null;
  recommendation?: string;
}

export interface FatigueSummary {
  algorithm_label?: string;
  label: string;
  confidence: number;
  risk_level: 'low' | 'medium' | 'high' | string;
  indicators?: {
    yawn_score?: number;
    look_away_score?: number;
    fatigue_score?: number;
  };
}

export interface AnalysisResult {
  job_id: string | null;
  source: {
    kind: 'upload' | 'camera' | string;
    name: string;
  };
  frame_id: number | null;
  timestamp: number;
  stats: DrivingStats;
  detections: Detection[];
  current_behavior?: BehaviorSummary;
  current_fatigue?: FatigueSummary;
  llm_analysis: string;
  report_url: string | null;
  capabilities: {
    behavior?: Capability;
    fatigue?: Capability;
    llm?: Capability;
  };
  metrics: {
    latency_ms?: number;
    behavior_risk?: number;
    fatigue_risk?: number;
    frames_total?: number;
    frames_processed?: number;
    fps?: number;
  };
  error?: string;
}
