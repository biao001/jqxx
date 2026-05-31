export type Severity = 'none' | 'low' | 'medium' | 'high' | 'critical';

export interface Detection {
  id: string;
  type: string;
  code?: string; // 英文行为代码(phone_use/drinking/smoking…)，识别框用它做英文标签
  timestamp: string;
  confidence: number;
  severity: Severity;
  source: 'behavior' | 'fatigue' | string;
  bbox?: [number, number, number, number] | number[];
  vt?: number; // 客户端标注的发生时刻(秒)，用于列表点击跳转
}

export interface DrivingStats {
  score: number; // 综合评分
  status: string;
  behavior_score?: number; // 驾驶行为评分
  fatigue_score?: number; // 疲劳评分
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
  signals?: {
    perclos?: number;
    ear?: number;
    eyes_closed?: boolean;
    head_yaw?: number;
    head_pitch?: number;
    head_roll?: number;
    gaze_zone?: string;
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
  frame_detections?: Detection[]; // 当前帧检测(画 bbox 用，每类仅一框)
  current_behavior?: BehaviorSummary;
  current_fatigue?: FatigueSummary;
  driver?: { name: string | null; status: string; distance?: number; box?: number[] };
  driver_roi?: number[] | null; // 驾驶员区域(xyxy 像素)，行为/疲劳只算此区域
  driver_present?: boolean; // 是否检出驾驶员（实时模式恒 true）
  llm_analysis: string;
  report_url: string | null;
  client_capture_t?: number; // 客户端送检时刻(performance.now ms)，FIFO 配对得到，用于视频↔检测同步
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
