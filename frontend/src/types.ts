// Mirrors backend/app/models/schemas.py. Keep the two in sync.

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Keypoint {
  name: string;
  x: number;
  y: number;
  confidence: number;
  visible: boolean;
}

export interface PoseData {
  keypoints: Keypoint[];
  mean_confidence: number;
  visible_keypoints: number;
}

export interface FaceData {
  detected: boolean;
  box: BoundingBox | null;
  confidence: number;
  source: string;
}

/**
 * Probabilities describe the visible facial expression in a single crop.
 * They are not a reading of anyone's mental or emotional state.
 */
export interface ExpressionResult {
  available: boolean;
  probabilities: Record<string, number>;
  top_label: string | null;
  top_probability: number;
  model_name: string | null;
  inference_ms: number;
}

export type ActivityLevel = 'low' | 'moderate' | 'high' | 'unknown';
export type Posture = 'standing' | 'sitting_or_crouching' | 'lying' | 'unknown';

export interface BehaviorFeatures {
  movement_speed: number;
  movement_amount: number;
  activity_level: ActivityLevel;
  posture: Posture;
  posture_confidence: number;
  pose_change: number;
  head_orientation: string | null;
  head_yaw_ratio: number | null;
  torso_lean_degrees: number | null;
  time_visible_seconds: number;
}

export interface PersonAnalysis {
  track_id: number;
  box: BoundingBox;
  confidence: number;
  center: [number, number];
  pose: PoseData;
  face: FaceData;
  expression: ExpressionResult;
  behavior: BehaviorFeatures;
  first_seen: number;
  last_seen: number;
}

export type StageStatus = 'ok' | 'degraded' | 'disabled' | 'error';

export interface PipelineStage {
  name: string;
  status: StageStatus;
  detail: string;
  last_duration_ms: number;
}

export interface FrameAnalysis {
  type: 'analysis';
  frame_id: number;
  timestamp: number;
  frame_width: number;
  frame_height: number;
  people_count: number;
  people: PersonAnalysis[];
  fps: number;
  capture_fps: number;
  processing_ms: number;
  camera_connected: boolean;
  source: string;
  stages: PipelineStage[];
}

export interface ModelInfo {
  key: string;
  name: string;
  path: string;
  present: boolean;
  loaded: boolean;
  size_bytes: number;
  source_url: string;
  license: string;
  description: string;
  error: string;
}

export interface HelloMessage {
  type: 'hello';
  source: string;
  device: string;
  ws_fps: number;
  stream_url: string;
  models: ModelInfo[];
  stages: PipelineStage[];
  notice: string;
}

export interface StatusMessage {
  type: 'status';
  timestamp: number;
  camera_connected: boolean;
  source: string;
  last_error: string;
  fps: number;
  stages: PipelineStage[];
}

export type LiveMessage = FrameAnalysis | HelloMessage | StatusMessage;

export interface SourceInfo {
  kind: string;
  target: string;
  connected: boolean;
  width: number;
  height: number;
  capture_fps: number;
  requested_fps: number;
  last_error: string;
}

export type ConnectionState = 'connecting' | 'live' | 'reconnecting' | 'offline';
