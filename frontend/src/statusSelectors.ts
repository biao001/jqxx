import { AnalysisResult } from './types';

export interface DriverStatusState {
  present: boolean;
  seatbelt: boolean;
  hands: boolean;
  gaze: boolean;
  noPhone: boolean;
  noSmoke: boolean;
}

export function currentDetectionTypes(result: AnalysisResult | null): string[] {
  return (result?.frame_detections ?? []).map((d) => d.type);
}

export function driverStatusFromCurrentFrame(result: AnalysisResult | null): DriverStatusState | null {
  if (!result) return null;
  const types = currentDetectionTypes(result);
  const has = (kw: string) => types.some((t) => t.includes(kw));
  return {
    present: !has('无人'),
    seatbelt: !has('未系'),
    hands: !has('离开方向盘') && !has('离盘'),
    gaze: !has('视线'),
    noPhone: !has('手机') && !has('电话'),
    noSmoke: !has('吸烟'),
  };
}

export function activeBehaviorsFromCurrentFrame(result: AnalysisResult | null): string[] {
  const types = currentDetectionTypes(result);
  const fat = result?.current_fatigue;
  if (fat && fat.risk_level !== 'low' && fat.label) types.push(fat.label);
  return types;
}
