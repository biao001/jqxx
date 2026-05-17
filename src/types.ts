/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export enum DetectionType {
  CELL_PHONE = '玩手机',
  DISTRACTION = '视线偏离',
  YAWN = '疑似疲劳 (打哈欠)',
  SMOKING = '抽烟',
}

export interface Detection {
  id: string;
  type: DetectionType;
  timestamp: string;
  confidence: number;
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
