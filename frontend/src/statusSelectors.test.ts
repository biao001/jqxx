import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { activeBehaviorsFromCurrentFrame, driverStatusFromCurrentFrame } from './statusSelectors';
import { AnalysisResult } from './types';

function result(overrides: Partial<AnalysisResult>): AnalysisResult {
  return {
    job_id: null,
    source: { kind: 'camera', name: 'camera' },
    frame_id: 1,
    timestamp: 0,
    stats: {
      score: 100,
      status: '正常',
      focus: 100,
      reaction: 100,
      compliance: 100,
      fatigue: 100,
      stability: 100,
    },
    detections: [],
    frame_detections: [],
    llm_analysis: '',
    report_url: null,
    capabilities: {},
    metrics: {},
    ...overrides,
  };
}

describe('status selectors', () => {
  it('ignores historical detections when deriving current driver status', () => {
    const status = driverStatusFromCurrentFrame(result({
      detections: [
        {
          id: 'history-1',
          type: '驾驶中使用手机',
          timestamp: '00:01',
          confidence: 0.9,
          severity: 'high',
          source: 'behavior',
        },
      ],
      frame_detections: [],
    }));

    assert.equal(status?.noPhone, true);
  });

  it('uses current frame detections for active behavior tips', () => {
    const active = activeBehaviorsFromCurrentFrame(result({
      detections: [
        {
          id: 'history-1',
          type: '驾驶中吸烟',
          timestamp: '00:01',
          confidence: 0.9,
          severity: 'medium',
          source: 'behavior',
        },
      ],
      frame_detections: [],
      current_fatigue: { label: '正常', confidence: 1, risk_level: 'low' },
    }));

    assert.deepEqual(active, []);
  });
});
