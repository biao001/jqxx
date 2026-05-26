import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { buildDriverAlertItems, parseAlertTimingConfig, pickSpeech, shouldSpeakAlert } from './useDriverAlerts';
import { AnalysisResult } from '../types';

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
    llm_analysis: '',
    report_url: null,
    capabilities: {},
    metrics: {},
    ...overrides,
  };
}

describe('driver alert speech', () => {
  it('picks conversational copy for phone usage from the matching label bank', () => {
    const speech = pickSpeech('驾驶中使用手机', false, () => 0);

    assert.equal(speech, '手机先放一放，路比消息急。');
  });

  it('uses urgent copy when a high-risk phone label requests urgent speech', () => {
    const speech = pickSpeech('驾驶中使用手机', true, () => 0.99);

    assert.equal(speech, '手机放下，注意前方。');
  });

  it('includes medium-risk yawning fatigue as an alert item', () => {
    const items = buildDriverAlertItems(result({
      current_fatigue: { label: '打哈欠', confidence: 0.8, risk_level: 'medium' },
    }), () => 0);

    assert.deepEqual(items, [
      {
        key: 'f:打哈欠',
        label: '打哈欠',
        level: 'warning',
        speech: '困了吧？找个地方歇会儿。',
      },
    ]);
  });

  it('uses urgent fatigue copy for high-risk fatigue driving', () => {
    const items = buildDriverAlertItems(result({
      current_fatigue: { label: '疲劳驾驶', confidence: 0.9, risk_level: 'high' },
    }), () => 0);

    assert.equal(items[0]?.speech, '危险了，请尽快靠边休息。');
  });

  it('does not include normal or low-risk fatigue labels', () => {
    const items = buildDriverAlertItems(result({
      current_fatigue: { label: '正常', confidence: 1, risk_level: 'low' },
    }), () => 0);

    assert.deepEqual(items, []);
  });

  it('blocks repeated speech during the same alert cooldown window', () => {
    const firstSeen = { 'b:驾驶中使用手机': 0 };
    const lastSpoken = { 'b:驾驶中使用手机': 1000 };

    assert.equal(shouldSpeakAlert('b:驾驶中使用手机', 7000, firstSeen, lastSpoken), false);
    assert.equal(shouldSpeakAlert('b:驾驶中使用手机', 8000, firstSeen, lastSpoken), true);
  });

  it('uses default timing when alert env values are missing or invalid', () => {
    assert.deepEqual(parseAlertTimingConfig({}), { durationMs: 700, cooldownMs: 7000 });
    assert.deepEqual(parseAlertTimingConfig({
      VITE_DRIVER_ALERT_DURATION_MS: '0',
      VITE_DRIVER_ALERT_COOLDOWN_MS: 'abc',
    }), { durationMs: 700, cooldownMs: 7000 });
  });

  it('parses alert timing from Vite env values', () => {
    assert.deepEqual(parseAlertTimingConfig({
      VITE_DRIVER_ALERT_DURATION_MS: '1200',
      VITE_DRIVER_ALERT_COOLDOWN_MS: '9000',
    }), { durationMs: 1200, cooldownMs: 9000 });
  });
});
