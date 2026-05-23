import { useEffect, useRef, useState } from 'react';
import { AnalysisResult } from '../types';

export type AlertLevel = 'none' | 'warning' | 'critical';

export interface DriverAlert {
  level: AlertLevel;
  text: string;
}

// 行为中文标签 → 语音播报词
const SPEECH: Record<string, string> = {
  驾驶中使用手机: '请勿使用手机，专注驾驶',
  驾驶中打电话: '请勿接打电话',
  驾驶中吸烟: '请勿吸烟',
  驾驶中饮水: '请勿饮水，注意前方',
  驾驶中进食: '请勿进食，注意前方',
  未系安全带: '请系好安全带',
  双手离开方向盘: '请双手握住方向盘',
  驾驶位无人: '未检测到驾驶员',
  摄像头被遮挡: '摄像头被遮挡，请检查',
};
const FATIGUE_SPEECH = '检测到疲劳迹象，请注意休息';

// 行为需持续这么久(毫秒)才播报，避免一闪而过的误检触发语音
const DURATION_MS = 700;
// 同一类提示的冷却时间，避免连环轰炸
const COOLDOWN_MS = 7000;

const ORDER: Record<AlertLevel, number> = { none: 0, warning: 1, critical: 2 };

function speak(text: string, urgent: boolean) {
  const synth = window.speechSynthesis;
  if (!synth) return;
  synth.cancel(); // 清空排队，避免语音堆积延迟
  const u = new SpeechSynthesisUtterance(text);
  u.lang = 'zh-CN';
  u.rate = urgent ? 1.15 : 1.0;
  u.pitch = urgent ? 1.1 : 1.0;
  u.volume = 1;
  synth.speak(u);
}

/**
 * 驾驶告警 hook：根据逐帧结果做
 *   - 持续时长触发(行为稳定存在 DURATION_MS 才报)
 *   - 分级语音播报(critical 语速更快) + 每类冷却
 *   - 返回当前视觉告警等级，供横幅/全屏脉冲使用
 */
export function useDriverAlerts(result: AnalysisResult | null, enabled: boolean): DriverAlert {
  const [alert, setAlert] = useState<DriverAlert>({ level: 'none', text: '' });
  const firstSeenRef = useRef<Record<string, number>>({});
  const lastSpokenRef = useRef<Record<string, number>>({});

  useEffect(() => {
    if (!result) return;
    const now = performance.now();

    // 收集本帧可告警项：风险行为 + 高疲劳
    const active: { key: string; label: string; level: AlertLevel; speech: string }[] = [];
    const beh = result.current_behavior;
    if (beh && beh.severity !== 'none' && beh.label && beh.label !== '未检测到行为风险') {
      const level: AlertLevel = beh.severity === 'critical' ? 'critical' : 'warning';
      active.push({ key: `b:${beh.label}`, label: beh.label, level, speech: SPEECH[beh.label] || `注意，${beh.label}` });
    }
    const fat = result.current_fatigue;
    if (fat && fat.risk_level === 'high') {
      active.push({ key: 'f:fatigue', label: fat.label || '疲劳', level: 'warning', speech: FATIGUE_SPEECH });
    }

    // 清理已消失项的计时
    const keys = new Set(active.map((a) => a.key));
    for (const k of Object.keys(firstSeenRef.current)) {
      if (!keys.has(k)) delete firstSeenRef.current[k];
    }
    for (const a of active) {
      if (!firstSeenRef.current[a.key]) firstSeenRef.current[a.key] = now;
    }

    // 视觉告警取最高级别
    let level: AlertLevel = 'none';
    let text = '';
    for (const a of active) {
      if (ORDER[a.level] >= ORDER[level]) {
        level = a.level;
        text = a.label;
      }
    }
    setAlert((prev) => (prev.level === level && prev.text === text ? prev : { level, text }));

    // 语音：持续达标 + 过了冷却
    if (enabled) {
      for (const a of active) {
        const persisted = now - (firstSeenRef.current[a.key] ?? now);
        const sinceSpoken = now - (lastSpokenRef.current[a.key] ?? 0);
        if (persisted >= DURATION_MS && sinceSpoken >= COOLDOWN_MS) {
          speak(a.speech, a.level === 'critical');
          lastSpokenRef.current[a.key] = now;
        }
      }
    }
  }, [result, enabled]);

  // 关闭语音时立即停掉正在播报的
  useEffect(() => {
    if (!enabled) window.speechSynthesis?.cancel();
  }, [enabled]);

  return alert;
}
