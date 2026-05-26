import { useEffect, useRef, useState } from 'react';
import { AnalysisResult } from '../types';

export type AlertLevel = 'none' | 'warning' | 'critical';

export interface DriverAlert {
  level: AlertLevel;
  text: string;
}

interface SpeechBankItem {
  normal: string[];
  urgent?: string[];
}

interface AlertItem {
  key: string;
  label: string;
  level: AlertLevel;
  speech: string;
}

// 现有检测标签 → 口语化语音播报词
const SPEECH_BANK: Record<string, SpeechBankItem> = {
  驾驶中使用手机: {
    normal: [
      '手机先放一放，路比消息急。',
      '别刷了，前面可不会暂停。',
      '朋友圈等会儿看，先把车开稳。',
    ],
    urgent: [
      '别看手机了，马上看路。',
      '手机放下，注意前方。',
    ],
  },
  驾驶中打电话: {
    normal: [
      '电话先别聊了，开车要紧。',
      '这通电话先欠着，安全不能欠。',
      '别边开边聊，先看路。',
    ],
    urgent: [
      '请放下电话，注意前方。',
      '别打电话了，先专心开车。',
    ],
  },
  驾驶中吸烟: {
    normal: [
      '烟先放一放，路还得看着呢。',
      '先别抽了，开车不是点烟时间。',
      '烟等会儿再抽，先稳住方向盘。',
    ],
  },
  驾驶中饮水: {
    normal: [
      '水先放一下，开稳了再喝。',
      '口渴能忍一下，方向盘不能没人管。',
      '先别喝，注意前面。',
    ],
  },
  驾驶中进食: {
    normal: [
      '先别吃了，前面这段路更需要你。',
      '嘴巴先休息，眼睛看前方。',
      '吃的等会儿再说，先把车开稳。',
    ],
  },
  未系安全带: {
    normal: [
      '安全带还没系，先把它扣上。',
      '安全带安排一下，别让它闲着。',
      '先系安全带，这个真不能省。',
    ],
    urgent: [
      '请立即系好安全带。',
    ],
  },
  双手离开方向盘: {
    normal: [
      '手回来，方向盘需要你。',
      '别撒手，车还得你管。',
      '双手握好，别让方向盘自己发挥。',
    ],
    urgent: [
      '双手握住方向盘，注意安全。',
    ],
  },
  驾驶位无人: {
    normal: [
      '驾驶位没人，这车可不能自己开。',
      '没看到驾驶员，请确认人在不在。',
      '驾驶位异常，先检查一下。',
    ],
    urgent: [
      '驾驶位异常，请立即检查。',
    ],
  },
  摄像头被遮挡: {
    normal: [
      '摄像头被挡住了，我看不见你了。',
      '镜头被遮住了，帮我露个脸。',
      '摄像头视线受阻，检查一下镜头。',
    ],
  },
  打哈欠: {
    normal: [
      '困了吧？找个地方歇会儿。',
      '这个哈欠有点明显，别硬撑。',
      '精神快掉线了，建议休息一下。',
    ],
  },
  视线偏离: {
    normal: [
      '抬头抬头，路在前面呢。',
      '别到处看了，先盯住前方。',
      '眼神收一收，前面更重要。',
    ],
    urgent: [
      '请马上看向前方。',
    ],
  },
  疲劳驾驶: {
    normal: [
      '状态有点累了，建议靠边休息。',
      '别硬撑了，找个安全地方歇一下。',
      '危险提醒，注意清醒，必要时停车休息。',
    ],
    urgent: [
      '危险了，请尽快靠边休息。',
      '请保持清醒，必要时立即停车休息。',
    ],
  },
};

interface AlertTimingConfig {
  durationMs: number;
  cooldownMs: number;
}

interface AlertTimingEnv {
  VITE_DRIVER_ALERT_DURATION_MS?: string;
  VITE_DRIVER_ALERT_COOLDOWN_MS?: string;
}

const DEFAULT_ALERT_TIMING: AlertTimingConfig = {
  // 行为需持续这么久(毫秒)才播报，避免一闪而过的误检触发语音
  durationMs: 700,
  // 同一类提示的冷却时间，避免连环轰炸
  cooldownMs: 7000,
};

const ORDER: Record<AlertLevel, number> = { none: 0, warning: 1, critical: 2 };

function positiveMs(value: string | undefined, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export function parseAlertTimingConfig(env: AlertTimingEnv = {}): AlertTimingConfig {
  return {
    durationMs: positiveMs(env.VITE_DRIVER_ALERT_DURATION_MS, DEFAULT_ALERT_TIMING.durationMs),
    cooldownMs: positiveMs(env.VITE_DRIVER_ALERT_COOLDOWN_MS, DEFAULT_ALERT_TIMING.cooldownMs),
  };
}

const ALERT_TIMING = parseAlertTimingConfig((import.meta as ImportMeta & { env?: AlertTimingEnv }).env);

function choose(items: string[], random: () => number) {
  const index = Math.min(items.length - 1, Math.floor(random() * items.length));
  return items[index];
}

export function pickSpeech(label: string, urgent: boolean, random: () => number = Math.random) {
  const bank = SPEECH_BANK[label];
  if (!bank) return `注意，${label}`;
  const items = urgent && bank.urgent?.length ? bank.urgent : bank.normal;
  return choose(items, random);
}

function isUrgentSeverity(severity: string | undefined) {
  return severity === 'high' || severity === 'critical';
}

function isUrgentFatigue(riskLevel: string | undefined) {
  return riskLevel === 'high' || riskLevel === 'critical';
}

export function buildDriverAlertItems(result: AnalysisResult, random: () => number = Math.random): AlertItem[] {
  const active: AlertItem[] = [];
  const beh = result.current_behavior;
  if (beh && beh.severity !== 'none' && beh.label && beh.label !== '未检测到行为风险') {
    const level: AlertLevel = beh.severity === 'critical' ? 'critical' : 'warning';
    active.push({
      key: `b:${beh.label}`,
      label: beh.label,
      level,
      speech: pickSpeech(beh.label, isUrgentSeverity(beh.severity), random),
    });
  }

  const fat = result.current_fatigue;
  if (fat && fat.risk_level !== 'low' && fat.label && fat.label !== '正常') {
    active.push({
      key: `f:${fat.label}`,
      label: fat.label,
      level: 'warning',
      speech: pickSpeech(fat.label, isUrgentFatigue(fat.risk_level), random),
    });
  }

  return active;
}

export function shouldSpeakAlert(
  key: string,
  now: number,
  firstSeen: Record<string, number>,
  lastSpoken: Record<string, number>,
  timing: AlertTimingConfig = DEFAULT_ALERT_TIMING,
) {
  const persisted = now - (firstSeen[key] ?? now);
  const sinceSpoken = now - (lastSpoken[key] ?? 0);
  return persisted >= timing.durationMs && sinceSpoken >= timing.cooldownMs;
}

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
 *   - 持续时长触发(行为稳定存在 VITE_DRIVER_ALERT_DURATION_MS 才报)
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

    // 收集本帧可告警项：风险行为 + 非低风险疲劳
    const active = buildDriverAlertItems(result);

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
        if (shouldSpeakAlert(a.key, now, firstSeenRef.current, lastSpokenRef.current, ALERT_TIMING)) {
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
