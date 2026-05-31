import { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import type { AlertLevel } from '../lib/useDriverAlerts';

interface Props {
  level: AlertLevel; // 当前最高告警级别(none/warning/critical)
  active: boolean; // 仅分析中显示
}

const ESCALATE_MS = 5000; // 同一危险持续多久升到最强(0→1)

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

// 告警颜色：critical 恒红；warning 随持续时长由琥珀→红，呈「柔和→强烈」渐进
function glowRgb(level: AlertLevel, stage: number): [number, number, number] {
  if (level === 'critical') return [239, 68, 68];
  return [Math.round(lerp(245, 239, stage)), Math.round(lerp(158, 68, stage)), Math.round(lerp(11, 68, stage))];
}

/**
 * 画面边缘呼吸光：按风险等级让视频四周泛色脉冲(不挡画面，余光即可感知)，替代中间的大告警块。
 * 配合告警分级升级——同一危险持续越久，边缘光越厚越红、由柔和渐变强烈(图标→边缘光→语音的中间层)。
 *
 * 实现要点：呼吸(opacity)用 motion 做稳定循环、周期只随 level 变(避免每次升级重启动画)；
 * 升级的「变厚/变红」放在 boxShadow 上、用 CSS transition 平滑过渡，每 250ms 推进一次 stage。
 */
export default function EdgeGlow({ level, active }: Props) {
  const [stage, setStage] = useState(0);
  const startRef = useRef(0);

  // 升级计时只在 level 变化(none↔warning↔critical)时重置 —— 不随同级行为标签逐帧抖动
  // (如手机↔打电话切换)重置，否则持续分心也永远升不到强烈，escalation 形同虚设。
  useEffect(() => {
    if (!active || level === 'none') {
      setStage(0);
      return;
    }
    startRef.current = performance.now();
    setStage(0); // 进入/升级告警从柔和开始，5s 内渐强
    const id = window.setInterval(() => {
      setStage(Math.min(1, (performance.now() - startRef.current) / ESCALATE_MS));
    }, 250);
    return () => window.clearInterval(id);
  }, [active, level]);

  if (!active || level === 'none') return null;

  const isCritical = level === 'critical';
  const [r, g, b] = glowRgb(level, stage);
  const thick = (isCritical ? 6 : 3) + stage * 4; // 内描边厚度随升级增厚
  const blur = (isCritical ? 50 : 28) + stage * 45; // 内辉光扩散随升级加大
  const dur = isCritical ? 1.1 : 1.8; // 呼吸周期：critical 更急促(周期只随 level 变，避免重启)
  const loOp = isCritical ? 0.5 : 0.32;
  const hiOp = isCritical ? 1 : 0.7;

  return (
    <motion.div
      key={level} // level 变化时重建以套用新的呼吸周期
      className="absolute inset-0 pointer-events-none z-30"
      initial={{ opacity: loOp }}
      animate={{ opacity: [loOp, hiOp, loOp] }}
      transition={{ duration: dur, repeat: Infinity, ease: 'easeInOut' }}
      style={{
        boxShadow: `inset 0 0 0 ${thick}px rgba(${r},${g},${b},0.9), inset 0 0 ${blur}px rgba(${r},${g},${b},0.5)`,
        transition: 'box-shadow 0.3s ease', // 升级时厚度/颜色平滑过渡
      }}
    />
  );
}
