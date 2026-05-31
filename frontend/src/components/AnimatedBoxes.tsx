import { type CSSProperties, type RefObject, useEffect, useRef, useState } from 'react';
import { videoBoxStyle } from '../lib/videoBox';
import type { Detection, Severity } from '../types';

function sevHex(sev?: string): string {
  return sev === 'critical' || sev === 'high' ? '#f87171' : sev === 'medium' ? '#fbbf24' : '#38bdf8';
}

interface Target {
  cx: number; cy: number; w: number; h: number;
  vx: number; vy: number; // 中心速度 px/ms（EMA 平滑），用于外推补偿延迟
  t: number;              // 最近一次后端结果的时刻(ms)
  severity: string; confidence: number;
}
interface Anim {
  cx: number; cy: number; w: number; h: number;
  severity: string; confidence: number; lastSeen: number;
}

interface Props {
  videoRef: RefObject<HTMLVideoElement | null>;
  detections: Detection[]; // 当前帧带 bbox 的检测（坐标已还原到视频原始尺寸）
  fresh: boolean;
  posSmooth?: number;    // 位置向(外推后)目标平滑：越大越跟手
  sizeSmooth?: number;   // 尺寸(宽高)平滑：越小框大小越稳
  velSmooth?: number;    // 速度 EMA 系数：越大越灵敏、越小越稳
  maxExtrapMs?: number;  // 最大外推时长(ms)，限制过冲——物体突停时框最多冲这么久
  ttlMs?: number;        // 某类暂时丢失后保留多久(冻结)，抑制单帧漏检闪烁
  sizeDeadzone?: number; // 尺寸冻结阈值：相对变化小于此值则框大小不变(同一物体不忽大忽小)
}

/**
 * 识别框追踪层：后端检测有 ~30fps + 50~100ms 延迟，直接画框会「跟不上」移动的物体。
 * 这里对每个 type 估计移动速度，用 requestAnimationFrame(~60fps) 做**运动外推**——
 * 把框画在物体「预测的当前位置」而非「检测到的过去位置」，抵消延迟，使框跟手贴物。
 *
 * 纯视觉层：不改动后端动作判定/告警时机。位置外推+平滑兼顾「跟手」与「丝滑」，
 * 尺寸单独平滑使框大小稳定；maxExtrapMs 限幅防止物体突停时框冲出去。
 */
export default function AnimatedBoxes({
  videoRef, detections, fresh,
  posSmooth = 0.45, sizeSmooth = 0.28, velSmooth = 0.35, maxExtrapMs = 110, ttlMs = 700,
  sizeDeadzone = 0.35,
}: Props) {
  const targetRef = useRef<Record<string, Target>>({});
  const animRef = useRef<Record<string, Anim>>({});
  const freshRef = useRef(fresh);
  const [, force] = useState(0);

  freshRef.current = fresh;

  // 后端新结果 → 更新目标位置，并用前后两次位置估计移动速度(EMA 平滑)
  useEffect(() => {
    const now = performance.now();
    const cur = targetRef.current;
    const next: Record<string, Target> = {};
    for (const d of detections) {
      if (!Array.isArray(d.bbox) || d.bbox.length !== 4) continue;
      const [x1, y1, x2, y2] = d.bbox as number[];
      const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2, w = x2 - x1, h = y2 - y1;
      const key = d.code || d.type; // 用英文行为代码(phone_use…)做键+标签，稳定且英文
      const prev = cur[key];
      let vx = 0, vy = 0;
      if (prev) {
        const dt = now - prev.t;
        if (dt > 1 && dt < 500) { // 合理帧间隔才更新速度；过长(暂停/丢失)则不算
          vx = prev.vx * (1 - velSmooth) + ((cx - prev.cx) / dt) * velSmooth;
          vy = prev.vy * (1 - velSmooth) + ((cy - prev.cy) / dt) * velSmooth;
        } else {
          vx = prev.vx; vy = prev.vy;
        }
      }
      next[key] = { cx, cy, w, h, vx, vy, t: now, severity: d.severity, confidence: d.confidence ?? 0 };
    }
    targetRef.current = next;
  }, [detections, velSmooth]);

  // fresh 转 false（结果停了）→ 清空，避免恢复时从旧位置跳变
  useEffect(() => { if (!fresh) animRef.current = {}; }, [fresh]);

  // ~60fps 外推 + 平滑循环
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const targets = targetRef.current;
      const anim = animRef.current;
      const now = performance.now();

      for (const type in targets) {
        const tg = targets[type];
        // 外推：用速度预测物体「现在」的中心位置，抵消推理+网络延迟（限幅防过冲）
        const dt = Math.min(now - tg.t, maxExtrapMs);
        const px = tg.cx + tg.vx * dt;
        const py = tg.cy + tg.vy * dt;
        const a = anim[type];
        if (!a) {
          anim[type] = { cx: px, cy: py, w: tg.w, h: tg.h, severity: tg.severity, confidence: tg.confidence, lastSeen: tg.t };
        } else {
          a.cx += (px - a.cx) * posSmooth;   // 位置向外推目标平滑（跟手）
          a.cy += (py - a.cy) * posSmooth;
          // 尺寸冻结：同一物体的小幅尺寸抖动忽略(框大小不变)；仅当尺寸显著变化
          // (物体明显靠近/远离，相对变化 > sizeDeadzone) 才平滑跟随，避免框忽大忽小。
          if (Math.abs(tg.w - a.w) / Math.max(1, a.w) > sizeDeadzone) a.w += (tg.w - a.w) * sizeSmooth;
          if (Math.abs(tg.h - a.h) / Math.max(1, a.h) > sizeDeadzone) a.h += (tg.h - a.h) * sizeSmooth;
          a.severity = tg.severity;
          a.confidence = tg.confidence;
          a.lastSeen = tg.t;
        }
      }
      // 目标已消失且超 TTL：移除（冻结一小段吸收单帧漏检）
      for (const type in anim) {
        if (!(type in targets) && now - anim[type].lastSeen > ttlMs) delete anim[type];
      }

      if (freshRef.current) force((n) => (n + 1) & 0x3fffffff);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [posSmooth, sizeSmooth, maxExtrapMs, ttlMs, sizeDeadzone]);

  if (!fresh) return null;

  return (
    <>
      {(Object.entries(animRef.current) as [string, Anim][]).map(([type, b]) => {
        const style = videoBoxStyle(videoRef.current, [b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2]);
        if (!style) return null;
        const c = sevHex(b.severity as Severity);
        return (
          <div key={type} className="absolute pointer-events-none" style={style as CSSProperties}>
            <div className="absolute inset-0 rounded-sm" style={{ border: `1px solid ${c}55`, boxShadow: `0 0 16px ${c}66, inset 0 0 14px ${c}22` }} />
            {[
              'top-0 left-0 border-t-2 border-l-2',
              'top-0 right-0 border-t-2 border-r-2',
              'bottom-0 left-0 border-b-2 border-l-2',
              'bottom-0 right-0 border-b-2 border-r-2',
            ].map((cls, i) => (
              <span key={i} className={`absolute w-3.5 h-3.5 ${cls}`} style={{ borderColor: c }} />
            ))}
            <div className="absolute left-0 top-0 -translate-y-full flex items-center gap-1.5 px-2 py-0.5 text-[11px] font-bold text-white" style={{ background: c }}>
              <span>{type}</span>
              <span className="opacity-85 tabular-nums">{Math.round((b.confidence ?? 0) * 100)}%</span>
            </div>
          </div>
        );
      })}
    </>
  );
}
