import { type MutableRefObject, type RefObject, useEffect, useRef } from 'react';
import type { Detection } from '../types';

/** 一帧的检测集合：t = 该帧在客户端被抓取(送检)的时刻(performance.now ms)。 */
export interface DetSet {
  t: number;
  dets: Detection[];
}

interface Props {
  videoRef: RefObject<HTMLVideoElement | null>;
  /** 仅在实时相机 + 正在分析时启用(开启抓帧/绘制循环)。*/
  enabled: boolean;
  /** 历史检测集合(最新在末尾)，每条带其送检时刻 t。*/
  detSetsRef: MutableRefObject<DetSet[]>;
  /** 自适应显示延迟(ms)：≈端到端检测延迟，让画面对齐"那一帧的检测结果"。*/
  delayMsRef: MutableRefObject<number>;
  /** 是否绘制识别框(关掉只做延迟同步的纯画面)。*/
  showBoxes?: boolean;
}

function sevHex(sev?: string): string {
  return sev === 'critical' || sev === 'high' ? '#f87171' : sev === 'medium' ? '#fbbf24' : '#38bdf8';
}

const POOL = 32; // 帧环形缓冲槽位(每 rAF 写一格)：60Hz≈533ms、144Hz≈222ms 历史；不足部分由 effDelay 夹紧兜底
const BOX_SMOOTH = 0.85; // 换帧(新检测集合)那一拍的位置混合：≈贴合新检测、仅留少量去抖；越接近 1 越"硬焊"
const LOCK_MS = 180; // 新框「收拢锁定」动效时长
const STALE_MS = 320; // 匹配到的检测比显示帧旧超过此值 → 不画框(抑制残留)
const MAX_DPR = 2; // 限制画布背板分辨率，避免大屏 4K 重绘拖慢

interface Box {
  cx: number;
  cy: number;
  w: number;
  h: number;
  born: number;
  setT: number; // 当前框对应的检测集合时刻；仅当它变化时才移动框(与背景帧同拍 → 精确焊接)
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

/** 把一个(已平滑的)框画在画布上：四角取景括号 + 标签条，带锁定淡入。*/
function drawBox(
  ctx: CanvasRenderingContext2D,
  ox: number,
  oy: number,
  scale: number,
  b: Box,
  det: Detection,
  lock: number,
  color: string,
) {
  const x = ox + (b.cx - b.w / 2) * scale;
  const y = oy + (b.cy - b.h / 2) * scale;
  const w = Math.max(2, b.w * scale);
  const h = Math.max(2, b.h * scale);
  const grow = (1 - lock) * 10; // 锁定前角标略微外扩，制造"收拢"感
  const alpha = 0.4 + 0.6 * lock; // 锁定前淡入

  ctx.save();
  // 外框微光
  ctx.globalAlpha = alpha * 0.5;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  roundRect(ctx, x, y, w, h, 3);
  ctx.stroke();

  // 四角 L 形取景括号
  ctx.globalAlpha = alpha;
  ctx.lineWidth = 2;
  const c = Math.min(16, w * 0.32, h * 0.32) + grow;
  ctx.beginPath();
  ctx.moveTo(x, y + c); ctx.lineTo(x, y); ctx.lineTo(x + c, y); // 左上
  ctx.moveTo(x + w - c, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + c); // 右上
  ctx.moveTo(x, y + h - c); ctx.lineTo(x, y + h); ctx.lineTo(x + c, y + h); // 左下
  ctx.moveTo(x + w - c, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - c); // 右下
  ctx.stroke();

  // 标签条：行为代码 + 置信度
  const label = `${det.code || det.type}  ${Math.round((det.confidence ?? 0) * 100)}%`;
  ctx.font = '700 11px ui-sans-serif, system-ui, -apple-system, sans-serif';
  const ph = 15;
  const pad = 6;
  const tw = ctx.measureText(label).width;
  const ly = Math.max(0, y - ph); // 贴近顶边时下沉，避免画到画布外
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.fillRect(x, ly, tw + pad * 2, ph);
  ctx.fillStyle = '#fff';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, x + pad, ly + ph / 2 + 0.5);
  ctx.restore();
}

/**
 * 视频↔检测同步层(「丝滑」模式)：给直播画面加一个 ≈检测延迟 的缓冲延迟，
 * 让显示的那一帧正好对齐"它自己的检测结果"，框就像焊在物体上——
 * 不靠运动外推(会过冲/抖)，而是把画面拉回到检测所在的时刻。
 *
 * 做法：rAF 把直播帧连同时间戳存入环形缓冲；每帧取 (now - delay) 时刻最接近的
 * 缓冲帧绘到画布(letterbox 对齐 object-contain)，再取与"该显示帧时刻"最接近的
 * 检测集合，把框画在同一画布、同一坐标系上(像素级锁定)。画面整体晚 ~延迟，
 * 肉眼几乎无感，但框完美跟手。纯视觉层：不改后端判定/告警时机。
 */
export default function SyncedVideo({ videoRef, enabled, detSetsRef, delayMsRef, showBoxes = true }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const poolRef = useRef<{ t: number; cv: HTMLCanvasElement }[]>([]);
  const writeRef = useRef(0);
  const boxRef = useRef<Record<string, Box>>({});

  useEffect(() => {
    if (!enabled) return;
    // 进入同步模式：清空旧缓冲/框，避免恢复时显示上次会话的陈旧帧
    poolRef.current = [];
    writeRef.current = 0;
    boxRef.current = {};

    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const v = videoRef.current;
      const display = canvasRef.current;
      const ctx = display?.getContext('2d') ?? null;
      if (!v || !display || !ctx || v.videoWidth === 0 || v.readyState < 2) return;

      const now = performance.now();
      const vw = v.videoWidth;
      const vh = v.videoHeight;

      // 1) 抓当前直播帧入环形缓冲(带时间戳)
      let slot = poolRef.current[writeRef.current % POOL];
      if (!slot) {
        slot = { t: 0, cv: document.createElement('canvas') };
        poolRef.current[writeRef.current % POOL] = slot;
      }
      if (slot.cv.width !== vw || slot.cv.height !== vh) {
        slot.cv.width = vw;
        slot.cv.height = vh;
      }
      const sctx = slot.cv.getContext('2d');
      if (sctx) {
        sctx.drawImage(v, 0, 0, vw, vh);
        slot.t = now;
        writeRef.current++;
      }

      // 2) 选 (now - delay) 时刻最接近的缓冲帧 —— 这就是"延迟显示"的那一帧。
      //    缓冲只覆盖 POOL 个刷新间隔；高刷屏(如 144Hz)其时间跨度可能 < 最大延迟 240ms。
      //    若 target 落到环外，最近搜索会锁定最旧帧 → 该帧检测尚未回传、焊框漂移。
      //    故把生效延迟夹到"环内实际可达"(留 16ms 余量)，宁可少延一点也保证显示帧的检测已就位。
      let oldestT = now;
      for (const s of poolRef.current) {
        if (s.t > 0 && s.t < oldestT) oldestT = s.t;
      }
      const effDelay = Math.max(0, Math.min(delayMsRef.current, now - oldestT - 16));
      const targetT = now - effDelay;
      let frame: { t: number; cv: HTMLCanvasElement } = slot;
      let best = Infinity;
      for (const s of poolRef.current) {
        if (s.t <= 0) continue;
        const d = Math.abs(s.t - targetT);
        if (d < best) {
          best = d;
          frame = s;
        }
      }

      // 3) 画延迟帧(letterbox 复刻 object-contain，使框坐标系一致)
      const dpr = Math.min(MAX_DPR, window.devicePixelRatio || 1);
      const cw = display.clientWidth;
      const ch = display.clientHeight;
      const bw = Math.round(cw * dpr);
      const bh = Math.round(ch * dpr);
      if (display.width !== bw || display.height !== bh) {
        display.width = bw;
        display.height = bh;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cw, ch);
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, cw, ch);
      const scale = Math.min(cw / vw, ch / vh);
      const rw = vw * scale;
      const rh = vh * scale;
      const ox = (cw - rw) / 2;
      const oy = (ch - rh) / 2;
      ctx.drawImage(frame.cv, ox, oy, rw, rh);

      // 4) 取与"显示帧时刻"最接近的检测集合，把框焊在物体上
      if (showBoxes) {
        const sets = detSetsRef.current;
        let bestSet: DetSet | null = null;
        let bd = Infinity;
        for (const s of sets) {
          const d = Math.abs(s.t - frame.t);
          if (d < bd) {
            bd = d;
            bestSet = s;
          }
        }
        const seen = new Set<string>();
        if (bestSet && bd <= STALE_MS) {
          for (const det of bestSet.dets) {
            if (!Array.isArray(det.bbox) || det.bbox.length !== 4) continue;
            const [x1, y1, x2, y2] = det.bbox as number[];
            const key = det.code || det.type;
            seen.add(key);
            const tcx = (x1 + x2) / 2;
            const tcy = (y1 + y2) / 2;
            const tw = x2 - x1;
            const th = y2 - y1;
            let b = boxRef.current[key];
            if (!b) {
              b = { cx: tcx, cy: tcy, w: tw, h: th, born: now, setT: bestSet.t };
              boxRef.current[key] = b;
            } else if (b.setT !== bestSet.t) {
              // 仅当"时间对齐的检测集合"换新时才移动框 —— 与背景帧同拍更新，做到像素级焊接，
              // 不在同一检测上每帧 EMA(那会让框落后于已对齐的背景)。换帧时轻混一下吸收模型 1~2px 抖动。
              b.cx += (tcx - b.cx) * BOX_SMOOTH;
              b.cy += (tcy - b.cy) * BOX_SMOOTH;
              b.w += (tw - b.w) * BOX_SMOOTH;
              b.h += (th - b.h) * BOX_SMOOTH;
              b.setT = bestSet.t;
            }
            const lock = Math.min(1, (now - b.born) / LOCK_MS);
            drawBox(ctx, ox, oy, scale, b, det, lock, sevHex(det.severity));
          }
        }
        // 清理已消失的框(下帧重新淡入)
        for (const k in boxRef.current) {
          if (!seen.has(k)) delete boxRef.current[k];
        }
      }
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [enabled, videoRef, detSetsRef, delayMsRef, showBoxes]);

  if (!enabled) return null;
  // bg-black：首帧绘制前先盖住底层直播 video，避免切到同步模式时一帧"双画面"闪烁
  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none bg-black" />;
}
