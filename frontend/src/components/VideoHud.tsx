import { motion } from 'motion/react';

interface Props {
  active: boolean;
  scan?: boolean;
  latencyMs?: number;
  frameId?: number | null;
}

const CORNER = 'absolute w-7 h-7 border-cyan-400/70';

/** 视频上的机器视觉 HUD：四角取景框 + 扫描线 + 实时遥测读数。*/
export default function VideoHud({ active, scan = true, latencyMs, frameId }: Props) {
  return (
    <div className="absolute inset-0 pointer-events-none z-20">
      {/* 四角取景框 */}
      <div className={`${CORNER} top-3 left-3 border-t-2 border-l-2 rounded-tl`} />
      <div className={`${CORNER} top-3 right-3 border-t-2 border-r-2 rounded-tr`} />
      <div className={`${CORNER} bottom-3 left-3 border-b-2 border-l-2 rounded-bl`} />
      <div className={`${CORNER} bottom-3 right-3 border-b-2 border-r-2 rounded-br`} />

      {/* 扫描线 */}
      {active && scan && (
        <motion.div
          className="absolute left-3 right-3 h-[2px] bg-cyan-400/50"
          style={{ boxShadow: '0 0 14px rgba(34,211,238,0.7)' }}
          initial={{ top: '8%' }}
          animate={{ top: ['8%', '92%', '8%'] }}
          transition={{ duration: 4.5, repeat: Infinity, ease: 'easeInOut' }}
        />
      )}

      {/* 实时遥测读数 */}
      <div className="absolute top-4 right-4 text-right font-mono text-[10px] leading-tight text-cyan-300/85 select-none">
        <div className="flex items-center justify-end gap-1.5">
          <span className={`w-1.5 h-1.5 rounded-full ${active ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'}`} />
          <span className="tracking-widest">{active ? 'ANALYZING' : 'STANDBY'}</span>
        </div>
        <div className="mt-1 tabular-nums">LATENCY {latencyMs != null ? Math.round(latencyMs) : '--'} ms</div>
        <div className="tabular-nums">FRAME #{frameId ?? '--'}</div>
      </div>
    </div>
  );
}
