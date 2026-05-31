import { type MouseEvent, useEffect, useState } from 'react';
import { Clock } from 'lucide-react';
import { Severity } from '../types';

export interface TimelineEvent {
  id: string;
  t: number; // 秒：视频模式为视频时间，相机模式为会话已用时
  label: string;
  severity: Severity;
  thumb?: string; // 事件发生瞬间的抓拍缩略图(dataURL)，悬停预览/点击回看
}

interface Props {
  events: TimelineEvent[];
  duration: number;
  currentTime: number;
  seekable: boolean;
  onSeek: (t: number) => void;
}

function fmt(t: number): string {
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function sevColor(sev: Severity): string {
  if (sev === 'critical' || sev === 'high') return '#ef4444';
  if (sev === 'medium') return '#f59e0b';
  return '#38bdf8';
}

/** 事件时间轴：彩色标记每个行为事件，悬停看抓拍缩略图，点击可跳转(上传视频)。*/
export default function EventTimeline({ events, duration, currentTime, seekable, onSeek }: Props) {
  const dur = duration > 0 ? duration : Math.max(currentTime, 1);
  const pct = (t: number) => Math.min(100, Math.max(0, (t / dur) * 100));
  const [hover, setHover] = useState<TimelineEvent | null>(null);
  // 悬停的事件被列表上限(slice -200)挤掉时，其 onMouseLeave 不会触发 → 主动清除，避免缩略图卡住
  useEffect(() => {
    setHover((h) => (h && events.some((e) => e.id === h.id) ? h : null));
  }, [events]);

  const handleTrackClick = (e: MouseEvent<HTMLDivElement>) => {
    if (!seekable) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    onSeek(Math.max(0, Math.min(dur, ratio * dur)));
  };

  return (
    <div className="card p-4 shrink-0">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-display text-base font-bold text-on-surface flex items-center gap-2">
          <Clock size={16} className="text-primary" /> 事件时间轴
        </h3>
        <span className="text-xs text-outline">{events.length} 个事件 · 悬停看抓拍{seekable ? ' · 点击跳转' : ''}</span>
      </div>

      <div
        className={`relative h-9 rounded-md bg-slate-100 border border-outline-variant/40 ${seekable ? 'cursor-pointer' : ''}`}
        onClick={handleTrackClick}
      >
        {/* 已播放进度 */}
        <div className="absolute inset-y-0 left-0 bg-primary/10 rounded-l-md" style={{ width: `${pct(currentTime)}%` }} />
        {/* 事件标记：外层加宽透明热区便于悬停/点击，内层细条为视觉标记 */}
        {events.map((ev) => (
          <div
            key={ev.id}
            onMouseEnter={() => setHover(ev)}
            onMouseLeave={() => setHover((h) => (h?.id === ev.id ? null : h))}
            onClick={(e) => {
              e.stopPropagation();
              if (seekable) onSeek(ev.t);
            }}
            className={`absolute top-0 bottom-0 flex justify-center ${seekable ? 'cursor-pointer' : 'cursor-default'}`}
            style={{ left: `calc(${pct(ev.t)}% - 6px)`, width: 12 }}
          >
            <span
              className="my-1 w-[3px] rounded-full"
              style={{
                backgroundColor: sevColor(ev.severity),
                transform: hover?.id === ev.id ? 'scaleX(2)' : 'scaleX(1)',
                transition: 'transform 0.15s',
              }}
            />
          </div>
        ))}
        {/* 播放头 */}
        <div className="absolute top-0 bottom-0 w-[2px] bg-slate-700" style={{ left: `calc(${pct(currentTime)}% - 1px)` }}>
          <div className="absolute -top-1 -left-[3px] w-2 h-2 rounded-full bg-slate-700" />
        </div>

        {/* 悬停缩略图预览：浮于标记上方，缩略图 + 行为 + 时刻 */}
        {hover && (
          <div
            className="absolute bottom-full mb-2 z-20 pointer-events-none -translate-x-1/2"
            style={{ left: `${pct(hover.t)}%` }}
          >
            <div className="rounded-lg overflow-hidden border-2 bg-slate-900 shadow-xl" style={{ borderColor: sevColor(hover.severity) }}>
              {hover.thumb ? (
                <img src={hover.thumb} alt={hover.label} className="block w-40 h-[90px] object-cover" />
              ) : (
                <div className="w-40 h-[90px] flex items-center justify-center text-[11px] text-slate-400">无抓拍</div>
              )}
              <div className="px-2 py-1 flex items-center justify-between gap-2 text-[11px] font-bold text-white">
                <span className="truncate">{hover.label}</span>
                <span className="tabular-nums opacity-80 shrink-0">{fmt(hover.t)}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="flex justify-between mt-1 text-[10px] font-mono text-outline-variant">
        <span>{fmt(currentTime)}</span>
        <span>{fmt(dur)}</span>
      </div>
    </div>
  );
}
