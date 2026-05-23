import { type MouseEvent } from 'react';
import { Clock } from 'lucide-react';
import { Severity } from '../types';

export interface TimelineEvent {
  id: string;
  t: number; // 秒：视频模式为视频时间，相机模式为会话已用时
  label: string;
  severity: Severity;
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

/** 事件时间轴：彩色标记每个行为事件，点击可跳转(上传视频)。*/
export default function EventTimeline({ events, duration, currentTime, seekable, onSeek }: Props) {
  const dur = duration > 0 ? duration : Math.max(currentTime, 1);
  const pct = (t: number) => Math.min(100, Math.max(0, (t / dur) * 100));

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
        <span className="text-xs text-outline">{events.length} 个事件{seekable ? ' · 点击跳转' : ''}</span>
      </div>

      <div
        className={`relative h-9 rounded-md bg-slate-100 border border-outline-variant/40 ${seekable ? 'cursor-pointer' : ''}`}
        onClick={handleTrackClick}
      >
        {/* 已播放进度 */}
        <div className="absolute inset-y-0 left-0 bg-primary/10 rounded-l-md" style={{ width: `${pct(currentTime)}%` }} />
        {/* 事件标记 */}
        {events.map((ev) => (
          <div
            key={ev.id}
            title={`${ev.label} @ ${fmt(ev.t)}`}
            onClick={(e) => {
              e.stopPropagation();
              if (seekable) onSeek(ev.t);
            }}
            className={`absolute top-1 bottom-1 w-[3px] rounded-full ${seekable ? 'cursor-pointer hover:w-[5px]' : ''}`}
            style={{ left: `calc(${pct(ev.t)}% - 1.5px)`, backgroundColor: sevColor(ev.severity), transition: 'width 0.15s' }}
          />
        ))}
        {/* 播放头 */}
        <div className="absolute top-0 bottom-0 w-[2px] bg-slate-700" style={{ left: `calc(${pct(currentTime)}% - 1px)` }}>
          <div className="absolute -top-1 -left-[3px] w-2 h-2 rounded-full bg-slate-700" />
        </div>
      </div>

      <div className="flex justify-between mt-1 text-[10px] font-mono text-outline-variant">
        <span>{fmt(currentTime)}</span>
        <span>{fmt(dur)}</span>
      </div>
    </div>
  );
}
