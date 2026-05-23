import { motion } from 'motion/react';
import { AlertTriangle, BrainCircuit, Coffee, Download, EyeOff, Hourglass, List as ListIcon, Play, Radar, Smartphone } from 'lucide-react';
import { AnalysisResult, Detection, DrivingStats } from '../types';
import ScoreGauge from './ScoreGauge';
import MetricsRadar from './MetricsRadar';

interface SidebarProps {
  stats: DrivingStats | null;
  detections: Detection[];
  analysisText: string;
  isAnalyzing: boolean;
  latestResult: AnalysisResult | null;
  onDownloadReport: () => void;
  currentTime: number;
  seekable: boolean;
  onSeek: (t: number) => void;
}

function iconForDetection(type: string) {
  if (type.includes('手机') || type.includes('电话')) return Smartphone;
  if (type.includes('视线') || type.includes('姿态')) return EyeOff;
  if (type.includes('疲劳') || type.includes('哈欠')) return Coffee;
  return AlertTriangle;
}

function statusColor(status?: string) {
  if (status === '正常') return 'bg-green-50 text-green-600 border-green-100';
  if (status === '注意') return 'bg-yellow-50 text-yellow-600 border-yellow-100';
  if (status === '警告') return 'bg-orange-50 text-orange-600 border-orange-100';
  if (status === '危险') return 'bg-red-50 text-red-600 border-red-100';
  return 'bg-slate-100 text-slate-500 border-slate-200';
}

function scoreColorClass(v?: number) {
  if (v == null) return 'text-slate-300';
  if (v >= 80) return 'text-green-600';
  if (v >= 60) return 'text-amber-500';
  return 'text-red-500';
}

function scoreColorHex(v?: number) {
  if (v == null) return '#cbd5e1';
  if (v >= 80) return '#22c55e';
  if (v >= 60) return '#f59e0b';
  return '#ef4444';
}

function sourceLabel(source: string) {
  if (source === 'behavior') return '行为识别';
  if (source === 'fatigue') return '疲劳检测';
  return source || '算法';
}

function fmtSec(s?: number): string | null {
  if (s == null) return null;
  const m = Math.floor(s / 60);
  const ss = Math.floor(s % 60);
  return `${m}:${ss.toString().padStart(2, '0')}`;
}

export default function Sidebar({ stats, detections, analysisText, isAnalyzing, latestResult, onDownloadReport, currentTime, seekable, onSeek }: SidebarProps) {
  const capabilityMode = latestResult?.capabilities.behavior?.mode || latestResult?.capabilities.fatigue?.mode;
  // 找到与当前播放头最接近的检测条目用于高亮(只在可跳转/已记录 vt 时)
  const activeId = (() => {
    let best: { id: string; d: number } | null = null;
    for (const it of detections) {
      if (it.vt == null) continue;
      const dd = Math.abs(it.vt - currentTime);
      if (dd <= 1.2 && (!best || dd < best.d)) best = { id: it.id, d: dd };
    }
    return best?.id ?? null;
  })();

  return (
    <div className="w-[380px] shrink-0 flex flex-col gap-gutter h-full">
      <div className="card p-6 flex flex-col items-center justify-center relative overflow-hidden shrink-0">
        <h3 className="font-display text-xl font-bold text-on-surface mb-2 w-full text-left border-b border-outline-variant pb-2">驾驶状态评分</h3>

        <div className="w-full flex flex-col items-center mt-2 -mb-1">
          <ScoreGauge score={stats ? stats.score : null} size={236} />
          <span className="text-xs font-bold text-slate-400 -mt-2 tracking-widest">综合评分</span>
        </div>

        <div className="mt-2">
          <div className={`px-6 py-2 rounded-full font-bold flex items-center gap-2 border ${statusColor(stats?.status)}`}>
            {stats ? <AlertTriangle size={18} /> : <Hourglass size={18} />}
            <span>{stats ? stats.status : '等待数据'}</span>
          </div>
        </div>

        {stats && (
          <div className="grid grid-cols-2 gap-3 w-full mt-4">
            {([
              ['驾驶行为评分', stats.behavior_score],
              ['疲劳状态评分', stats.fatigue_score],
            ] as const).map(([label, value]) => (
              <div key={label} className="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-500 font-medium">{label}</span>
                  <span className={`text-2xl font-bold font-display ${scoreColorClass(value)}`}>{value ?? '--'}</span>
                </div>
                <div className="mt-1.5 h-1.5 rounded-full bg-slate-200 overflow-hidden">
                  <div className="h-full rounded-full" style={{ width: `${value ?? 0}%`, backgroundColor: scoreColorHex(value), transition: 'width 0.5s ease' }} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="card p-4 flex flex-col min-h-0 overflow-hidden" style={{ height: '280px' }}>
        <h3 className="font-display text-lg font-bold text-on-surface mb-3 border-b border-outline-variant pb-2 flex items-center justify-between">
          <span>检测结果列表</span>
          <span className="text-xs font-medium text-outline">{detections.length} 条{seekable ? ' · 点击跳转' : ''}</span>
        </h3>
        {detections.length > 0 ? (
          <div className="overflow-y-auto flex-1 pr-2 space-y-2">
            {[...detections].reverse().map((d) => {
              const Icon = iconForDetection(d.type);
              const canJump = seekable && d.vt != null;
              const active = d.id === activeId;
              const timeLabel = fmtSec(d.vt) ?? d.timestamp;
              return (
                <motion.div
                  key={d.id}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  onClick={() => canJump && onSeek(d.vt as number)}
                  className={`flex items-center justify-between p-3 rounded border transition-all ${
                    active
                      ? 'bg-primary/10 border-primary/50 shadow-[0_0_0_1px_rgba(59,130,246,0.3)]'
                      : 'bg-surface-container border-outline-variant/50 hover:shadow-[0_4px_12px_rgba(0,0,0,0.06)]'
                  } ${canJump ? 'cursor-pointer hover:border-primary/40' : ''}`}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="p-2 rounded bg-white shadow-sm border border-outline-variant/20">
                      <Icon className="text-primary" size={16} />
                    </div>
                    <div className="min-w-0">
                      <span className="font-medium text-sm truncate block">{d.type}</span>
                      <span className="text-[11px] text-outline">{sourceLabel(d.source)} · {(d.confidence * 100).toFixed(0)}% · {d.severity}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    <span className="text-xs text-outline font-mono">{timeLabel}</span>
                    {canJump && <Play size={13} className={active ? 'text-primary' : 'text-outline-variant'} />}
                  </div>
                </motion.div>
              );
            })}
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-400 opacity-60">
            <ListIcon size={48} className="mb-2" />
            <p className="text-sm">暂无检测记录</p>
          </div>
        )}
      </div>

      <div className="card p-4 flex flex-col min-h-0 flex-1 overflow-hidden">
        <h3 className="font-display text-lg font-bold text-on-surface mb-3 border-b border-outline-variant pb-2">大模型分析区</h3>
        <div className="flex-1 flex flex-col min-h-0">
          <div className="relative flex items-center justify-center bg-surface-container-low rounded-lg border border-outline-variant/30 shrink-0 mb-4 py-2 min-h-[176px]">
            {stats ? (
              <MetricsRadar stats={stats} size={200} />
            ) : (
              <div className="h-40 flex flex-col items-center justify-center text-slate-300">
                <Radar size={32} className="mb-1" />
                <p className="text-[10px] font-bold uppercase tracking-tight">实时五维状态</p>
              </div>
            )}
            {capabilityMode === 'fallback' && (
              <p className="absolute bottom-1 right-2 text-[10px] text-orange-500">部分模型回退</p>
            )}
          </div>

          <div className="flex-1 flex flex-col min-h-0">
            <div className="flex-1 p-4 bg-slate-50/50 rounded-lg border border-dashed border-outline-variant overflow-y-auto">
              <div className="flex items-start gap-2 text-sm text-outline leading-relaxed">
                <BrainCircuit size={18} className="shrink-0 text-primary mt-0.5" />
                <p>{analysisText || (isAnalyzing ? '正在基于真实检测结果生成分析。' : '等待真实检测数据。')}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <button className="w-full btn-secondary mt-auto" disabled={!latestResult} onClick={onDownloadReport}>
        <Download size={20} />
        下载结果报告
      </button>
    </div>
  );
}
