import { FileText, Printer, X } from 'lucide-react';
import type { Severity } from '../types';
import type { TimelineEvent } from './EventTimeline';
import type { Snapshot } from './SnapshotGallery';

interface Props {
  open: boolean;
  onClose: () => void;
  score: number;
  behaviorScore?: number;
  fatigueScore?: number;
  drivingSeconds: number;
  driverName?: string | null;
  scoreHistory: number[];
  events: TimelineEvent[];
  snapshots: Snapshot[];
  review: string;
  source?: string;
}

const SEV_COLOR: Record<string, string> = {
  none: '#22c55e', low: '#22c55e', medium: '#f59e0b', high: '#ef4444', critical: '#b91c1c',
};
const grade = (s: number) => (s >= 90 ? 'A 优秀' : s >= 80 ? 'B 良好' : s >= 70 ? 'C 一般' : s >= 60 ? 'D 偏差' : 'E 较差');
const fmtDur = (sec: number) => `${Math.floor(sec / 60)} 分 ${sec % 60} 秒`;

/** 行程图文报告：评分/曲线/事件/AI 点评/抓拍，可一键打印为 PDF。*/
export default function TripReportModal({ open, onClose, score, behaviorScore, fatigueScore, drivingSeconds, driverName, scoreHistory, events, snapshots, review, source }: Props) {
  if (!open) return null;

  // 评分曲线 sparkline
  const w = 600;
  const h = 90;
  const pts = scoreHistory.length > 1
    ? scoreHistory.map((v, i) => `${(i / (scoreHistory.length - 1)) * w},${h - (Math.max(0, Math.min(100, v)) / 100) * h}`).join(' ')
    : '';

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/50 backdrop-blur-sm print:bg-white print:block" onClick={onClose}>
      <div id="trip-report" className="w-[760px] max-h-[90vh] overflow-y-auto bg-white rounded-2xl shadow-2xl print:shadow-none print:rounded-none print:max-h-none print:w-full" onClick={(e) => e.stopPropagation()}>
        {/* 操作条(打印时隐藏) */}
        <div className="sticky top-0 flex items-center justify-between px-6 h-14 border-b border-outline-variant bg-white print:hidden">
          <h3 className="font-display font-bold text-slate-800 flex items-center gap-2"><FileText size={18} className="text-primary" /> 行程报告</h3>
          <div className="flex items-center gap-2">
            <button onClick={() => window.print()} className="btn-primary px-4 py-1.5 text-sm"><Printer size={15} /> 打印 / 导出 PDF</button>
            <button onClick={onClose} className="p-1.5 rounded-full hover:bg-surface-container text-slate-500"><X size={20} /></button>
          </div>
        </div>

        <div className="p-8 space-y-6 text-slate-800">
          {/* 抬头 */}
          <div className="flex items-start justify-between">
            <div>
              <h1 className="font-display text-2xl font-bold">驾驶状态行程报告</h1>
              <p className="text-sm text-slate-500 mt-1">{new Date().toLocaleString('zh-CN')}</p>
            </div>
            <div className="text-right text-sm text-slate-600">
              <div>驾驶员：<b>{driverName || '未登记'}</b></div>
              <div>数据来源：{source || '实时相机'}</div>
              <div>行驶时长：{fmtDur(drivingSeconds)}</div>
            </div>
          </div>

          {/* 总评分 */}
          <div className="grid grid-cols-3 gap-4">
            <div className="rounded-xl border border-outline-variant p-4 text-center">
              <div className="text-xs text-slate-400">综合评分</div>
              <div className="text-4xl font-bold" style={{ color: score >= 80 ? '#16a34a' : score >= 60 ? '#f59e0b' : '#ef4444' }}>{Math.round(score)}</div>
              <div className="text-sm font-medium text-slate-600">{grade(score)}</div>
            </div>
            <div className="rounded-xl border border-outline-variant p-4 text-center">
              <div className="text-xs text-slate-400">驾驶行为</div>
              <div className="text-4xl font-bold text-slate-700">{behaviorScore != null ? Math.round(behaviorScore) : '--'}</div>
            </div>
            <div className="rounded-xl border border-outline-variant p-4 text-center">
              <div className="text-xs text-slate-400">疲劳状态</div>
              <div className="text-4xl font-bold text-slate-700">{fatigueScore != null ? Math.round(fatigueScore) : '--'}</div>
            </div>
          </div>

          {/* 评分曲线 */}
          <div>
            <div className="text-sm font-bold text-slate-500 mb-2">评分曲线</div>
            {pts ? (
              <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-24 rounded-lg bg-slate-50 border border-outline-variant/60">
                <polyline points={pts} fill="none" stroke="#0ea5e9" strokeWidth={2} />
              </svg>
            ) : <p className="text-sm text-slate-400">数据不足</p>}
          </div>

          {/* 事件时间线 */}
          <div>
            <div className="text-sm font-bold text-slate-500 mb-2">关键事件（{events.length}）</div>
            {events.length === 0 ? <p className="text-sm text-slate-400">本次行程无告警事件</p> : (
              <div className="space-y-1 max-h-52 overflow-y-auto print:max-h-none">
                {events.slice(-30).map((e) => (
                  <div key={e.id} className="flex items-center gap-2 text-sm">
                    <span className="w-2 h-2 rounded-full shrink-0" style={{ background: SEV_COLOR[e.severity] || '#94a3b8' }} />
                    <span className="tabular-nums text-slate-400 w-14">{Math.floor(e.t / 60)}:{String(Math.floor(e.t % 60)).padStart(2, '0')}</span>
                    <span className="text-slate-700">{e.label}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* AI 点评 */}
          <div>
            <div className="text-sm font-bold text-slate-500 mb-2">AI 驾驶点评</div>
            <p className="text-sm leading-relaxed text-slate-700 whitespace-pre-wrap bg-slate-50 border border-outline-variant/60 rounded-lg p-3">{review || '暂无 AI 点评。'}</p>
          </div>

          {/* 抓拍 */}
          {snapshots.length > 0 && (
            <div>
              <div className="text-sm font-bold text-slate-500 mb-2">告警抓拍（{snapshots.length}）</div>
              <div className="grid grid-cols-4 gap-2">
                {snapshots.slice(-8).map((s) => (
                  <div key={s.id} className="rounded-lg overflow-hidden border border-outline-variant/60">
                    <img src={s.url} alt={s.label} className="w-full h-20 object-cover" />
                    <div className="px-1 py-0.5 text-[10px] truncate" style={{ color: SEV_COLOR[s.severity] }}>{s.label}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="text-center text-[11px] text-slate-400 pt-4 border-t border-outline-variant/60">由 DMS 驾驶状态推演与行为测算平台生成</div>
        </div>
      </div>
    </div>
  );
}
