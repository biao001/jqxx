import { useEffect, useState } from 'react';
import { Clock, Sparkles, Trash2, X } from 'lucide-react';

interface SessionRow {
  id: string;
  created_at: number;
  source: string;
  duration: number;
  score_avg: number;
  score_min: number;
  behavior_avg?: number;
  fatigue_avg?: number;
  events_count: number;
}
interface SessionDetail extends SessionRow {
  events?: { id: string; t: number; label: string; severity: string }[];
  score_series?: { t: number; score: number }[];
}

interface Props {
  open: boolean;
  onClose: () => void;
  backendUrl: string;
}

const scoreColor = (v?: number) => (v == null ? '#94a3b8' : v >= 80 ? '#22c55e' : v >= 60 ? '#f59e0b' : '#ef4444');
const fmtDur = (s: number) => `${Math.floor(s / 60)}分${Math.floor(s % 60)}秒`;
const fmtDate = (sec: number) => new Date(sec * 1000).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });

function Sparkline({ series }: { series: { t: number; score: number }[] }) {
  if (!series || series.length < 2) return <div className="text-xs text-slate-400">数据不足</div>;
  const w = 560;
  const h = 90;
  const maxT = series[series.length - 1].t || 1;
  const pts = series.map((p) => `${(p.t / maxT) * w},${h - (p.score / 100) * h}`).join(' ');
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ height: 90 }}>
      {[20, 40, 60, 80].map((y) => (
        <line key={y} x1={0} y1={h - (y / 100) * h} x2={w} y2={h - (y / 100) * h} stroke="#eef2f7" strokeWidth={1} />
      ))}
      <polyline points={pts} fill="none" stroke="#3b82f6" strokeWidth={2} strokeLinejoin="round" />
    </svg>
  );
}

/** 历史行程报告：列表 + 选中会话的评分/事件详情。*/
export default function HistoryModal({ open, onClose, backendUrl }: Props) {
  const [list, setList] = useState<SessionRow[]>([]);
  const [sel, setSel] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [review, setReview] = useState<string | null>(null);
  const [reviewing, setReviewing] = useState(false);

  const refresh = () => {
    setLoading(true);
    fetch(`${backendUrl}/api/sessions`)
      .then((r) => r.json())
      .then((d) => setList(d.sessions || []))
      .catch(() => setList([]))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (open) {
      refresh();
      setSel(null);
    }
  }, [open]);

  if (!open) return null;

  const openDetail = (id: string) => {
    setReview(null);
    fetch(`${backendUrl}/api/sessions/${id}`).then((r) => r.json()).then(setSel).catch(() => {});
  };
  const genReview = () => {
    if (!sel) return;
    setReviewing(true);
    fetch(`${backendUrl}/api/llm/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ summary: sel }),
    })
      .then((r) => r.json())
      .then((d) => setReview(d.review || '点评生成失败'))
      .catch(() => setReview('点评生成失败'))
      .finally(() => setReviewing(false));
  };
  const del = (id: string) =>
    fetch(`${backendUrl}/api/sessions/${id}`, { method: 'DELETE' }).then(() => {
      if (sel?.id === id) setSel(null);
      refresh();
    });

  const breakdown = (sel?.events || []).reduce<Record<string, number>>((acc, e) => {
    acc[e.label] = (acc[e.label] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 backdrop-blur-sm" onClick={onClose}>
      <div className="w-[1000px] h-[640px] bg-white rounded-2xl shadow-2xl flex overflow-hidden" onClick={(e) => e.stopPropagation()}>
        {/* 列表 */}
        <div className="w-[320px] border-r border-outline-variant flex flex-col">
          <div className="px-4 h-14 flex items-center justify-between border-b border-outline-variant">
            <h3 className="font-display font-bold text-slate-800 flex items-center gap-2"><Clock size={18} className="text-primary" />行程历史</h3>
            <span className="text-xs text-slate-400">{list.length} 次</span>
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-2">
            {loading && <div className="text-sm text-slate-400 p-4 text-center">加载中…</div>}
            {!loading && list.length === 0 && <div className="text-sm text-slate-400 p-4 text-center">暂无历史行程</div>}
            {list.map((s) => (
              <div
                key={s.id}
                onClick={() => openDetail(s.id)}
                className={`group p-3 rounded-lg border cursor-pointer transition-all ${sel?.id === s.id ? 'border-primary/50 bg-primary/5' : 'border-outline-variant/60 hover:border-primary/30'}`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-slate-700 truncate max-w-[180px]">{s.source}</span>
                  <span className="text-2xl font-bold font-display" style={{ color: scoreColor(s.score_avg) }}>{Math.round(s.score_avg)}</span>
                </div>
                <div className="flex items-center justify-between mt-1 text-[11px] text-slate-400">
                  <span>{fmtDate(s.created_at)} · {fmtDur(s.duration)}</span>
                  <span>{s.events_count} 事件</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 详情 */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="px-5 h-14 flex items-center justify-between border-b border-outline-variant">
            <h3 className="font-display font-bold text-slate-800 truncate">{sel ? sel.source : '行程报告'}</h3>
            <button onClick={onClose} className="p-1.5 rounded-full hover:bg-surface-container text-slate-500"><X size={20} /></button>
          </div>
          {!sel ? (
            <div className="flex-1 flex items-center justify-center text-slate-400 text-sm">从左侧选择一次行程查看报告</div>
          ) : (
            <div className="flex-1 overflow-y-auto p-5 space-y-5">
              <div className="grid grid-cols-4 gap-3">
                {([
                  ['综合(均)', sel.score_avg],
                  ['最低分', sel.score_min],
                  ['行为(均)', sel.behavior_avg],
                  ['疲劳(均)', sel.fatigue_avg],
                ] as const).map(([label, v]) => (
                  <div key={label} className="rounded-xl border border-outline-variant/60 bg-slate-50 p-3 text-center">
                    <div className="text-[11px] text-slate-500">{label}</div>
                    <div className="text-3xl font-bold font-display" style={{ color: scoreColor(v as number) }}>{v == null ? '--' : Math.round(v as number)}</div>
                  </div>
                ))}
              </div>

              <div>
                <div className="text-sm font-bold text-slate-600 mb-1">综合评分走势</div>
                <div className="rounded-xl border border-outline-variant/60 p-3">
                  <Sparkline series={sel.score_series || []} />
                </div>
              </div>

              <div>
                <div className="text-sm font-bold text-slate-600 mb-2">事件分布（{(sel.events || []).length} 次）</div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(breakdown).length === 0 && <span className="text-sm text-slate-400">无事件</span>}
                  {Object.entries(breakdown).map(([label, n]) => (
                    <span key={label} className="px-3 py-1 rounded-full bg-slate-100 border border-outline-variant/60 text-sm font-medium text-slate-700">
                      {label} <span className="text-primary font-bold">×{n}</span>
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-bold text-slate-600">AI 驾驶点评</span>
                  <button
                    onClick={genReview}
                    disabled={reviewing}
                    className="text-xs font-medium px-3 py-1.5 rounded-lg bg-primary text-white hover:bg-primary/90 disabled:opacity-50 flex items-center gap-1"
                  >
                    <Sparkles size={13} /> {reviewing ? '生成中…' : review ? '重新生成' : '生成 AI 点评'}
                  </button>
                </div>
                {review && (
                  <div className="rounded-xl border border-primary/20 bg-primary/5 p-4 text-sm text-slate-700 whitespace-pre-wrap leading-relaxed">
                    {review}
                  </div>
                )}
              </div>

              <button onClick={() => del(sel.id)} className="text-xs text-red-500 hover:text-red-600 flex items-center gap-1">
                <Trash2 size={13} /> 删除此行程
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
