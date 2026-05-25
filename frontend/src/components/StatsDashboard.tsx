import { useEffect, useState } from 'react';
import { ArrowLeft, BarChart3, Car, Clock, RefreshCw, ShieldAlert, TrendingUp } from 'lucide-react';

interface Stats {
  total_sessions: number;
  total_duration: number;
  avg_score: number;
  total_events: number;
  by_day: { day: string; score: number }[];
  event_freq: { label: string; count: number }[];
  recent_scores: number[];
}

interface Props {
  backendUrl: string;
  onBack: () => void;
}

const fmtDur = (s: number) => (s >= 3600 ? `${(s / 3600).toFixed(1)}h` : `${Math.round(s / 60)}min`);
const scoreColor = (v: number) => (v >= 80 ? '#22c55e' : v >= 60 ? '#f59e0b' : '#ef4444');

function DayChart({ data }: { data: { day: string; score: number }[] }) {
  if (data.length === 0) return <div className="h-44 flex items-center justify-center text-slate-400 text-sm">暂无数据</div>;
  const w = 640;
  const h = 150;
  const n = data.length;
  const x = (i: number) => (n === 1 ? w / 2 : (i / (n - 1)) * (w - 30) + 15);
  const y = (s: number) => h - (s / 100) * (h - 20) - 10;
  const line = data.map((d, i) => `${x(i)},${y(d.score)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${w} ${h + 22}`} className="w-full">
      {[80, 60, 40].map((g) => (
        <line key={g} x1="0" y1={y(g)} x2={w} y2={y(g)} stroke="#eef2f7" strokeWidth="1" />
      ))}
      <polyline points={line} fill="none" stroke="#3b82f6" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
      {data.map((d, i) => (
        <g key={d.day}>
          <circle cx={x(i)} cy={y(d.score)} r="4" fill={scoreColor(d.score)} />
          <text x={x(i)} y={y(d.score) - 10} textAnchor="middle" style={{ fontSize: 11, fontWeight: 700, fill: '#475569' }}>{d.score}</text>
          <text x={x(i)} y={h + 16} textAnchor="middle" style={{ fontSize: 10, fill: '#94a3b8' }}>{d.day}</text>
        </g>
      ))}
    </svg>
  );
}

/** 数据看板：跨会话统计总览 + 评分趋势 + 违规频次。*/
export default function StatsDashboard({ backendUrl, onBack }: Props) {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    fetch(`${backendUrl}/api/stats`).then((r) => r.json()).then(setStats).catch(() => setStats(null)).finally(() => setLoading(false));
  };
  useEffect(load, []);

  const maxFreq = Math.max(1, ...(stats?.event_freq.map((e) => e.count) || [1]));

  return (
    <div className="h-screen overflow-y-auto bg-gradient-to-br from-slate-100 to-blue-50/40">
      <header className="bg-white/80 backdrop-blur border-b border-outline-variant/60 px-8 h-16 flex items-center justify-between sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="flex items-center gap-1 text-slate-500 hover:text-primary text-sm font-medium">
            <ArrowLeft size={18} /> 返回系统
          </button>
          <span className="w-px h-5 bg-outline-variant" />
          <h1 className="font-display text-xl font-bold text-slate-900 flex items-center gap-2"><BarChart3 size={20} className="text-primary" /> 数据看板</h1>
        </div>
        <button onClick={load} className="p-2 rounded-full hover:bg-slate-100 text-slate-500"><RefreshCw size={18} /></button>
      </header>

      <div className="max-w-[1080px] mx-auto p-8 space-y-6">
        {loading ? (
          <div className="text-center text-slate-400 py-20">加载中…</div>
        ) : !stats || stats.total_sessions === 0 ? (
          <div className="text-center text-slate-400 py-20">暂无行程数据，先去系统里跑几次分析吧。</div>
        ) : (
          <>
            <div className="grid grid-cols-4 gap-5">
              {[
                { icon: Car, label: '总行程数', value: stats.total_sessions, unit: '次', c: 'text-primary' },
                { icon: Clock, label: '总驾驶时长', value: fmtDur(stats.total_duration), unit: '', c: 'text-cyan-600' },
                { icon: TrendingUp, label: '平均综合评分', value: stats.avg_score, unit: '分', c: scoreColor(stats.avg_score) },
                { icon: ShieldAlert, label: '累计违规事件', value: stats.total_events, unit: '次', c: 'text-orange-600' },
              ].map((s) => (
                <div key={s.label} className="card p-5">
                  <div className="flex items-center gap-2 text-slate-400 text-xs font-medium mb-2"><s.icon size={15} /> {s.label}</div>
                  <div className="font-display text-3xl font-bold" style={{ color: typeof s.c === 'string' && s.c.startsWith('#') ? s.c : undefined }}>
                    <span className={s.c.startsWith('#') ? '' : s.c}>{s.value}</span>
                    <span className="text-base text-slate-400 font-medium ml-1">{s.unit}</span>
                  </div>
                </div>
              ))}
            </div>

            <div className="card p-6">
              <h3 className="font-display text-lg font-bold text-slate-800 mb-4 flex items-center gap-2"><TrendingUp size={18} className="text-primary" /> 每日平均评分趋势</h3>
              <DayChart data={stats.by_day} />
            </div>

            <div className="card p-6">
              <h3 className="font-display text-lg font-bold text-slate-800 mb-4 flex items-center gap-2"><ShieldAlert size={18} className="text-orange-500" /> 违规事件频次排行</h3>
              {stats.event_freq.length === 0 ? (
                <p className="text-sm text-slate-400">暂无违规事件</p>
              ) : (
                <div className="space-y-2.5">
                  {stats.event_freq.map((e) => (
                    <div key={e.label} className="flex items-center gap-3">
                      <span className="w-28 text-sm text-slate-600 shrink-0 truncate">{e.label}</span>
                      <div className="flex-1 h-5 rounded-full bg-slate-100 overflow-hidden">
                        <div className="h-full rounded-full bg-gradient-to-r from-orange-400 to-red-400" style={{ width: `${(e.count / maxFreq) * 100}%` }} />
                      </div>
                      <span className="w-8 text-right text-sm font-bold text-slate-700">{e.count}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
