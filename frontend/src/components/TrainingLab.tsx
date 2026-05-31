import { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, AlertTriangle, ArrowLeft, CheckCircle2, FlaskConical, Images as ImagesIcon, RefreshCw, Table2, Terminal, X } from 'lucide-react';

interface Props {
  backendUrl: string;
  onBack: () => void;
}

interface RunInfo { name: string; mtime: number; has_csv: boolean }
type Row = Record<string, number | string>;
interface ClassMetric { map50: number; map: number; p?: number; r?: number }
interface PerClass { epoch?: number; map50?: number; map?: number; per_class?: Record<string, ClassMetric> }
interface Metrics { columns: string[]; rows: Row[]; per_class?: PerClass }
interface LiveStatus {
  running: boolean; finished?: boolean; success?: boolean;
  project?: string; base?: string; out_model?: string; run?: string;
  epoch?: number; total_epochs?: number; best_map?: number; elapsed?: number; is_active?: boolean;
}

interface Series { name: string; color: string; data: (number | null)[] }

const fmtClock = (s = 0) => `${Math.floor(s / 60)}分${Math.round(s % 60)}秒`;
const fmtTime = (t: number) => new Date(t * 1000).toLocaleString('zh-CN', { hour12: false });
const num = (v: number | string | undefined) =>
  typeof v === 'number' ? (Math.abs(v) > 0 && Math.abs(v) < 1 ? v.toFixed(4) : v.toFixed(v % 1 === 0 ? 0 : 3)) : (v ?? '');

/** 通用多线折线图(SVG，无三方库)：x=epoch，多条 series；pct=true 时 y 锁定到 0~1。*/
function MetricChart({ title, xs, series, pct }: { title: string; xs: number[]; series: Series[]; pct?: boolean }) {
  const w = 520, h = 190, padL = 40, padR = 14, padT = 14, padB = 26;
  const vals = series.flatMap((s) => s.data).filter((v): v is number => typeof v === 'number' && isFinite(v));
  const hasData = xs.length > 0 && vals.length > 0;
  let ymin = pct ? 0 : Math.min(...(vals.length ? vals : [0]));
  let ymax = pct ? 1 : Math.max(...(vals.length ? vals : [1]));
  if (ymin === ymax) { ymax = ymin + 1; }
  if (!pct) { const pad = (ymax - ymin) * 0.08; ymin -= pad; ymax += pad; }
  const xmin = xs.length ? Math.min(...xs) : 0;
  const xmax = xs.length ? Math.max(...xs) : 1;
  const X = (x: number) => padL + (xmax === xmin ? 0.5 : (x - xmin) / (xmax - xmin)) * (w - padL - padR);
  const Y = (v: number) => padT + (1 - (v - ymin) / (ymax - ymin)) * (h - padT - padB);
  const grid = [0, 0.25, 0.5, 0.75, 1].map((f) => ymin + (ymax - ymin) * f);

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-bold text-slate-700 flex items-center gap-1.5"><Activity size={14} className="text-primary" /> {title}</h3>
        <div className="flex flex-wrap gap-x-3 gap-y-0.5 justify-end">
          {series.map((s) => {
            const last = [...s.data].reverse().find((v): v is number => typeof v === 'number' && isFinite(v));
            return (
              <span key={s.name} className="inline-flex items-center gap-1 text-[11px] text-slate-500">
                <span className="w-2.5 h-2.5 rounded-sm" style={{ background: s.color }} />
                {s.name} <b className="tabular-nums text-slate-700">{last != null ? (pct ? (last * 100).toFixed(1) : last.toFixed(3)) : '--'}</b>
              </span>
            );
          })}
        </div>
      </div>
      {!hasData ? (
        <div className="h-40 flex items-center justify-center text-slate-400 text-sm">暂无数据</div>
      ) : (
        <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
          {grid.map((g, i) => (
            <g key={i}>
              <line x1={padL} y1={Y(g)} x2={w - padR} y2={Y(g)} stroke="#eef2f7" strokeWidth="1" />
              <text x={padL - 5} y={Y(g) + 3} textAnchor="end" style={{ fontSize: 9, fill: '#94a3b8' }}>{pct ? `${Math.round(g * 100)}` : g.toFixed(2)}</text>
            </g>
          ))}
          <text x={X(xmin)} y={h - 8} textAnchor="start" style={{ fontSize: 9, fill: '#94a3b8' }}>e{xmin}</text>
          <text x={X(xmax)} y={h - 8} textAnchor="end" style={{ fontSize: 9, fill: '#94a3b8' }}>e{xmax}</text>
          {series.map((s) => {
            const pts: string[] = [];
            xs.forEach((x, i) => { const v = s.data[i]; if (typeof v === 'number' && isFinite(v)) pts.push(`${X(x)},${Y(v)}`); });
            return <polyline key={s.name} points={pts.join(' ')} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />;
          })}
        </svg>
      )}
    </div>
  );
}

export default function TrainingLab({ backendUrl, onBack }: Props) {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [run, setRun] = useState('');
  const [metrics, setMetrics] = useState<Metrics>({ columns: [], rows: [] });
  const [plots, setPlots] = useState<string[]>([]);
  const [log, setLog] = useState('');
  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [bigPlot, setBigPlot] = useState<string | null>(null);
  const [showTable, setShowTable] = useState(false);
  const [doneToast, setDoneToast] = useState<{ ok: boolean; map: number; model: string } | null>(null);
  const logRef = useRef<HTMLPreElement | null>(null);
  const pickedRef = useRef(false); // 用户是否手动选过 run(选过就不再自动跟随)
  const prevRunningRef = useRef(false); // 上一拍是否在训练(用于检测 完成 跳变)

  // 进页面请求一次通知权限(用户后续可在浏览器撤销)
  useEffect(() => {
    try { if ('Notification' in window && Notification.permission === 'default') void Notification.requestPermission(); } catch { /* ignore */ }
  }, []);

  const loadRun = useCallback((r: string) => {
    if (!r) return;
    fetch(`${backendUrl}/api/training/metrics?run=${encodeURIComponent(r)}`).then((x) => x.json()).then(setMetrics).catch(() => {});
    fetch(`${backendUrl}/api/training/plots?run=${encodeURIComponent(r)}`).then((x) => x.json()).then((d) => setPlots(d.plots || [])).catch(() => {});
  }, [backendUrl]);

  // 轮询：状态 + run 列表；正在训练时自动跟随当前 run，并刷新曲线/图/日志
  useEffect(() => {
    let alive = true;
    const tick = () => {
      fetch(`${backendUrl}/api/training/status`).then((r) => r.json()).then((s: LiveStatus) => {
        if (!alive) return;
        setStatus(s);
        setRun((cur) => {
          if (!pickedRef.current && s.running && s.run) return s.run; // 训练中自动跟随
          return cur;
        });
        // 检测 训练中→结束 跳变 → 桌面通知 + 持久 toast
        const wasRunning = prevRunningRef.current;
        prevRunningRef.current = !!s.running;
        if (wasRunning && !s.running && s.finished) {
          const info = { ok: !!s.success, map: s.best_map ?? 0, model: s.out_model ?? '' };
          setDoneToast(info);
          try {
            if ('Notification' in window && Notification.permission === 'granted') {
              new Notification(info.ok ? '✅ 训练完成' : '⚠️ 训练中止', { body: `${info.model} · 最佳 mAP50 ${info.map}` });
            }
          } catch { /* ignore */ }
        }
      }).catch(() => {});
      fetch(`${backendUrl}/api/training/runs`).then((r) => r.json()).then((d: { runs?: RunInfo[] }) => {
        if (!alive) return;
        const rs = d.runs || [];
        setRuns(rs);
        setRun((cur) => cur || rs.find((r) => r.has_csv)?.name || rs[0]?.name || '');
      }).catch(() => {});
      fetch(`${backendUrl}/api/training/log`).then((r) => r.json()).then((d) => alive && setLog(d.log || '')).catch(() => {});
    };
    tick();
    const t = window.setInterval(tick, 3000);
    return () => { alive = false; window.clearInterval(t); };
  }, [backendUrl]);

  useEffect(() => { loadRun(run); }, [run, loadRun]);
  // 训练中(跟随当前 run)时随轮询刷新曲线/图
  useEffect(() => {
    if (status?.running && !pickedRef.current && run) {
      const t = window.setInterval(() => loadRun(run), 3000);
      return () => window.clearInterval(t);
    }
  }, [status?.running, run, loadRun]);

  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [log]);

  const rows = metrics.rows;
  const findCol = (sub: string, excl?: string) => metrics.columns.find((c) => c.includes(sub) && (!excl || !c.includes(excl))) || '';
  const epochs = rows.map((r) => Number(r['epoch'] ?? 0));
  const ser = (col: string): (number | null)[] => rows.map((r) => { const v = col ? r[col] : undefined; return typeof v === 'number' ? v : null; });

  const perClass = metrics.per_class?.per_class;
  const isLive = !!status?.running && run === status?.run;

  return (
    <div className="h-screen overflow-y-auto bg-gradient-to-br from-slate-100 to-blue-50/40">
      <header className="bg-white/80 backdrop-blur border-b border-outline-variant/60 px-8 h-16 flex items-center justify-between sticky top-0 z-10">
        <div className="flex items-center gap-3 min-w-0">
          <button onClick={onBack} className="flex items-center gap-1 text-slate-500 hover:text-primary text-sm font-medium shrink-0"><ArrowLeft size={18} /> 返回系统</button>
          <span className="w-px h-5 bg-outline-variant" />
          <h1 className="font-display text-xl font-bold text-slate-900 flex items-center gap-2 shrink-0"><FlaskConical size={20} className="text-primary" /> 训练实验室</h1>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={run}
            onChange={(e) => { pickedRef.current = true; setRun(e.target.value); }}
            className="px-2 py-1.5 rounded-lg border border-outline-variant text-sm focus:outline-none focus:border-primary max-w-[280px]"
          >
            {runs.length === 0 && <option value="">暂无实验</option>}
            {runs.map((r) => (
              <option key={r.name} value={r.name}>{r.name}{r.has_csv ? '' : '（无数据）'} · {fmtTime(r.mtime)}</option>
            ))}
          </select>
          <button onClick={() => loadRun(run)} title="刷新" className="p-2 rounded-full hover:bg-slate-100 text-slate-500"><RefreshCw size={18} /></button>
        </div>
      </header>

      <div className="max-w-[1180px] mx-auto p-8 space-y-6">
        {/* 概览 */}
        <div className="card p-5">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
            <span className={`inline-flex items-center gap-1.5 text-sm font-bold ${isLive ? 'text-primary' : 'text-slate-500'}`}>
              <span className={`w-2 h-2 rounded-full ${isLive ? 'bg-primary animate-pulse' : 'bg-slate-400'}`} />
              {isLive ? `训练中 · Epoch ${status?.epoch ?? 0}/${status?.total_epochs ?? '--'}` : (status?.finished && run === status?.run ? (status?.success ? '已完成' : '已中止') : '查看历史实验')}
            </span>
            <span className="text-sm text-slate-600">数据集 <b>{status?.project ?? '—'}</b></span>
            <span className="text-sm text-slate-600">基础权重 <b>{String(status?.base ?? '—').replace('models/', '')}</b></span>
            <span className="text-sm text-slate-600">产出 <b>{status?.out_model ?? '—'}</b>{status?.is_active ? '（实时生效中）' : ''}</span>
            {typeof status?.best_map === 'number' && <span className="text-sm text-slate-600">最佳 mAP50 <b className="text-primary">{status.best_map}</b></span>}
            {isLive && <span className="text-sm text-slate-600">已用 <b>{fmtClock(status?.elapsed)}</b></span>}
            <span className="text-xs text-slate-400 ml-auto">{rows.length} epoch 数据 · run <code>{run || '—'}</code></span>
          </div>
          {isLive && (
            <div className="h-2 rounded-full bg-slate-200 overflow-hidden mt-3">
              <div className="h-full rounded-full bg-gradient-to-r from-primary to-cyan-400 transition-all" style={{ width: `${status?.total_epochs ? ((status?.epoch ?? 0) / status.total_epochs) * 100 : 0}%` }} />
            </div>
          )}
        </div>

        {/* 实时曲线 */}
        <div className="grid md:grid-cols-2 gap-5">
          <MetricChart title="mAP" xs={epochs} pct series={[
            { name: 'mAP50', color: '#3b82f6', data: ser(findCol('mAP50', 'mAP50-95')) },
            { name: 'mAP50-95', color: '#06b6d4', data: ser(findCol('mAP50-95')) },
          ]} />
          <MetricChart title="精确率 / 召回率" xs={epochs} pct series={[
            { name: 'precision', color: '#22c55e', data: ser(findCol('precision')) },
            { name: 'recall', color: '#f59e0b', data: ser(findCol('recall')) },
          ]} />
          <MetricChart title="训练损失" xs={epochs} series={[
            { name: 'box', color: '#ef4444', data: ser(findCol('train/box')) },
            { name: 'cls', color: '#a855f7', data: ser(findCol('train/cls')) },
            { name: 'dfl', color: '#f59e0b', data: ser(findCol('train/dfl')) },
          ]} />
          <MetricChart title="验证损失" xs={epochs} series={[
            { name: 'box', color: '#ef4444', data: ser(findCol('val/box')) },
            { name: 'cls', color: '#a855f7', data: ser(findCol('val/cls')) },
            { name: 'dfl', color: '#f59e0b', data: ser(findCol('val/dfl')) },
          ]} />
        </div>

        {/* 各类别 mAP */}
        {perClass && Object.keys(perClass).length > 0 && (
          <div className="card p-5">
            <h3 className="text-sm font-bold text-slate-700 mb-3 flex items-center gap-1.5">
              <Activity size={14} className="text-primary" /> 各类别 mAP@50
              <span className="font-normal text-slate-400 ml-1">epoch {metrics.per_class?.epoch ?? '--'} · 总 mAP50 {((metrics.per_class?.map50 ?? 0) * 100).toFixed(1)}%</span>
            </h3>
            <div className="space-y-1.5">
              {Object.entries(perClass as Record<string, ClassMetric>).map(([name, v]) => (
                <div key={name} className="flex items-center gap-3">
                  <span className="w-32 text-sm text-slate-600 shrink-0 truncate">{name}</span>
                  <div className="flex-1 h-3.5 rounded-full bg-slate-100 overflow-hidden">
                    <div className="h-full rounded-full bg-gradient-to-r from-primary to-cyan-400" style={{ width: `${Math.min(100, Math.round((v.map50 ?? 0) * 100))}%` }} />
                  </div>
                  <span className="w-12 text-right text-sm font-bold text-slate-700 tabular-nums">{((v.map50 ?? 0) * 100).toFixed(1)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 生成的图表(混淆矩阵 / PR 曲线 / 批次预览…) */}
        {plots.length > 0 && (
          <div className="card p-5">
            <h3 className="text-sm font-bold text-slate-700 mb-3 flex items-center gap-1.5"><ImagesIcon size={14} className="text-primary" /> 生成图表 <span className="font-normal text-slate-400">({plots.length})</span></h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
              {plots.map((p) => (
                <button key={p} onClick={() => setBigPlot(p)} className="group rounded-lg overflow-hidden border border-outline-variant/50 bg-white hover:border-primary transition-colors text-left">
                  <img src={`${backendUrl}/api/training/plot_file?run=${encodeURIComponent(run)}&name=${encodeURIComponent(p)}`} alt={p} className="w-full h-28 object-contain bg-slate-50" loading="lazy" />
                  <div className="text-[11px] text-slate-500 px-2 py-1 truncate group-hover:text-primary">{p}</div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* 逐 epoch 数值表 */}
        {rows.length > 0 && (
          <div className="card p-5">
            <button onClick={() => setShowTable((s) => !s)} className="text-sm font-bold text-slate-700 flex items-center gap-1.5">
              <Table2 size={14} className="text-primary" /> 逐 epoch 数值表（{rows.length} 行 × {metrics.columns.length} 列）<span className="text-xs font-normal text-primary ml-1">{showTable ? '收起' : '展开'}</span>
            </button>
            {showTable && (
              <div className="mt-3 overflow-auto max-h-[420px] border border-outline-variant/40 rounded-lg">
                <table className="text-[11px] tabular-nums whitespace-nowrap">
                  <thead className="sticky top-0 bg-slate-50">
                    <tr>{metrics.columns.map((c) => <th key={c} className="px-2 py-1.5 text-left font-bold text-slate-500 border-b border-outline-variant/40">{c}</th>)}</tr>
                  </thead>
                  <tbody>
                    {rows.map((r, i) => (
                      <tr key={i} className={i % 2 ? 'bg-slate-50/50' : ''}>
                        {metrics.columns.map((c) => <td key={c} className="px-2 py-1 text-slate-600 border-b border-outline-variant/20">{num(r[c])}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* 实时日志 */}
        <div className="card p-5">
          <h3 className="text-sm font-bold text-slate-700 mb-3 flex items-center gap-1.5"><Terminal size={14} className="text-primary" /> 训练日志{isLive && <span className="text-[11px] font-normal text-primary">· 实时</span>}</h3>
          <pre ref={logRef} className="text-[11px] leading-relaxed bg-slate-900 text-slate-300 rounded-lg p-3 max-h-80 overflow-auto whitespace-pre-wrap">{log || '（暂无日志；开始训练后这里会实时刷新）'}</pre>
        </div>
      </div>

      {/* 训练完成通知(running→done 时弹出，可手动关闭) */}
      {doneToast && (
        <div className={`fixed bottom-6 right-6 z-50 max-w-sm rounded-xl border shadow-2xl px-4 py-3 flex items-start gap-3 ${doneToast.ok ? 'bg-emerald-50 border-emerald-200' : 'bg-amber-50 border-amber-200'}`}>
          {doneToast.ok ? <CheckCircle2 size={20} className="text-emerald-600 shrink-0 mt-0.5" /> : <AlertTriangle size={20} className="text-amber-600 shrink-0 mt-0.5" />}
          <div className="min-w-0">
            <div className={`text-sm font-bold ${doneToast.ok ? 'text-emerald-700' : 'text-amber-700'}`}>{doneToast.ok ? '训练完成' : '训练中止'}</div>
            <div className="text-xs text-slate-600 mt-0.5">{doneToast.model || '模型'} · 最佳 mAP50 <b>{doneToast.map}</b>{doneToast.ok ? '；可到「模型与数据」设为生效' : ''}</div>
          </div>
          <button onClick={() => setDoneToast(null)} className="text-slate-400 hover:text-slate-600 shrink-0"><X size={16} /></button>
        </div>
      )}

      {/* 图表放大 */}
      {bigPlot && (
        <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-8" onClick={() => setBigPlot(null)}>
          <button className="absolute top-5 right-5 text-white/80 hover:text-white" onClick={() => setBigPlot(null)}><X size={28} /></button>
          <img src={`${backendUrl}/api/training/plot_file?run=${encodeURIComponent(run)}&name=${encodeURIComponent(bigPlot)}`} alt={bigPlot} className="max-w-full max-h-full object-contain rounded-lg shadow-2xl" onClick={(e) => e.stopPropagation()} />
        </div>
      )}
    </div>
  );
}
