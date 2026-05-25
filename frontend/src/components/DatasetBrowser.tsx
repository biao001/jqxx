import { type ChangeEvent, type MouseEvent as ReactMouseEvent, useCallback, useEffect, useRef, useState } from 'react';
import { ArrowLeft, ChevronLeft, ChevronRight, ImagePlus, Save, SquarePen, Trash2, X } from 'lucide-react';

interface Box {
  cls: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
}
interface Item {
  name: string;
  boxes: Box[];
  classes: number[];
  tiny: boolean;
  huge: boolean;
}
interface Props {
  open: boolean;
  onClose: () => void;
  backendUrl: string;
}

const PALETTE = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];
const boxColor = (cls: number) => PALETTE[cls % PALETTE.length];
const boxStyle = (b: Box) => ({
  left: `${(b.cx - b.w / 2) * 100}%`,
  top: `${(b.cy - b.h / 2) * 100}%`,
  width: `${b.w * 100}%`,
  height: `${b.h * 100}%`,
});

export default function DatasetBrowser({ open, onClose, backendUrl }: Props) {
  const [split, setSplit] = useState('train');
  const [cls, setCls] = useState<number | ''>('');
  const [issue, setIssue] = useState('');
  const [page, setPage] = useState(0);
  const [data, setData] = useState<{ total: number; names: string[]; items: Item[]; page_size: number } | null>(null);
  const [sel, setSel] = useState<Item | null>(null);
  const [selIdx, setSelIdx] = useState(-1);
  const [saving, setSaving] = useState('');
  const [drawCls, setDrawCls] = useState(0);
  const [draw, setDraw] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const pendingRef = useRef<'first' | 'last' | null>(null);
  const imgWrapRef = useRef<HTMLDivElement | null>(null);

  const clone = (it: Item): Item => ({ ...it, boxes: it.boxes.map((b) => ({ ...b })) });
  const imgUrl = (name: string) => `${backendUrl}/api/dataset/image_file?split=${split}&name=${encodeURIComponent(name)}`;

  const load = useCallback(() => {
    const q = new URLSearchParams({ split, page: String(page), page_size: '24' });
    if (cls !== '') q.set('cls', String(cls));
    if (issue) q.set('issue', issue);
    fetch(`${backendUrl}/api/dataset/images?${q}`).then((r) => r.json()).then(setData).catch(() => setData(null));
  }, [backendUrl, split, cls, issue, page]);

  useEffect(() => { if (open) load(); }, [open, load]);
  useEffect(() => { setPage(0); }, [split, cls, issue]);

  const openItem = (it: Item, idx: number) => { setSel(clone(it)); setSelIdx(idx); setDraw(null); };

  const go = (dir: number) => {
    if (!data) return;
    const tp = Math.max(1, Math.ceil(data.total / data.page_size));
    const ni = selIdx + dir;
    if (ni >= 0 && ni < data.items.length) openItem(data.items[ni], ni);
    else if (dir > 0 && page < tp - 1) { pendingRef.current = 'first'; setPage((p) => p + 1); }
    else if (dir < 0 && page > 0) { pendingRef.current = 'last'; setPage((p) => p - 1); }
  };

  // 翻页后若在编辑态，自动选首/末张(支持跨页"下一张")
  useEffect(() => {
    if (!data || pendingRef.current == null) return;
    const items = data.items;
    if (items.length) openItem(items[pendingRef.current === 'first' ? 0 : items.length - 1], pendingRef.current === 'first' ? 0 : items.length - 1);
    pendingRef.current = null;
  }, [data]);

  // 方向键切换上一张/下一张
  useEffect(() => {
    if (!sel) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight') go(1);
      else if (e.key === 'ArrowLeft') go(-1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const uploadSingle = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const fd = new FormData();
    fd.append('file', f);
    fd.append('split', split);
    fetch(`${backendUrl}/api/dataset/image_upload`, { method: 'POST', body: fd })
      .then((r) => r.json())
      .then((d) => { if (d.name) { setSel({ name: d.name, boxes: [], classes: [], tiny: false, huge: false }); setSelIdx(-1); load(); } });
    e.target.value = '';
  };

  // 鼠标画框(归一化)
  const ptr = (e: ReactMouseEvent) => {
    const el = imgWrapRef.current;
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) };
  };
  const onDown = (e: ReactMouseEvent) => { const p = ptr(e); if (p) setDraw({ x0: p.x, y0: p.y, x1: p.x, y1: p.y }); };
  const onMove = (e: ReactMouseEvent) => { if (!draw) return; const p = ptr(e); if (p) setDraw({ ...draw, x1: p.x, y1: p.y }); };
  const onUp = () => {
    if (draw && sel) {
      const w = Math.abs(draw.x1 - draw.x0);
      const h = Math.abs(draw.y1 - draw.y0);
      if (w > 0.02 && h > 0.02) {
        setSel({ ...sel, boxes: [...sel.boxes, { cls: drawCls, cx: (draw.x0 + draw.x1) / 2, cy: (draw.y0 + draw.y1) / 2, w, h }] });
      }
    }
    setDraw(null);
  };

  if (!open) return null;
  const names = data?.names || [];
  const totalPages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1;

  const saveLabels = () => {
    if (!sel) return;
    setSaving('保存中…');
    fetch(`${backendUrl}/api/dataset/labels`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ split, name: sel.name, boxes: sel.boxes }),
    }).then((r) => r.json()).then(() => { setSaving('✅ 已保存'); setTimeout(() => setSaving(''), 1500); load(); });
  };
  const delImage = () => {
    if (!sel) return;
    fetch(`${backendUrl}/api/dataset/image?split=${split}&name=${encodeURIComponent(sel.name)}`, { method: 'DELETE' })
      .then(() => { setSel(null); load(); });
  };

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={onClose}>
      <div className="w-[1100px] h-[720px] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden" onClick={(e) => e.stopPropagation()}>
        {/* 头部 + 筛选 */}
        <div className="px-5 h-14 flex items-center gap-3 border-b border-outline-variant shrink-0">
          {sel ? (
            <div className="flex items-center gap-2">
              <button onClick={() => setSel(null)} className="flex items-center gap-1 text-slate-500 hover:text-primary text-sm"><ArrowLeft size={16} /> 返回网格</button>
              <span className="w-px h-4 bg-outline-variant" />
              <button onClick={() => go(-1)} title="上一张 (←)" className="p-1 rounded hover:bg-slate-100"><ChevronLeft size={18} /></button>
              <button onClick={() => go(1)} title="下一张 (→)" className="p-1 rounded hover:bg-slate-100"><ChevronRight size={18} /></button>
              <span className="text-xs text-outline">← / → 切换</span>
            </div>
          ) : (
            <h3 className="font-display font-bold text-slate-800">数据集标注浏览</h3>
          )}
          {!sel && (
            <div className="flex items-center gap-2 text-sm ml-2">
              <select value={split} onChange={(e) => setSplit(e.target.value)} className="border border-outline-variant rounded-lg px-2 py-1">
                <option value="train">train</option><option value="valid">valid</option><option value="test">test</option>
              </select>
              <select value={cls} onChange={(e) => setCls(e.target.value === '' ? '' : Number(e.target.value))} className="border border-outline-variant rounded-lg px-2 py-1">
                <option value="">全部类别</option>
                {names.map((n, i) => <option key={n} value={i}>{n}</option>)}
              </select>
              <select value={issue} onChange={(e) => setIssue(e.target.value)} className="border border-outline-variant rounded-lg px-2 py-1">
                <option value="">全部质量</option><option value="tiny">含极小框</option><option value="huge">含超大框</option><option value="empty">空标注</option>
              </select>
              {data && <span className="text-xs text-outline">共 {data.total} 张</span>}
              <label className="ml-1 inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-primary text-white cursor-pointer text-xs font-medium hover:brightness-110">
                <ImagePlus size={14} /> 上传单张
                <input type="file" accept="image/*" className="hidden" onChange={uploadSingle} />
              </label>
            </div>
          )}
          <button onClick={onClose} className="ml-auto p-1.5 rounded-full hover:bg-surface-container text-slate-500"><X size={20} /></button>
        </div>

        {sel ? (
          /* 编辑视图 */
          <div className="flex-1 flex min-h-0">
            <div className="flex-1 bg-slate-900 flex items-center justify-center p-4 min-w-0">
              <div
                ref={imgWrapRef}
                className="relative max-h-full max-w-full cursor-crosshair select-none"
                onMouseDown={onDown}
                onMouseMove={onMove}
                onMouseUp={onUp}
                onMouseLeave={onUp}
              >
                <img src={imgUrl(sel.name)} alt={sel.name} draggable={false} className="max-h-[600px] max-w-full object-contain block pointer-events-none" />
                {sel.boxes.map((b, i) => (
                  <div key={i} className="absolute border-2 pointer-events-none" style={{ ...boxStyle(b), borderColor: boxColor(b.cls) }}>
                    <span className="absolute -top-5 left-0 px-1 text-[10px] font-bold text-white whitespace-nowrap" style={{ background: boxColor(b.cls) }}>
                      {names[b.cls] ?? b.cls}
                    </span>
                  </div>
                ))}
                {draw && (
                  <div
                    className="absolute border-2 border-dashed border-white/90 bg-white/10 pointer-events-none"
                    style={{ left: `${Math.min(draw.x0, draw.x1) * 100}%`, top: `${Math.min(draw.y0, draw.y1) * 100}%`, width: `${Math.abs(draw.x1 - draw.x0) * 100}%`, height: `${Math.abs(draw.y1 - draw.y0) * 100}%` }}
                  />
                )}
              </div>
            </div>
            <div className="w-[280px] border-l border-outline-variant flex flex-col shrink-0">
              <div className="p-3 text-xs text-outline truncate border-b border-outline-variant">{sel.name}</div>
              <div className="px-3 py-2 border-b border-outline-variant">
                <div className="flex items-center gap-2 text-xs">
                  <SquarePen size={14} className="text-primary shrink-0" />
                  <span className="text-slate-500 shrink-0">画框类别</span>
                  <select value={drawCls} onChange={(e) => setDrawCls(Number(e.target.value))} className="flex-1 border border-outline-variant rounded px-1 py-0.5 min-w-0">
                    {names.map((n, i) => <option key={n} value={i}>{n}</option>)}
                  </select>
                </div>
                <p className="text-[11px] text-outline-variant mt-1">在左侧图上按住拖拽即可新建框</p>
              </div>
              <div className="flex-1 overflow-y-auto p-3 space-y-2">
                {sel.boxes.length === 0 && <p className="text-sm text-slate-400">无标注框</p>}
                {sel.boxes.map((b, i) => (
                  <div key={i} className="flex items-center gap-2 p-2 rounded-lg border border-outline-variant/60">
                    <span className="w-3 h-3 rounded-sm shrink-0" style={{ background: boxColor(b.cls) }} />
                    <select
                      value={b.cls}
                      onChange={(e) => setSel({ ...sel, boxes: sel.boxes.map((x, j) => (j === i ? { ...x, cls: Number(e.target.value) } : x)) })}
                      className="flex-1 text-sm border border-outline-variant rounded px-1 py-0.5 min-w-0"
                    >
                      {names.map((n, ci) => <option key={n} value={ci}>{n}</option>)}
                    </select>
                    <button onClick={() => setSel({ ...sel, boxes: sel.boxes.filter((_, j) => j !== i) })} className="text-red-500 hover:text-red-600 shrink-0"><Trash2 size={15} /></button>
                  </div>
                ))}
              </div>
              <div className="p-3 border-t border-outline-variant flex gap-2">
                <button onClick={saveLabels} className="btn-primary flex-1 py-2 text-sm"><Save size={15} /> 保存{saving && ` ${saving}`}</button>
                <button onClick={delImage} className="px-3 rounded-lg border border-red-200 text-red-500 hover:bg-red-50"><Trash2 size={16} /></button>
              </div>
            </div>
          </div>
        ) : (
          /* 网格 */
          <>
            <div className="flex-1 overflow-y-auto p-4 grid grid-cols-4 gap-3 content-start">
              {data?.items.map((it, i) => (
                <div key={it.name} onClick={() => openItem(it, i)} className="group cursor-pointer rounded-lg overflow-hidden border border-outline-variant/60 hover:border-primary bg-slate-900 relative">
                  <div className="relative">
                    <img src={imgUrl(it.name)} alt={it.name} loading="lazy" className="w-full h-32 object-cover" />
                    {it.boxes.map((b, i) => (
                      <div key={i} className="absolute border-2 pointer-events-none" style={{ ...boxStyle(b), borderColor: boxColor(b.cls) }} />
                    ))}
                  </div>
                  <div className="absolute top-1 right-1 flex gap-1">
                    {it.tiny && <span className="px-1 rounded bg-amber-500 text-white text-[9px] font-bold">极小</span>}
                    {it.huge && <span className="px-1 rounded bg-red-500 text-white text-[9px] font-bold">超大</span>}
                  </div>
                  <div className="absolute bottom-1 left-1 flex flex-wrap gap-0.5">
                    {it.classes.map((c) => <span key={c} className="px-1 rounded text-white text-[9px] font-bold" style={{ background: boxColor(c) }}>{names[c] ?? c}</span>)}
                  </div>
                </div>
              ))}
              {data && data.items.length === 0 && <div className="col-span-4 text-center text-slate-400 py-12">无匹配图片</div>}
            </div>
            <div className="h-12 flex items-center justify-center gap-4 border-t border-outline-variant shrink-0">
              <button disabled={page <= 0} onClick={() => setPage((p) => p - 1)} className="p-1.5 rounded hover:bg-slate-100 disabled:opacity-30"><ChevronLeft size={18} /></button>
              <span className="text-sm text-slate-600">第 {page + 1} / {totalPages} 页</span>
              <button disabled={page >= totalPages - 1} onClick={() => setPage((p) => p + 1)} className="p-1.5 rounded hover:bg-slate-100 disabled:opacity-30"><ChevronRight size={18} /></button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
