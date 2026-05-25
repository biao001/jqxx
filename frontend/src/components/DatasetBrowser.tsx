import { type ChangeEvent, type DragEvent as ReactDragEvent, type MouseEvent as ReactMouseEvent, type WheelEvent as ReactWheelEvent, useCallback, useEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowLeftRight, Check, ChevronLeft, ChevronRight, Copy, Film, ImagePlus, Redo2, Save, Shuffle, Sparkles, SquarePen, Trash2, Undo2, X } from 'lucide-react';

interface Box {
  cls: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
}
interface Item {
  name: string;
  split?: string; // 该图所在 split(在"全部"视图下各项不同)
  boxes: Box[];
  classes: number[];
  tiny: boolean;
  huge: boolean;
}
interface Props {
  open: boolean;
  onClose: () => void;
  backendUrl: string;
  dataset?: string;
}

const PALETTE = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];
const boxColor = (cls: number) => PALETTE[cls % PALETTE.length];
const boxStyle = (b: Box) => ({
  left: `${(b.cx - b.w / 2) * 100}%`,
  top: `${(b.cy - b.h / 2) * 100}%`,
  width: `${b.w * 100}%`,
  height: `${b.h * 100}%`,
});

export default function DatasetBrowser({ open, onClose, backendUrl, dataset = 'reckless_mapped' }: Props) {
  const dsq = `dataset=${encodeURIComponent(dataset)}`;
  const [split, setSplit] = useState('all');
  const [cls, setCls] = useState<number | ''>('');
  const [issue, setIssue] = useState('');
  const [page, setPage] = useState(0);
  const [data, setData] = useState<{ total: number; names: string[]; items: Item[]; page_size: number } | null>(null);
  const [sel, setSel] = useState<Item | null>(null);
  const [selIdx, setSelIdx] = useState(-1);
  const [saving, setSaving] = useState('');
  const [drawCls, setDrawCls] = useState(0);
  const [draw, setDraw] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const [selBox, setSelBox] = useState(-1); // 当前选中的框索引(可拖拽/缩放/改类/删除)
  const [drag, setDrag] = useState<{ idx: number; handle: 'move' | 'nw' | 'ne' | 'sw' | 'se'; ox: number; oy: number; box0: Box } | null>(null);
  const [aiMsg, setAiMsg] = useState('');
  const [busy, setBusy] = useState(''); // 网格级提示(批量上传/批量预标注)
  const [dragOver, setDragOver] = useState(false);
  const [everySec, setEverySec] = useState(1);
  const [maxFrames, setMaxFrames] = useState(60);
  const [aiConf, setAiConf] = useState(0.25);
  const [selMode, setSelMode] = useState(false); // 网格多选删除模式
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const undoRef = useRef<Box[][]>([]);
  const redoRef = useRef<Box[][]>([]);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [info, setInfo] = useState<{ splits?: Record<string, { images: number; labels: number; annotated: number }>; classes?: { name: string; count: number }[] } | null>(null);
  const pendingRef = useRef<'first' | 'last' | null>(null);
  const delIdxRef = useRef<number | null>(null); // 删除后停留在预览态、自动定位到下一张
  const origRef = useRef<string>(''); // 打开时的标注快照，用于未保存检测
  const panRef = useRef<{ px: number; py: number; ox: number; oy: number } | null>(null);
  const imgWrapRef = useRef<HTMLDivElement | null>(null);

  const clone = (it: Item): Item => ({ ...it, boxes: it.boxes.map((b) => ({ ...b })) });
  const dirty = !!sel && origRef.current !== JSON.stringify(sel.boxes);
  const confirmDiscard = () => !dirty || window.confirm('当前图片有未保存的标注改动，确定放弃?');

  // 撤销/重做：在每次离散修改前调用 snapshot() 存档
  const snapshot = () => { if (sel) { undoRef.current.push(sel.boxes.map((b) => ({ ...b }))); redoRef.current = []; } };
  const undo = () => {
    if (!sel || !undoRef.current.length) return;
    redoRef.current.push(sel.boxes.map((b) => ({ ...b })));
    setSel({ ...sel, boxes: undoRef.current.pop() as Box[] }); setSelBox(-1);
  };
  const redo = () => {
    if (!sel || !redoRef.current.length) return;
    undoRef.current.push(sel.boxes.map((b) => ({ ...b })));
    setSel({ ...sel, boxes: redoRef.current.pop() as Box[] }); setSelBox(-1);
  };
  // 单图操作用「该图自己的 split」(全部视图下 split='all' 不是真实目录)
  const itemSplit = (sp?: string) => (sp && sp !== 'all' ? sp : split === 'all' ? 'train' : split);
  const imgUrl = (name: string, sp?: string) => `${backendUrl}/api/dataset/image_file?split=${itemSplit(sp)}&name=${encodeURIComponent(name)}&${dsq}`;

  const load = useCallback(() => {
    const q = new URLSearchParams({ split, page: String(page), page_size: '24', dataset });
    if (cls !== '') q.set('cls', String(cls));
    if (issue) q.set('issue', issue);
    fetch(`${backendUrl}/api/dataset/images?${q}`).then((r) => r.json()).then(setData).catch(() => setData(null));
  }, [backendUrl, split, cls, issue, page, dataset]);

  useEffect(() => { if (open) load(); }, [open, load]);
  useEffect(() => { setPage(0); setPicked(new Set()); }, [split, cls, issue]);

  // 统计(已标/未标/各类)随数据变化刷新
  useEffect(() => {
    if (!open) return;
    fetch(`${backendUrl}/api/dataset/info?dataset=${encodeURIComponent(dataset)}`).then((r) => r.json()).then((d) => setInfo(d.active)).catch(() => {});
  }, [open, dataset, data, backendUrl]);

  const openItem = (it: Item, idx: number) => {
    const c = clone(it);
    origRef.current = JSON.stringify(c.boxes);
    setSel(c); setSelIdx(idx); setDraw(null); setSelBox(-1); setAiMsg(''); setZoom(1); setPan({ x: 0, y: 0 });
    undoRef.current = []; redoRef.current = [];
  };

  const go = (dir: number) => {
    if (!data) return;
    if (sel && !confirmDiscard()) return;
    const tp = Math.max(1, Math.ceil(data.total / data.page_size));
    const ni = selIdx + dir;
    if (ni >= 0 && ni < data.items.length) openItem(data.items[ni], ni);
    else if (dir > 0 && page < tp - 1) { pendingRef.current = 'first'; setPage((p) => p + 1); }
    else if (dir < 0 && page > 0) { pendingRef.current = 'last'; setPage((p) => p - 1); }
  };

  // —— AI 预标注 / 复制上一张 / 跨 split 移动 ——
  const aiAutolabel = () => {
    if (!sel) return;
    if (sel.boxes.length && !window.confirm('AI 预标注将覆盖当前已有的框，继续?')) return;
    snapshot();
    setAiMsg('AI 识别中…');
    fetch(`${backendUrl}/api/dataset/autolabel`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset, split: itemSplit(sel.split), name: sel.name, conf: aiConf }),
    }).then((r) => r.json()).then((d: { boxes?: Box[]; note?: string }) => {
      if (d.note) { setAiMsg(`⚠️ ${d.note}`); return; }
      setSel((s) => (s ? { ...s, boxes: (d.boxes || []).map((b) => ({ ...b })) } : s));
      setSelBox(-1);
      setAiMsg(`✅ 预标注 ${d.boxes?.length || 0} 个框，请检查微调后保存`);
    }).catch(() => setAiMsg('❌ 预标注失败'));
  };

  const copyPrev = () => {
    if (!sel || !data || selIdx <= 0) return;
    const prev = data.items[selIdx - 1];
    if (!prev) return;
    snapshot();
    setSel({ ...sel, boxes: prev.boxes.map((b) => ({ ...b })) });
    setSelBox(-1);
  };

  const moveTo = (toSplit: string) => {
    if (!sel) return;
    fetch(`${backendUrl}/api/dataset/move`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset, split: itemSplit(sel.split), name: sel.name, to_split: toSplit }),
    }).then((r) => r.json()).then((d) => {
      if (d.moved) { if (selIdx >= 0) delIdxRef.current = selIdx; else setSel(null); load(); }
    });
  };

  // 翻页后若在编辑态，自动选首/末张(支持跨页"下一张")
  useEffect(() => {
    if (!data || pendingRef.current == null) return;
    const items = data.items;
    if (items.length) openItem(items[pendingRef.current === 'first' ? 0 : items.length - 1], pendingRef.current === 'first' ? 0 : items.length - 1);
    pendingRef.current = null;
  }, [data]);

  // 删除后停留在预览态：自动定位到原位置的下一张(删除后该索引即为下一张)
  useEffect(() => {
    if (!data || delIdxRef.current == null) return;
    const idx = delIdxRef.current;
    delIdxRef.current = null;
    if (data.items.length === 0) {
      // 本页删空：退回上一页末张，否则回到网格
      if (page > 0) { pendingRef.current = 'last'; setPage((p) => p - 1); }
      else setSel(null);
      return;
    }
    const ni = Math.min(idx, data.items.length - 1);
    openItem(data.items[ni], ni);
  }, [data]);

  // 快捷键：← / → 切换，数字键选类别(选中框时改其类别)，Del 删框，S 保存，D 删图
  useEffect(() => {
    if (!sel) return;
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      const names = data?.names || [];
      if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) { e.preventDefault(); if (e.shiftKey) redo(); else undo(); return; }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || e.key === 'Y')) { e.preventDefault(); redo(); return; }
      if (e.key === 'ArrowRight') go(1);
      else if (e.key === 'ArrowLeft') go(-1);
      else if (/^[1-9]$/.test(e.key)) {
        const ci = Number(e.key) - 1;
        if (ci < names.length) {
          if (selBox >= 0) { snapshot(); setSel((s) => (s ? { ...s, boxes: s.boxes.map((b, j) => (j === selBox ? { ...b, cls: ci } : b)) } : s)); }
          else setDrawCls(ci);
        }
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && selBox >= 0) {
        e.preventDefault();
        snapshot();
        setSel((s) => (s ? { ...s, boxes: s.boxes.filter((_, j) => j !== selBox) } : s));
        setSelBox(-1);
      } else if (e.key === 's' || e.key === 'S') { e.preventDefault(); saveLabels(); }
      else if (e.key === 'd' || e.key === 'D') { e.preventDefault(); if (window.confirm('删除当前图片?')) delImage(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const uploadFiles = async (files: File[]) => {
    if (!files.length) return;
    let ok = 0;
    let lastName = '';
    for (let i = 0; i < files.length; i++) {
      setBusy(`上传中 ${i + 1}/${files.length}…`);
      const fd = new FormData();
      // 一律上传到"未划分"暂存区，避免误以为默认进了 train；之后用"自动划分"分配
      fd.append('file', files[i]); fd.append('split', 'unassigned'); fd.append('dataset', dataset);
      try {
        const d = await (await fetch(`${backendUrl}/api/dataset/image_upload`, { method: 'POST', body: fd })).json();
        if (d.name) { ok++; lastName = d.name; }
      } catch { /* 跳过失败的单张 */ }
    }
    setBusy(`✅ 上传 ${ok}/${files.length} 张到「未划分」，可标注后点“自动划分”`); setTimeout(() => setBusy(''), 4000);
    if (files.length === 1 && lastName) { origRef.current = '[]'; setSel({ name: lastName, split: 'unassigned', boxes: [], classes: [], tiny: false, huge: false }); setSelIdx(-1); setZoom(1); setPan({ x: 0, y: 0 }); }
    if (split !== 'unassigned') setSplit('unassigned'); else load();
  };

  const uploadImages = (e: ChangeEvent<HTMLInputElement>) => {
    const files: File[] = e.target.files ? Array.from(e.target.files) : [];
    e.target.value = '';
    uploadFiles(files);
  };

  const onDrop = (e: ReactDragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const files = (Array.from(e.dataTransfer.files) as File[]).filter((f) => f.type.startsWith('image/'));
    uploadFiles(files);
  };

  const redistribute = () => {
    const v = window.prompt('验证集比例 %（valid）：', '10');
    if (v == null) return;
    const t = window.prompt('测试集比例 %（test）：', '10');
    if (t == null) return;
    const val_frac = (Number(v) || 0) / 100;
    const test_frac = (Number(t) || 0) / 100;
    if (!window.confirm(`将全部图片(含未划分)随机划分为 train / ${v}% valid / ${t}% test，继续?`)) return;
    setBusy('重新划分中…');
    fetch(`${backendUrl}/api/dataset/redistribute`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset, val_frac, test_frac }),
    }).then((r) => r.json()).then((d: { detail?: string; total?: number; moved?: number }) => {
      setBusy(d.detail ? `❌ ${d.detail}` : `✅ 已划分 ${d.total} 张（移动 ${d.moved} 张）`);
      setTimeout(() => setBusy(''), 4000); load();
    }).catch(() => { setBusy('❌ 划分失败'); setTimeout(() => setBusy(''), 3000); });
  };

  // —— 网格多选批量删除 ——
  const togglePick = (name: string) => {
    setPicked((s) => { const n = new Set(s); if (n.has(name)) n.delete(name); else n.add(name); return n; });
  };
  const exitSelMode = () => { setSelMode(false); setPicked(new Set()); };
  const pickAll = () => { setPicked(new Set((data?.items || []).map((it) => it.name))); };
  const deletePicked = () => {
    const names = Array.from(picked);
    if (!names.length) return;
    if (!window.confirm(`确定删除选中的 ${names.length} 张图片(连标注)?不可恢复。`)) return;
    setBusy(`删除 ${names.length} 张中…`);
    fetch(`${backendUrl}/api/dataset/delete_batch`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset, split, names }),
    }).then((r) => r.json()).then((d: { deleted?: number }) => {
      setBusy(`✅ 已删除 ${d.deleted ?? 0} 张`); setTimeout(() => setBusy(''), 3000);
      exitSelMode(); load();
    }).catch(() => { setBusy('❌ 删除失败'); setTimeout(() => setBusy(''), 3000); });
  };

  const uploadVideo = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = '';
    if (!f) return;
    setBusy('视频切帧中…(请稍候)');
    const fd = new FormData();
    fd.append('file', f); fd.append('split', 'unassigned'); fd.append('dataset', dataset);
    fd.append('every_sec', String(everySec)); fd.append('max_frames', String(maxFrames));
    fetch(`${backendUrl}/api/dataset/video_frames`, { method: 'POST', body: fd })
      .then((r) => r.json())
      .then((d: { detail?: string; saved?: number }) => {
        setBusy(d.detail ? `❌ ${d.detail}` : `✅ 切出 ${d.saved} 帧到「未划分」，可标注后“自动划分”`);
        setTimeout(() => setBusy(''), 4000);
        if (split !== 'unassigned') setSplit('unassigned'); else load();
      })
      .catch(() => { setBusy('❌ 切帧失败'); setTimeout(() => setBusy(''), 3000); });
  };

  const batchAutolabel = () => {
    if (!window.confirm(`对 ${split} 下尚未标注的图片批量 AI 预标注?(用当前生效模型)`)) return;
    setBusy('批量预标注中…(图多时可能较久)');
    fetch(`${backendUrl}/api/dataset/autolabel_batch`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset, split, only_empty: true, conf: aiConf }),
    }).then((r) => r.json()).then((d: { note?: string; labeled?: number; processed?: number; boxes?: number }) => {
      setBusy(d.note ? `⚠️ ${d.note}` : `✅ 预标注 ${d.labeled}/${d.processed} 张，共 ${d.boxes} 框`);
      setTimeout(() => setBusy(''), 5000); load();
    }).catch(() => { setBusy('❌ 批量预标注失败'); setTimeout(() => setBusy(''), 3000); });
  };

  // —— 缩放 / 平移(右键拖拽平移，滚轮缩放) ——
  const onWheel = (e: ReactWheelEvent) => {
    if (!sel) return;
    e.preventDefault();
    setZoom((z) => {
      const nz = Math.min(5, Math.max(1, +(z + (e.deltaY < 0 ? 0.2 : -0.2)).toFixed(2)));
      if (nz === 1) setPan({ x: 0, y: 0 });
      return nz;
    });
  };
  const onPanDown = (e: ReactMouseEvent) => { if (e.button !== 2) return; e.preventDefault(); panRef.current = { px: pan.x, py: pan.y, ox: e.clientX, oy: e.clientY }; };
  const onPanMove = (e: ReactMouseEvent) => { if (!panRef.current) return; setPan({ x: panRef.current.px + (e.clientX - panRef.current.ox), y: panRef.current.py + (e.clientY - panRef.current.oy) }); };
  const onPanUp = () => { panRef.current = null; };

  // 鼠标画框(归一化)
  const ptr = (e: ReactMouseEvent) => {
    const el = imgWrapRef.current;
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) };
  };
  const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
  // 空白处按下 → 开始画新框(点到已有框/手柄时由其自身 handler 拦截，不会到这里)
  const onDown = (e: ReactMouseEvent) => { if (e.button !== 0) return; setSelBox(-1); const p = ptr(e); if (p) setDraw({ x0: p.x, y0: p.y, x1: p.x, y1: p.y }); };

  // 在已有框上按下 → 选中并开始移动；在手柄上按下 → 缩放
  const startDrag = (e: ReactMouseEvent, idx: number, handle: 'move' | 'nw' | 'ne' | 'sw' | 'se') => {
    if (e.button !== 0) return;
    e.stopPropagation();
    const p = ptr(e);
    if (!p || !sel) return;
    snapshot();
    setSelBox(idx);
    setDrag({ idx, handle, ox: p.x, oy: p.y, box0: { ...sel.boxes[idx] } });
  };

  const applyDrag = (px: number, py: number) => {
    if (!drag || !sel) return;
    const { box0, handle, ox, oy } = drag;
    let nb: Box;
    if (handle === 'move') {
      const cx = clamp01(box0.cx + (px - ox));
      const cy = clamp01(box0.cy + (py - oy));
      nb = { ...box0, cx, cy };
    } else {
      // 以对角为锚点缩放
      let x0 = box0.cx - box0.w / 2, y0 = box0.cy - box0.h / 2, x1 = box0.cx + box0.w / 2, y1 = box0.cy + box0.h / 2;
      if (handle === 'nw') { x0 = px; y0 = py; }
      else if (handle === 'ne') { x1 = px; y0 = py; }
      else if (handle === 'sw') { x0 = px; y1 = py; }
      else { x1 = px; y1 = py; }
      const nx0 = clamp01(Math.min(x0, x1)), ny0 = clamp01(Math.min(y0, y1));
      const nx1 = clamp01(Math.max(x0, x1)), ny1 = clamp01(Math.max(y0, y1));
      const w = Math.max(0.01, nx1 - nx0), h = Math.max(0.01, ny1 - ny0);
      nb = { ...box0, cx: nx0 + w / 2, cy: ny0 + h / 2, w, h };
    }
    setSel((s) => (s ? { ...s, boxes: s.boxes.map((b, j) => (j === drag.idx ? nb : b)) } : s));
  };

  const onMove = (e: ReactMouseEvent) => {
    const p = ptr(e);
    if (!p) return;
    if (drag) applyDrag(p.x, p.y);
    else if (draw) setDraw({ ...draw, x1: p.x, y1: p.y });
  };
  const onUp = () => {
    if (drag) { setDrag(null); return; }
    if (draw && sel) {
      const w = Math.abs(draw.x1 - draw.x0);
      const h = Math.abs(draw.y1 - draw.y0);
      if (w > 0.02 && h > 0.02) {
        snapshot();
        setSel({ ...sel, boxes: [...sel.boxes, { cls: drawCls, cx: (draw.x0 + draw.x1) / 2, cy: (draw.y0 + draw.y1) / 2, w, h }] });
        setSelBox(sel.boxes.length);
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
      body: JSON.stringify({ split: itemSplit(sel.split), name: sel.name, boxes: sel.boxes, dataset }),
    }).then((r) => r.json()).then(() => {
      origRef.current = JSON.stringify(sel.boxes); // 标注已落库，重置未保存标记
      setSaving('✅ 已保存'); setTimeout(() => setSaving(''), 1500); load();
    });
  };
  const delImage = () => {
    if (!sel) return;
    const fromEditor = selIdx >= 0; // 从预览态删除 → 删完留在预览并跳到下一张
    fetch(`${backendUrl}/api/dataset/image?split=${itemSplit(sel.split)}&name=${encodeURIComponent(sel.name)}&${dsq}`, { method: 'DELETE' })
      .then(() => {
        if (fromEditor) delIdxRef.current = selIdx;
        else setSel(null);
        load();
      });
  };

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={() => { if (confirmDiscard()) onClose(); }}>
      <div className="w-[1100px] h-[720px] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden" onClick={(e) => e.stopPropagation()}>
        {/* 顶栏：标题 / 编辑导航 + 关闭 */}
        <div className="px-5 h-14 flex items-center gap-3 border-b border-outline-variant shrink-0">
          {sel ? (
            <div className="flex items-center gap-2">
              <button onClick={() => { if (confirmDiscard()) setSel(null); }} className="flex items-center gap-1 text-slate-500 hover:text-primary text-sm"><ArrowLeft size={16} /> 返回网格</button>
              {dirty && <span className="text-[11px] text-amber-600 font-medium">● 未保存</span>}
              <span className="w-px h-4 bg-outline-variant" />
              <button onClick={() => go(-1)} title="上一张 (←)" className="p-1 rounded hover:bg-slate-100"><ChevronLeft size={18} /></button>
              <button onClick={() => go(1)} title="下一张 (→)" className="p-1 rounded hover:bg-slate-100"><ChevronRight size={18} /></button>
              <span className="text-xs text-outline">← / → 切换</span>
              <span className="w-px h-4 bg-outline-variant" />
              <button onClick={undo} title="撤销 (Ctrl+Z)" className="p-1 rounded hover:bg-slate-100"><Undo2 size={16} /></button>
              <button onClick={redo} title="重做 (Ctrl+Y)" className="p-1 rounded hover:bg-slate-100"><Redo2 size={16} /></button>
            </div>
          ) : (
            <h3 className="font-display font-bold text-slate-800 whitespace-nowrap">数据集标注浏览</h3>
          )}
          <button onClick={() => { if (confirmDiscard()) onClose(); }} className="ml-auto p-1.5 rounded-full hover:bg-surface-container text-slate-500"><X size={20} /></button>
        </div>

        {sel ? (
          /* 编辑视图 */
          <div className="flex-1 flex min-h-0">
            <div
              className="flex-1 bg-slate-900 flex items-center justify-center p-4 min-w-0 relative overflow-hidden"
              onWheel={onWheel}
              onMouseDown={onPanDown}
              onMouseMove={onPanMove}
              onMouseUp={onPanUp}
              onMouseLeave={onPanUp}
              onContextMenu={(e) => e.preventDefault()}
            >
              <div
                ref={imgWrapRef}
                className="relative max-h-full max-w-full cursor-crosshair select-none"
                style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transformOrigin: 'center' }}
                onMouseDown={onDown}
                onMouseMove={onMove}
                onMouseUp={onUp}
                onMouseLeave={onUp}
              >
                <img src={imgUrl(sel.name, sel.split)} alt={sel.name} draggable={false} className="max-h-[600px] max-w-full object-contain block pointer-events-none" />
                {sel.boxes.map((b, i) => (
                  <div
                    key={i}
                    onMouseDown={(e) => startDrag(e, i, 'move')}
                    className={`absolute border-2 cursor-move ${selBox === i ? 'ring-2 ring-white' : ''}`}
                    style={{ ...boxStyle(b), borderColor: boxColor(b.cls) }}
                  >
                    <span className="absolute -top-5 left-0 px-1 text-[10px] font-bold text-white whitespace-nowrap pointer-events-none" style={{ background: boxColor(b.cls) }}>
                      {names[b.cls] ?? b.cls}
                    </span>
                    {selBox === i && (['nw', 'ne', 'sw', 'se'] as const).map((h) => (
                      <div
                        key={h}
                        onMouseDown={(e) => startDrag(e, i, h)}
                        className="absolute w-2.5 h-2.5 bg-white border border-slate-700 rounded-sm"
                        style={{
                          left: h.includes('w') ? -5 : undefined, right: h.includes('e') ? -5 : undefined,
                          top: h.includes('n') ? -5 : undefined, bottom: h.includes('s') ? -5 : undefined,
                          cursor: h === 'nw' || h === 'se' ? 'nwse-resize' : 'nesw-resize',
                        }}
                      />
                    ))}
                  </div>
                ))}
                {draw && (
                  <div
                    className="absolute border-2 border-dashed border-white/90 bg-white/10 pointer-events-none"
                    style={{ left: `${Math.min(draw.x0, draw.x1) * 100}%`, top: `${Math.min(draw.y0, draw.y1) * 100}%`, width: `${Math.abs(draw.x1 - draw.x0) * 100}%`, height: `${Math.abs(draw.y1 - draw.y0) * 100}%` }}
                  />
                )}
              </div>
              {/* 缩放控制(滚轮缩放、右键拖拽平移) */}
              <div className="absolute bottom-3 right-3 flex items-center gap-1 bg-white/90 rounded-lg shadow px-1.5 py-1 text-slate-600 text-xs">
                <button onClick={() => setZoom((z) => Math.max(1, +(z - 0.25).toFixed(2)))} className="px-1.5 hover:text-primary">−</button>
                <span className="w-10 text-center tabular-nums">{Math.round(zoom * 100)}%</span>
                <button onClick={() => setZoom((z) => Math.min(5, +(z + 0.25).toFixed(2)))} className="px-1.5 hover:text-primary">＋</button>
                <button onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }} className="px-1.5 border-l border-slate-300 hover:text-primary">适应</button>
              </div>
            </div>
            <div className="w-[280px] border-l border-outline-variant flex flex-col shrink-0">
              <div className="p-3 text-xs text-outline truncate border-b border-outline-variant">{sel.name}</div>
              {/* 标注辅助操作 */}
              <div className="px-3 py-2.5 border-b border-outline-variant space-y-2">
                <div className="flex gap-2">
                  <button onClick={aiAutolabel} className="btn-primary flex-1 py-2 text-xs justify-center whitespace-nowrap"><Sparkles size={14} /> AI 预标注</button>
                  <button onClick={copyPrev} disabled={selIdx <= 0} className="btn-secondary flex-1 py-2 text-xs justify-center whitespace-nowrap disabled:opacity-40 disabled:cursor-not-allowed"><Copy size={14} /> 复制上一张</button>
                </div>
                <div className="flex items-center gap-1.5 text-xs">
                  <ArrowLeftRight size={14} className="text-slate-400 shrink-0" />
                  <span className="text-slate-500 shrink-0 whitespace-nowrap">移动到</span>
                  {(['unassigned', 'train', 'valid', 'test'] as const).filter((s) => s !== itemSplit(sel.split)).map((s) => (
                    <button key={s} onClick={() => moveTo(s)} className="flex-1 px-1 py-1 rounded border border-outline-variant hover:border-primary hover:text-primary whitespace-nowrap">{s === 'unassigned' ? '未划分' : s}</button>
                  ))}
                </div>
                {aiMsg && <p className="text-[11px] text-slate-500">{aiMsg}</p>}
              </div>
              <div className="px-3 py-2 border-b border-outline-variant">
                <div className="flex items-center gap-2 text-xs">
                  <SquarePen size={14} className="text-primary shrink-0" />
                  <span className="text-slate-500 shrink-0">画框类别</span>
                  <select value={drawCls} onChange={(e) => setDrawCls(Number(e.target.value))} className="flex-1 border border-outline-variant rounded px-1 py-0.5 min-w-0">
                    {names.map((n, i) => <option key={n} value={i}>{n}</option>)}
                  </select>
                </div>
                <p className="text-[11px] text-outline-variant mt-1">空白处拖拽=新建框 · 点选框可拖动/拖角缩放 · 数字键改类、Del 删框、S 保存</p>
              </div>
              <div className="flex-1 overflow-y-auto p-3 space-y-2">
                {sel.boxes.length === 0 && <p className="text-sm text-slate-400">无标注框</p>}
                {sel.boxes.map((b, i) => (
                  <div key={i} onClick={() => setSelBox(i)} className={`flex items-center gap-2 p-2 rounded-lg border cursor-pointer ${selBox === i ? 'border-primary bg-primary/5' : 'border-outline-variant/60'}`}>
                    <span className="w-3 h-3 rounded-sm shrink-0" style={{ background: boxColor(b.cls) }} />
                    <select
                      value={b.cls}
                      onChange={(e) => { snapshot(); setSel({ ...sel, boxes: sel.boxes.map((x, j) => (j === i ? { ...x, cls: Number(e.target.value) } : x)) }); }}
                      className="flex-1 text-sm border border-outline-variant rounded px-1 py-0.5 min-w-0"
                    >
                      {names.map((n, ci) => <option key={n} value={ci}>{n}</option>)}
                    </select>
                    <button onClick={(e) => { e.stopPropagation(); snapshot(); setSel({ ...sel, boxes: sel.boxes.filter((_, j) => j !== i) }); setSelBox(-1); }} className="text-red-500 hover:text-red-600 shrink-0"><Trash2 size={15} /></button>
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
            {/* 筛选 + 操作工具栏 */}
            <div className="px-4 py-2.5 border-b border-outline-variant shrink-0 flex items-center gap-2 flex-wrap text-sm">
              <select value={split} onChange={(e) => setSplit(e.target.value)} className="border border-outline-variant rounded-lg px-2 py-1">
                <option value="all">全部</option><option value="unassigned">未分配(暂存)</option><option value="train">train</option><option value="valid">valid</option><option value="test">test</option>
              </select>
              <select value={cls} onChange={(e) => setCls(e.target.value === '' ? '' : Number(e.target.value))} className="border border-outline-variant rounded-lg px-2 py-1">
                <option value="">全部类别</option>
                {names.map((n, i) => <option key={n} value={i}>{n}</option>)}
              </select>
              <button
                onClick={() => setIssue((v) => (v === 'empty' ? '' : 'empty'))}
                className={`px-2.5 py-1 rounded-lg border text-xs font-medium whitespace-nowrap ${issue === 'empty' ? 'border-primary bg-primary/10 text-primary' : 'border-outline-variant text-slate-600 hover:border-primary'}`}
              >
                仅未标注
              </button>
              {data && <span className="text-xs text-outline whitespace-nowrap">共 {data.total} 张</span>}

              {/* 右侧操作组 */}
              <div className="ml-auto flex items-center gap-2 flex-wrap justify-end">
                <label className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-primary text-white cursor-pointer text-xs font-medium hover:brightness-110 whitespace-nowrap shrink-0">
                  <ImagePlus size={14} /> 上传图片
                  <input type="file" accept="image/*" multiple className="hidden" onChange={uploadImages} />
                </label>
                <span className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-slate-50 border border-outline-variant/60 text-xs text-slate-500 shrink-0">
                  <button onClick={batchAutolabel} className="inline-flex items-center gap-1 text-primary font-medium hover:brightness-110 whitespace-nowrap"><Sparkles size={14} /> 批量预标注</button>
                  <span className="whitespace-nowrap">阈值</span>
                  <input type="number" min={0.1} max={0.9} step={0.05} value={aiConf} onChange={(e) => setAiConf(Math.min(0.9, Math.max(0.1, Number(e.target.value) || 0.25)))} className="w-12 border border-outline-variant rounded px-1 py-0.5" />
                </span>
                <button onClick={redistribute} className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-outline-variant text-slate-600 text-xs font-medium hover:border-primary hover:text-primary whitespace-nowrap shrink-0">
                  <Shuffle size={14} /> 自动划分
                </button>
                {selMode ? (
                  <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-lg bg-red-50 border border-red-200 text-xs shrink-0">
                    <button onClick={pickAll} className="text-slate-600 hover:text-primary whitespace-nowrap">全选本页</button>
                    <button onClick={deletePicked} disabled={!picked.size} className="inline-flex items-center gap-1 text-red-600 font-medium disabled:opacity-40 whitespace-nowrap"><Trash2 size={13} /> 删除选中({picked.size})</button>
                    <button onClick={exitSelMode} className="text-slate-500 hover:text-slate-700 whitespace-nowrap">取消</button>
                  </span>
                ) : (
                  <button onClick={() => setSelMode(true)} className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-red-200 text-red-500 text-xs font-medium hover:bg-red-50 whitespace-nowrap shrink-0">
                    <Trash2 size={14} /> 批量删除
                  </button>
                )}
                <span
                  className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-slate-50 border border-outline-variant/60 text-xs text-slate-500 shrink-0"
                  title="按时间均匀抽帧：每隔指定秒数取 1 张图，最多取到设定帧数后停止。例：每隔 1 秒抽 1 帧、最多 60 帧 → 一段 60 秒内的视频约抽 60 张。"
                >
                  <label className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-slate-700 text-white cursor-pointer font-medium hover:brightness-110 whitespace-nowrap">
                    <Film size={14} /> 视频切帧
                    <input type="file" accept="video/*" className="hidden" onChange={uploadVideo} />
                  </label>
                  <span className="whitespace-nowrap">每隔</span>
                  <input type="number" min={0.2} step={0.2} value={everySec} onChange={(e) => setEverySec(Number(e.target.value) || 1)} className="w-12 border border-outline-variant rounded px-1 py-0.5" />
                  <span className="whitespace-nowrap">秒抽1帧 · 上限</span>
                  <input type="number" min={1} max={500} value={maxFrames} onChange={(e) => setMaxFrames(Number(e.target.value) || 60)} className="w-14 border border-outline-variant rounded px-1 py-0.5" />
                  <span className="whitespace-nowrap">帧</span>
                </span>
              </div>
              {busy && <span className="w-full text-xs text-primary font-medium">{busy}</span>}
            </div>

            {/* 标注进度 + 类别统计 */}
            {(() => {
              const sp = info?.splits ?? {};
              const vals = Object.values(sp) as { images: number; annotated: number }[];
              const agg = split === 'all'
                ? vals.reduce((a, v) => ({ images: a.images + v.images, annotated: a.annotated + v.annotated }), { images: 0, annotated: 0 })
                : (sp[split] ?? { images: 0, annotated: 0 });
              const imgs = agg.images;
              const ann = agg.annotated;
              const pct = imgs ? Math.round((ann / imgs) * 100) : 0;
              return (
                <div className="px-4 py-2.5 border-b border-outline-variant/60 shrink-0 flex items-center gap-4 flex-wrap">
                  <div className="flex items-center gap-2 text-xs text-slate-600">
                    <span className="font-medium">{split === 'all' ? '全部' : split} 标注进度</span>
                    <div className="w-40 h-2 rounded-full bg-slate-100 overflow-hidden">
                      <div className="h-full rounded-full bg-gradient-to-r from-primary to-cyan-400" style={{ width: `${pct}%` }} />
                    </div>
                    <span className="tabular-nums">已标 <b className="text-slate-800">{ann}</b> / {imgs}（未标 {Math.max(0, imgs - ann)}）</span>
                  </div>
                  <div className="flex items-center gap-1.5 flex-wrap ml-auto">
                    {(info?.classes || []).map((c) => (
                      <span key={c.name} className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] text-slate-600 bg-slate-50 border border-outline-variant/50">
                        <span className="w-2 h-2 rounded-sm" style={{ background: boxColor(Math.max(0, names.indexOf(c.name))) }} />{c.name} <b>{c.count}</b>
                      </span>
                    ))}
                  </div>
                </div>
              );
            })()}
            <div
              className="flex-1 overflow-y-auto p-4 grid grid-cols-4 gap-3 content-start relative"
              onDragOver={(e) => { e.preventDefault(); if (!dragOver) setDragOver(true); }}
              onDragLeave={(e) => { if (e.currentTarget === e.target) setDragOver(false); }}
              onDrop={onDrop}
            >
              {dragOver && (
                <div className="absolute inset-0 z-10 m-2 rounded-xl border-2 border-dashed border-primary bg-primary/10 flex items-center justify-center pointer-events-none">
                  <span className="text-primary font-medium text-sm">松手即批量上传到「未划分」暂存区</span>
                </div>
              )}
              {data?.items.map((it, i) => (
                <div
                  key={it.name}
                  onClick={() => (selMode ? togglePick(it.name) : openItem(it, i))}
                  className={`group cursor-pointer rounded-lg overflow-hidden border bg-slate-900 relative ${selMode && picked.has(it.name) ? 'border-red-500 ring-2 ring-red-400' : 'border-outline-variant/60 hover:border-primary'}`}
                >
                  <div className="relative">
                    <img src={imgUrl(it.name, it.split)} alt={it.name} loading="lazy" className={`w-full h-32 object-cover ${selMode && picked.has(it.name) ? 'opacity-60' : ''}`} />
                    {it.boxes.map((b, i) => (
                      <div key={i} className="absolute border-2 pointer-events-none" style={{ ...boxStyle(b), borderColor: boxColor(b.cls) }} />
                    ))}
                    {selMode && (
                      <div className={`absolute top-1 left-1 w-5 h-5 rounded-md border-2 flex items-center justify-center ${picked.has(it.name) ? 'bg-red-500 border-red-500' : 'bg-black/40 border-white/70'}`}>
                        {picked.has(it.name) && <Check size={13} className="text-white" />}
                      </div>
                    )}
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
