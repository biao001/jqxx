import { Camera, Download, Trash2, X } from 'lucide-react';
import { Severity } from '../types';

export interface Snapshot {
  id: string;
  url: string; // dataURL(jpeg)
  label: string;
  severity: Severity;
  t: number; // 发生时刻(秒)
  wall: number; // 抓拍时的墙钟时间(ms)
}

interface Props {
  open: boolean;
  onClose: () => void;
  snapshots: Snapshot[];
  onClear: () => void;
}

const sevColor = (s: Severity) =>
  s === 'critical' || s === 'high' ? 'border-red-400' : s === 'medium' ? 'border-amber-400' : 'border-sky-400';

function download(snap: Snapshot) {
  const a = document.createElement('a');
  a.href = snap.url;
  a.download = `dms-${snap.label}-${new Date(snap.wall).toISOString().replace(/[:.]/g, '-')}.jpg`;
  a.click();
}

/** 关键帧抓拍画廊：展示自动/手动抓拍的证据帧，可下载。*/
export default function SnapshotGallery({ open, onClose, snapshots, onClear }: Props) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 backdrop-blur-sm" onClick={onClose}>
      <div className="w-[920px] h-[620px] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <div className="px-5 h-14 flex items-center justify-between border-b border-outline-variant">
          <h3 className="font-display font-bold text-slate-800 flex items-center gap-2">
            <Camera size={18} className="text-primary" /> 关键帧抓拍 <span className="text-xs text-slate-400 font-normal">{snapshots.length} 张</span>
          </h3>
          <div className="flex items-center gap-2">
            {snapshots.length > 0 && (
              <button onClick={onClear} className="text-xs text-red-500 hover:text-red-600 flex items-center gap-1 px-2 py-1 rounded hover:bg-red-50">
                <Trash2 size={13} /> 清空
              </button>
            )}
            <button onClick={onClose} className="p-1.5 rounded-full hover:bg-surface-container text-slate-500"><X size={20} /></button>
          </div>
        </div>
        {snapshots.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-400">
            <Camera size={48} className="mb-2 opacity-50" />
            <p className="text-sm">暂无抓拍 · 检测到高危行为时自动抓拍，或按 C / 点相机按钮手动抓拍</p>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto p-4 grid grid-cols-3 gap-3">
            {snapshots.map((s) => (
              <div key={s.id} className={`group relative rounded-lg overflow-hidden border-2 ${sevColor(s.severity)} bg-slate-900`}>
                <img src={s.url} alt={s.label} className="w-full h-36 object-cover" />
                <div className="absolute inset-x-0 bottom-0 bg-black/60 px-2 py-1 flex items-center justify-between">
                  <span className="text-[11px] font-bold text-white truncate">{s.label}</span>
                  <button onClick={() => download(s)} title="下载" className="text-white/80 hover:text-white">
                    <Download size={14} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
