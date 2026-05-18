import { Camera, Play, Radio, Settings2, Square, Upload } from 'lucide-react';

interface ControlPanelProps {
  onUpload: () => void;
  onStartUploadAnalysis: () => void;
  onToggleCamera: () => void;
  isAnalyzing: boolean;
  isCameraActive: boolean;
  hasSelectedFile: boolean;
  backendOnline: boolean;
}

export default function ControlPanel({
  onUpload,
  onStartUploadAnalysis,
  onToggleCamera,
  isAnalyzing,
  isCameraActive,
  hasSelectedFile,
  backendOnline,
}: ControlPanelProps) {
  return (
    <div className="shrink-0 flex flex-col bg-white text-on-surface-variant p-6 gap-4 border-t border-outline-variant/30">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <Settings2 size={24} className="text-primary" />
          <span className="text-primary font-bold text-xl tracking-tight font-display">系统控制</span>
        </div>
        <div className={`text-sm font-semibold ${backendOnline ? 'text-green-600' : 'text-error'}`}>
          后端服务：{backendOnline ? '已连接' : '未连接'}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
          <div className="text-xs font-bold text-outline uppercase">行为识别</div>
          <div className="font-semibold text-slate-800">最终版行为识别算法</div>
        </div>
        <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
          <div className="text-xs font-bold text-outline uppercase">疲劳检测</div>
          <div className="font-semibold text-slate-800">最终版疲劳检测算法</div>
        </div>
      </div>

      <div className="flex items-center justify-between gap-4 py-4 border-y border-slate-100">
        <div className="flex gap-4">
          <button onClick={onUpload} className="btn-secondary px-8 py-4 text-lg" disabled={isAnalyzing || isCameraActive}>
            <Upload size={24} className="text-slate-500" />
            上传视频
          </button>
          <button onClick={onToggleCamera} className="btn-secondary px-8 py-4 text-lg" disabled={isAnalyzing}>
            {isCameraActive ? <Square size={24} className="text-red-500" /> : <Camera size={24} className="text-slate-500" />}
            {isCameraActive ? '停止相机' : '实时相机'}
          </button>
        </div>

        <button
          onClick={onStartUploadAnalysis}
          className="btn-primary px-10 py-4 text-xl shadow-lg shadow-primary/20"
          disabled={!hasSelectedFile || isAnalyzing || isCameraActive || !backendOnline}
        >
          {isCameraActive ? <Radio size={28} /> : <Play size={28} fill="currentColor" />}
          开始分析
        </button>
      </div>

      <div className="mt-1 flex items-center justify-between text-sm text-slate-500 px-1">
        <span>系统状态：{isCameraActive ? '实时流式检测' : isAnalyzing ? '分析中' : '就绪'}</span>
        <span className="font-mono">输入源：{isCameraActive ? 'Camera WS' : hasSelectedFile ? 'Uploaded Video' : '--'}</span>
      </div>
    </div>
  );
}
