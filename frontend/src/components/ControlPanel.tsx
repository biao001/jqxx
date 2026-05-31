import { Camera, CircleDot, Play, Radio, Settings2, Square, Upload } from 'lucide-react';

type RecDuration = 'manual' | '10' | '20' | '30';

interface ControlPanelProps {
  onUpload: () => void;
  onStartUploadAnalysis: () => void;
  onToggleCamera: () => void;
  onRecordSamples?: () => void;
  isRecording?: boolean;
  recDatasets?: { name: string; label: string }[];
  recDataset?: string;
  onRecDatasetChange?: (v: string) => void;
  recDuration?: RecDuration;
  onRecDurationChange?: (v: RecDuration) => void;
  recAutolabel?: boolean;
  onRecAutolabelChange?: (v: boolean) => void;
  isAnalyzing: boolean;
  isCameraActive: boolean;
  hasSelectedFile: boolean;
  backendOnline: boolean;
}

export default function ControlPanel({
  onUpload,
  onStartUploadAnalysis,
  onToggleCamera,
  onRecordSamples,
  isRecording = false,
  recDatasets = [],
  recDataset = '',
  onRecDatasetChange,
  recDuration = '10',
  onRecDurationChange,
  recAutolabel = true,
  onRecAutolabelChange,
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

      <div className="flex items-center justify-between gap-4 py-4 border-y border-slate-100">
        <div className="flex gap-4">
          <button onClick={onUpload} className="btn-secondary px-8 py-4 text-lg" disabled={isAnalyzing || isCameraActive}>
            <Upload size={24} className="text-slate-500" />
            上传视频
          </button>
          <button onClick={onToggleCamera} className="btn-secondary px-8 py-4 text-lg" disabled={isAnalyzing && !isCameraActive}>
            {isCameraActive ? <Square size={24} className="text-red-500" /> : <Camera size={24} className="text-slate-500" />}
            {isCameraActive ? '停止相机' : '实时相机'}
          </button>
          {isCameraActive && onRecordSamples && (
            <button
              onClick={onRecordSamples}
              className="btn-secondary px-8 py-4 text-lg"
              title="录制画面入库，(可选)自动预标注后到「模型与数据」修正并重训"
            >
              {isRecording
                ? <Square size={24} className="text-red-500 animate-pulse" />
                : <CircleDot size={24} className="text-rose-500" />}
              {isRecording ? '停止并入库' : recDuration === 'manual' ? '开始录制' : `录制 ${recDuration}s`}
            </button>
          )}
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

      {isCameraActive && onRecordSamples && (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-slate-600 px-1 -mt-1">
          <span className="font-semibold text-slate-500">录制采样设置：</span>
          <label className="flex items-center gap-1.5">
            存入
            <select
              value={recDataset}
              onChange={(e) => onRecDatasetChange?.(e.target.value)}
              disabled={isRecording}
              className="rounded-lg border border-outline-variant/60 px-2 py-1 text-sm bg-white disabled:opacity-50"
            >
              {recDatasets.map((d) => (
                <option key={d.name} value={d.name}>{d.label || d.name}</option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-1.5">
            时长
            <select
              value={recDuration}
              onChange={(e) => onRecDurationChange?.(e.target.value as RecDuration)}
              disabled={isRecording}
              className="rounded-lg border border-outline-variant/60 px-2 py-1 text-sm bg-white disabled:opacity-50"
            >
              <option value="manual">手动停止</option>
              <option value="10">10 秒</option>
              <option value="20">20 秒</option>
              <option value="30">30 秒</option>
            </select>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={recAutolabel}
              onChange={(e) => onRecAutolabelChange?.(e.target.checked)}
              disabled={isRecording}
              className="accent-primary w-4 h-4"
            />
            自动预标注
          </label>
        </div>
      )}

      <div className="mt-1 flex items-center justify-between text-sm text-slate-500 px-1">
        <span>系统状态：{isCameraActive ? (isRecording ? '录制采样中' : '实时流式检测') : isAnalyzing ? '分析中' : '就绪'}</span>
        <span className="font-mono">输入源：{isCameraActive ? 'Camera WS' : hasSelectedFile ? 'Uploaded Video' : '--'}</span>
      </div>
    </div>
  );
}
