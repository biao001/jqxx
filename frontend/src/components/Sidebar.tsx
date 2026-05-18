import { motion } from 'motion/react';
import { AlertTriangle, BrainCircuit, Coffee, Download, EyeOff, Hourglass, List as ListIcon, Radar, Smartphone } from 'lucide-react';
import { AnalysisResult, Detection, DrivingStats } from '../types';

interface SidebarProps {
  stats: DrivingStats | null;
  detections: Detection[];
  analysisText: string;
  isAnalyzing: boolean;
  latestResult: AnalysisResult | null;
  onDownloadReport: () => void;
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

function sourceLabel(source: string) {
  if (source === 'behavior') return '行为识别';
  if (source === 'fatigue') return '疲劳检测';
  return source || '算法';
}

export default function Sidebar({ stats, detections, analysisText, isAnalyzing, latestResult, onDownloadReport }: SidebarProps) {
  const capabilityMode = latestResult?.capabilities.behavior?.mode || latestResult?.capabilities.fatigue?.mode;

  return (
    <div className="w-[380px] shrink-0 flex flex-col gap-gutter h-full">
      <div className="card p-6 flex flex-col items-center justify-center relative overflow-hidden shrink-0">
        <h3 className="font-display text-xl font-bold text-on-surface mb-2 w-full text-left border-b border-outline-variant pb-2">驾驶状态评分</h3>

        <div className="flex items-end justify-center gap-2 mt-4">
          <span className={`font-display text-[80px] leading-none transition-colors duration-500 ${stats ? 'text-primary' : 'text-slate-300'}`}>
            {stats ? stats.score : '--'}
          </span>
          <span className="font-medium text-lg text-outline mb-2">/ 100</span>
        </div>

        <div className="mt-4">
          <div className={`px-6 py-2 rounded-full font-bold flex items-center gap-2 border ${statusColor(stats?.status)}`}>
            {stats ? <AlertTriangle size={18} /> : <Hourglass size={18} />}
            <span>{stats ? stats.status : '等待数据'}</span>
          </div>
        </div>

        {stats && (
          <div className="grid grid-cols-5 gap-2 w-full mt-5 text-center">
            {[
              ['专注', stats.focus],
              ['反应', stats.reaction],
              ['规范', stats.compliance],
              ['清醒', stats.fatigue],
              ['稳定', stats.stability],
            ].map(([label, value]) => (
              <div key={label} className="rounded border border-slate-100 bg-slate-50 py-2">
                <div className="text-xs text-slate-500">{label}</div>
                <div className="font-bold text-slate-800">{value}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="card p-4 flex flex-col min-h-0 overflow-hidden" style={{ height: '280px' }}>
        <h3 className="font-display text-lg font-bold text-on-surface mb-3 border-b border-outline-variant pb-2">检测结果列表</h3>
        {detections.length > 0 ? (
          <div className="overflow-y-auto flex-1 pr-2 space-y-2">
            {detections.map((d) => {
              const Icon = iconForDetection(d.type);
              return (
                <motion.div
                  key={d.id}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="flex items-center justify-between p-3 bg-surface-container rounded border border-outline-variant/50 hover:shadow-[0_4px_12px_rgba(0,0,0,0.06)] transition-shadow"
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
                  <span className="text-xs text-outline font-mono">{d.timestamp}</span>
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
          <div className="h-40 relative flex items-center justify-center bg-surface-container-low rounded-lg border border-outline-variant/30 shrink-0 mb-4 overflow-hidden">
            <div className="absolute inset-0 flex items-center justify-center opacity-10">
              {[32, 24, 16].map((size) => (
                <div key={size} className="border border-outline rounded-full absolute" style={{ width: size * 4, height: size * 4 }} />
              ))}
              <div className="w-full h-[1px] bg-outline absolute rotate-0" />
              <div className="w-full h-[1px] bg-outline absolute rotate-[72deg]" />
              <div className="w-full h-[1px] bg-outline absolute rotate-[144deg]" />
            </div>

            <div className="relative z-10 text-center">
              <Radar size={32} className="text-primary mx-auto mb-1" />
              <p className="text-[10px] font-bold uppercase tracking-tight">实时五维状态</p>
              {capabilityMode === 'fallback' && <p className="text-[10px] text-orange-500 mt-1">部分模型使用轻量回退</p>}
            </div>

            <div className="absolute inset-0 pointer-events-none p-2 text-[10px] font-bold text-outline-variant">
              <span className="absolute top-1 left-1/2 -translate-x-1/2">专注度</span>
              <span className="absolute top-1/3 right-1">反应速度</span>
              <span className="absolute bottom-1 right-4">操作规范</span>
              <span className="absolute bottom-1 left-4">清醒程度</span>
              <span className="absolute top-1/3 left-1">状态稳定</span>
            </div>
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
