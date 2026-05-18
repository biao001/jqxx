/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { motion } from 'motion/react';
import { Hourglass, List as ListIcon, Radar, Download, Smartphone, EyeOff, Coffee, AlertTriangle } from 'lucide-react';
import { Detection, DrivingStats, DetectionType } from '../types';

interface SidebarProps {
  stats: DrivingStats | null;
  detections: Detection[];
  isAnalyzing: boolean;
}

const ICON_MAP = {
  [DetectionType.CELL_PHONE]: Smartphone,
  [DetectionType.DISTRACTION]: EyeOff,
  [DetectionType.YAWN]: Coffee,
  [DetectionType.SMOKING]: Coffee, // Placeholder
};

export default function Sidebar({ stats, detections, isAnalyzing }: SidebarProps) {
  return (
    <div className="w-[380px] shrink-0 flex flex-col gap-gutter h-full">
      {/* Score Card */}
      <div className="card p-6 flex flex-col items-center justify-center relative overflow-hidden shrink-0">
        <div className="absolute -right-6 -top-6 w-32 h-32 bg-primary/10 rounded-full blur-2xl"></div>
        <h3 className="font-display text-xl font-bold text-on-surface mb-2 w-full text-left border-b border-outline-variant pb-2">驾驶状态评分</h3>
        
        <div className="flex items-end justify-center gap-2 mt-4">
          <span className={`font-display text-[80px] leading-none transition-colors duration-500 ${stats ? 'text-primary' : 'text-slate-300'}`}>
            {stats ? stats.score : '--'}
          </span>
          <span className="font-medium text-lg text-outline mb-2">/ 100</span>
        </div>

        <div className="mt-4">
          {isAnalyzing ? (
             <div className="bg-orange-50 text-orange-500 px-6 py-2 rounded-full font-bold flex items-center gap-2 border border-orange-100">
               <AlertTriangle size={18} />
               <span>轻度危险</span>
             </div>
          ) : (
            <div className="bg-slate-100 text-slate-500 px-6 py-2 rounded-full font-bold flex items-center gap-2 underline-offset-4">
              <Hourglass size={18} />
              <span>等待数据</span>
            </div>
          )}
        </div>
      </div>

      {/* Detection List */}
      <div className="card p-4 flex flex-col min-h-0 overflow-hidden" style={{ height: '280px' }}>
        <h3 className="font-display text-lg font-bold text-on-surface mb-3 border-b border-outline-variant pb-2">检测结果列表</h3>
        {detections.length > 0 ? (
          <div className="overflow-y-auto flex-1 pr-2 space-y-2">
            {detections.map((d) => (
              <motion.div
                key={d.id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex items-center justify-between p-3 bg-surface-container rounded border border-outline-variant/50 hover:shadow-[0_4px_12px_rgba(0,0,0,0.06)] transition-shadow"
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded bg-white shadow-sm border border-outline-variant/20">
                    {(() => {
                      const Icon = ICON_MAP[d.type] || ListIcon;
                      return <Icon className="text-primary" size={16} />;
                    })()}
                  </div>
                  <span className="font-medium text-sm">{d.type}</span>
                </div>
                <span className="text-xs text-outline font-mono">{d.timestamp}</span>
              </motion.div>
            ))}
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-400 opacity-40">
            <ListIcon size={48} className="mb-2" />
            <p className="text-sm">暂无检测记录</p>
          </div>
        )}
      </div>

      {/* Big Model Analysis Area */}
      <div className="card p-4 flex flex-col min-h-0 flex-1 overflow-hidden">
        <h3 className="font-display text-lg font-bold text-on-surface mb-3 border-b border-outline-variant pb-2">大模型分析区</h3>
        <div className="flex-1 flex flex-col min-h-0">
          {/* Radar Chart Visual */}
          <div className="h-40 relative flex items-center justify-center bg-surface-container-low rounded-lg border border-outline-variant/30 shrink-0 mb-4 overflow-hidden">
             <div className="absolute inset-0 flex items-center justify-center opacity-10">
                {[32, 24, 16].map(size => (
                   <div key={size} className={`border border-outline rounded-full absolute`} style={{ width: size * 4, height: size * 4 }}></div>
                ))}
                <div className="w-full h-[1px] bg-outline absolute rotate-0"></div>
                <div className="w-full h-[1px] bg-outline absolute rotate-[72deg]"></div>
                <div className="w-full h-[1px] bg-outline absolute rotate-[144deg]"></div>
             </div>
             
             {isAnalyzing && (
               <svg className="absolute inset-0 w-full h-full p-4 pointer-events-none overflow-visible">
                 <motion.polygon
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 0.2 }}
                    points="50,10 90,30 80,85 20,85 10,30"
                    fill="#005daa"
                    className="origin-center"
                    style={{ transform: 'translate(50%, 50%) scale(0.8)' }}
                 />
               </svg>
             )}

             <div className="relative z-10 text-center opacity-40">
                <Radar size={32} className="text-primary mx-auto mb-1" />
                <p className="text-[10px] font-bold uppercase tracking-tight">五维推演模型</p>
             </div>

             {/* Radar Labels */}
             <div className="absolute inset-0 pointer-events-none p-2 text-[10px] font-bold text-outline-variant">
               <span className="absolute top-1 left-1/2 -translate-x-1/2">专注度</span>
               <span className="absolute top-1/3 right-1">反应速度</span>
               <span className="absolute bottom-1 right-4">操作规范</span>
               <span className="absolute bottom-1 left-4">疲劳程度</span>
               <span className="absolute top-1/3 left-1">情绪稳定</span>
             </div>
          </div>

          <div className="flex-1 flex flex-col min-h-0">
             <div className="flex-1 p-4 bg-slate-50/50 rounded-lg border border-dashed border-outline-variant flex items-center justify-center overflow-y-auto">
                <p className="text-sm text-outline italic text-center leading-relaxed">
                   {isAnalyzing 
                     ? "基于大模型的驾驶偏好分析：当前驾驶者在专注度和操作规范性方面表现良好，但疲劳程度指标偏低，建议在连续驾驶2小时后进行适当休息。"
                     : "正在等待分析以生成驾驶偏好推演报告..."
                   }
                </p>
             </div>
          </div>
        </div>
      </div>

      {/* Download Button */}
      <button className="w-full btn-secondary mt-auto" disabled={!isAnalyzing}>
        <Download size={20} />
        下载结果报告
      </button>
    </div>
  );
}
