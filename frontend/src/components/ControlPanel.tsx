/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Settings2, Upload, Camera, Play, ChevronDown } from 'lucide-react';
import { useState } from 'react';

interface ControlPanelProps {
  onUpload: () => void;
  onStart: () => void;
  isAnalyzing: boolean;
}

export default function ControlPanel({ onUpload, onStart, isAnalyzing }: ControlPanelProps) {
  const [primaryAlg, setPrimaryAlg] = useState('行为识别算法A');
  const [secondaryAlg, setSecondaryAlg] = useState('疲劳检测算法A');

  return (
    <div className="shrink-0 flex flex-col bg-white text-on-surface-variant p-6 gap-4 border-t border-outline-variant/30">
      <div className="flex items-center gap-2 mb-2">
        <Settings2 size={24} className="text-primary" />
        <span className="text-primary font-bold text-xl tracking-tight font-display">系统控制</span>
      </div>

      <div className="flex items-center justify-between gap-4 py-4 border-y border-slate-100">
        <div className="flex gap-4">
          <button onClick={onUpload} className="btn-secondary px-8 py-4 text-lg">
            <Upload size={24} className="text-slate-500" />
            视频上传
          </button>
          <button className="btn-secondary px-8 py-4 text-lg">
            <Camera size={24} className="text-slate-500" />
            本地拍摄
          </button>
        </div>
        
        <button 
          onClick={onStart}
          className="btn-primary px-10 py-4 text-xl shadow-lg shadow-primary/20"
        >
          <Play size={28} fill="currentColor" />
          开始分析
        </button>
      </div>

      <div className="grid grid-cols-2 gap-6 mt-2">
        <div className="flex flex-col gap-2">
          <label className="input-label px-1">算法选择 (PRIMARY)</label>
          <div className="relative">
            <select 
              value={primaryAlg}
              onChange={(e) => setPrimaryAlg(e.target.value)}
              className="w-full border border-slate-200 rounded-lg px-4 py-4 focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none appearance-none bg-slate-50/50 text-slate-800 transition-all cursor-pointer text-lg font-medium"
            >
              <option>行为识别算法A</option>
              <option>V3.1 综合疲劳检测模型</option>
              <option>V2.4 基础行为识别</option>
            </select>
            <ChevronDown size={24} className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400" />
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <label className="input-label px-1">算法选择 (SECONDARY)</label>
          <div className="relative">
            <select 
              value={secondaryAlg}
              onChange={(e) => setSecondaryAlg(e.target.value)}
              className="w-full border border-slate-200 rounded-lg px-4 py-4 focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none appearance-none bg-slate-50/50 text-slate-800 transition-all cursor-pointer text-lg font-medium"
            >
              <option>疲劳检测算法A</option>
              <option>V4.0 (Beta) 情绪综合推演</option>
            </select>
            <ChevronDown size={24} className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400" />
          </div>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-slate-100 flex items-center justify-between text-sm text-slate-500 px-1">
        <span>系统状态: {isAnalyzing ? '分析中' : '就绪'}</span>
        <span className="font-mono">延迟: {isAnalyzing ? '24ms' : '-- ms'}</span>
      </div>
    </div>
  );
}
