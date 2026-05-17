/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { motion, AnimatePresence } from 'motion/react';
import { X, Film, Check } from 'lucide-react';
import { useState } from 'react';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (fileName: string) => void;
}

const MOCK_FILES = [
  { name: 'drive_test_01.mp4', size: '24.5 MB', date: '2023-11-20 14:30' },
  { name: 'fatigue_sample_v2.avi', size: '158.2 MB', date: '2023-11-19 09:15' },
  { name: 'night_driving_obs.mp4', size: '42.1 MB', date: '2023-11-18 22:45' },
  { name: 'urban_lane_change.mov', size: '89.0 MB', date: '2023-11-17 11:20' },
];

export default function UploadModal({ isOpen, onClose, onConfirm }: UploadModalProps) {
  const [selected, setSelected] = useState(MOCK_FILES[0].name);

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm"
          />
          
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="relative w-full max-w-[640px] bg-white rounded-lg shadow-2xl flex flex-col overflow-hidden border border-slate-200"
          >
            <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
              <h2 className="text-xl font-bold font-display text-slate-900">上传视频文件</h2>
              <button onClick={onClose} className="text-slate-400 hover:text-slate-600 transition-colors">
                <X size={24} />
              </button>
            </div>

            <div className="p-6 flex flex-col gap-6">
              <div className="flex flex-col gap-2">
                <label className="input-label">文件路径</label>
                <div className="flex gap-2">
                  <input
                    className="flex-1 bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-slate-600 font-medium focus:outline-none focus:ring-2 focus:ring-primary/20"
                    readOnly
                    type="text"
                    value="C:\Users\Admin\Videos"
                  />
                  <button className="px-4 py-2 bg-slate-100 text-slate-600 rounded-lg border border-slate-200 font-medium hover:bg-slate-200 transition-colors">
                    浏览
                  </button>
                </div>
              </div>

              <div className="flex flex-col gap-2">
                <label className="input-label">选择视频文件</label>
                <div className="border border-slate-200 rounded-lg overflow-hidden flex flex-col divide-y divide-slate-100 max-h-[300px] overflow-y-auto">
                  {MOCK_FILES.map((file) => (
                    <label
                      key={file.name}
                      className={`flex items-center gap-4 px-4 py-3 hover:bg-slate-50 cursor-pointer transition-colors group ${
                        selected === file.name ? 'bg-blue-50/50' : ''
                      }`}
                    >
                      <input
                        type="radio"
                        name="video_select"
                        checked={selected === file.name}
                        onChange={() => setSelected(file.name)}
                        className="w-4 h-4 text-primary focus:ring-primary border-slate-300"
                      />
                      <Film className={`text-slate-400 transition-colors ${selected === file.name ? 'text-primary' : 'group-hover:text-primary'}`} size={20} />
                      <div className="flex-1 min-w-0">
                        <p className={`font-medium truncate ${selected === file.name ? 'text-primary' : 'text-slate-800'}`}>
                          {file.name}
                        </p>
                        <p className="text-slate-400 text-xs">
                          {file.size} • {file.date}
                        </p>
                      </div>
                      {selected === file.name && <Check className="text-primary" size={16} />}
                    </label>
                  ))}
                </div>
              </div>
            </div>

            <div className="px-6 py-4 bg-slate-50 border-t border-slate-100 flex justify-end gap-3">
              <button
                onClick={onClose}
                className="px-6 py-2.5 rounded-lg border border-slate-300 text-slate-600 font-semibold hover:bg-slate-100 transition-all active:scale-95"
              >
                取消
              </button>
              <button
                onClick={() => onConfirm(selected)}
                className="px-8 py-2.5 rounded-lg bg-primary text-white font-semibold shadow-md hover:bg-primary/90 transition-all active:scale-95"
              >
                确认
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
