import { AnimatePresence, motion } from 'motion/react';
import { Check, FileVideo, X } from 'lucide-react';
import { useState } from 'react';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (file: File) => void;
}

function formatSize(size: number) {
  if (size >= 1024 * 1024) {
    return `${(size / 1024 / 1024).toFixed(1)} MB`;
  }
  return `${(size / 1024).toFixed(1)} KB`;
}

export default function UploadModal({ isOpen, onClose, onConfirm }: UploadModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

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
              <label className="border-2 border-dashed border-outline-variant rounded-lg bg-slate-50/70 min-h-[220px] flex flex-col items-center justify-center gap-4 cursor-pointer hover:border-primary hover:bg-blue-50/40 transition-colors">
                <input
                  className="sr-only"
                  type="file"
                  accept="video/*"
                  onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
                />
                <FileVideo size={48} className={selectedFile ? 'text-primary' : 'text-slate-400'} />
                <div className="text-center">
                  <p className="font-semibold text-slate-800">{selectedFile ? selectedFile.name : '选择本地真实视频文件'}</p>
                  <p className="text-sm text-slate-500 mt-1">
                    {selectedFile ? `${formatSize(selectedFile.size)} · ${selectedFile.type || 'video'}` : '支持 mp4、avi、mov 等浏览器可选择的视频'}
                  </p>
                </div>
              </label>

              {selectedFile && (
                <div className="flex items-center gap-3 rounded-lg border border-blue-100 bg-blue-50 px-4 py-3 text-primary">
                  <Check size={18} />
                  <span className="font-medium text-sm">已选择真实视频，确认后可上传到本地后端分析。</span>
                </div>
              )}
            </div>

            <div className="px-6 py-4 bg-slate-50 border-t border-slate-100 flex justify-end gap-3">
              <button
                onClick={onClose}
                className="px-6 py-2.5 rounded-lg border border-slate-300 text-slate-600 font-semibold hover:bg-slate-100 transition-all active:scale-95"
              >
                取消
              </button>
              <button
                onClick={() => selectedFile && onConfirm(selectedFile)}
                disabled={!selectedFile}
                className="px-8 py-2.5 rounded-lg bg-primary text-white font-semibold shadow-md hover:bg-primary/90 transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
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
