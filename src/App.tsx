/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect } from 'react';
import { Bell, HelpCircle, UserCircle, Film, PlayCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import Sidebar from './components/Sidebar';
import ControlPanel from './components/ControlPanel';
import UploadModal from './components/UploadModal';
import { Detection, DrivingStats, DetectionType } from './types';

const MOCK_DETECTIONS: Detection[] = [
  { id: '1', type: DetectionType.CELL_PHONE, timestamp: '02:14', confidence: 0.98 },
  { id: '2', type: DetectionType.DISTRACTION, timestamp: '05:32', confidence: 0.85 },
  { id: '3', type: DetectionType.YAWN, timestamp: '12:05', confidence: 0.92 },
];

const MOCK_STATS: DrivingStats = {
  score: 78,
  status: '警告',
  focus: 85,
  reaction: 72,
  compliance: 90,
  fatigue: 45,
  stability: 88,
};

export default function App() {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [stats, setStats] = useState<DrivingStats | null>(null);

  const handleStartAnalysis = () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    // Simulate process
    setTimeout(() => {
      setDetections(MOCK_DETECTIONS);
      setStats(MOCK_STATS);
    }, 1500);
  };

  const handleFileConfirm = (file: string) => {
    setSelectedFile(file);
    setIsModalOpen(false);
    setDetections([]);
    setStats(null);
    setIsAnalyzing(false);
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-slate-100 py-4 font-sans antialiased">
      <div className="w-[1280px] h-[1065px] bg-surface text-on-surface flex flex-col shadow-2xl overflow-hidden rounded-xl border border-outline-variant relative">
        {/* Header */}
        <header className="bg-white px-6 h-16 border-b border-outline-variant flex justify-between items-center z-50 shrink-0">
          <div className="flex items-center gap-6">
            <h1 className="text-2xl font-bold font-display tracking-tight text-slate-900">驾驶状态推演与行为测算平台</h1>
          </div>
          <div className="flex items-center gap-4 text-primary">
            <button className="p-2 hover:bg-surface-container rounded-full transition-colors active:scale-90">
              <Bell size={22} />
            </button>
            <button className="p-2 hover:bg-surface-container rounded-full transition-colors active:scale-90">
              <HelpCircle size={22} />
            </button>
            <button className="p-2 hover:bg-surface-container rounded-full transition-colors active:scale-90">
              <UserCircle size={22} />
            </button>
          </div>
        </header>

        <div className="flex-1 flex overflow-hidden">
          {/* Main Content */}
          <main className="flex-1 p-margin overflow-y-auto bg-surface-container-low">
            <div className="flex gap-gutter h-full">
              {/* Left Column */}
              <div className="flex-1 flex flex-col gap-gutter h-full min-w-0">
                {/* Video Area */}
                <div className="card flex-1 flex flex-col overflow-hidden relative group">
                  <div className="w-full flex-1 bg-surface-container-high relative flex items-center justify-center min-h-0">
                    <AnimatePresence mode="wait">
                      {!selectedFile ? (
                        <motion.div
                          key="empty"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          className="flex flex-col items-center text-center p-xl z-10"
                        >
                          <div className="w-24 h-24 mb-6 bg-white rounded-full flex items-center justify-center text-primary/30 shadow-sm border border-outline-variant/30 group-hover:scale-105 transition-transform duration-500">
                            <Film size={48} />
                          </div>
                          <h3 className="font-display text-2xl font-bold text-slate-800 mb-2">等待视频上传</h3>
                          <p className="font-medium text-slate-500 max-w-sm">请先上传视频或开始拍摄，以启动 AI 行为多维测算引擎。</p>
                        </motion.div>
                      ) : (
                        <motion.div
                          key="player"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="absolute inset-0 bg-slate-900"
                        >
                          <img 
                            src="https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?auto=format&fit=crop&q=80&w=2070" 
                            className="w-full h-full object-cover opacity-60"
                            alt="Video Thumbnail"
                          />
                          {!isAnalyzing && (
                            <div className="absolute inset-0 flex items-center justify-center">
                              <button onClick={handleStartAnalysis} className="text-white hover:scale-110 transition-transform active:scale-90 drop-shadow-2xl">
                                <PlayCircle size={100} strokeWidth={1} />
                              </button>
                            </div>
                          )}
                          <div className="absolute top-4 left-4 flex gap-2">
                            <span className="px-3 py-1 bg-black/60 backdrop-blur-sm text-white text-xs font-bold rounded flex items-center gap-2">
                              {isAnalyzing && <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />}
                              LIVE
                            </span>
                            <span className="px-3 py-1 bg-black/60 backdrop-blur-sm text-white text-xs font-bold rounded">
                              {selectedFile}
                            </span>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                    
                    <div className="absolute bottom-base right-base text-[10px] font-mono text-outline-variant uppercase tracking-widest opacity-60">
                      SIGNAL STATE: {selectedFile ? (isAnalyzing ? 'ANALYZING' : 'READY') : 'AWAITING DATA'}
                    </div>
                  </div>

                  <ControlPanel 
                    onUpload={() => setIsModalOpen(true)} 
                    onStart={handleStartAnalysis}
                    isAnalyzing={isAnalyzing}
                  />
                </div>
              </div>

              {/* Right Column */}
              <Sidebar 
                stats={stats} 
                detections={detections} 
                isAnalyzing={isAnalyzing} 
              />
            </div>
          </main>
        </div>

        <UploadModal 
          isOpen={isModalOpen} 
          onClose={() => setIsModalOpen(false)} 
          onConfirm={handleFileConfirm}
        />
      </div>
    </div>
  );
}

