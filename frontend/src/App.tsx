import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, Bell, Film, HelpCircle, UserCircle } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import Sidebar from './components/Sidebar';
import ControlPanel from './components/ControlPanel';
import UploadModal from './components/UploadModal';
import { AnalysisResult, BehaviorSummary, Detection, DrivingStats, FatigueSummary } from './types';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
const CAMERA_FPS = Number(import.meta.env.VITE_CAMERA_FPS || 6);

function websocketUrl(path: string) {
  const url = new URL(BACKEND_URL);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  url.pathname = path;
  return url.toString();
}

function mergeDetections(previous: Detection[], incoming: Detection[]) {
  const merged = [...previous];
  const seen = new Set(merged.map((item) => `${item.timestamp}-${item.type}-${item.source}`));
  for (const item of incoming) {
    const key = `${item.timestamp}-${item.type}-${item.source}`;
    if (!seen.has(key)) {
      merged.push(item);
      seen.add(key);
    }
  }
  return merged.slice(-80);
}

function videoBoxStyle(video: HTMLVideoElement | null, bbox?: number[] | null): CSSProperties | null {
  if (!video || !bbox || bbox.length !== 4 || video.videoWidth === 0 || video.videoHeight === 0) return null;
  const rect = video.getBoundingClientRect();
  const scale = Math.min(rect.width / video.videoWidth, rect.height / video.videoHeight);
  const renderedWidth = video.videoWidth * scale;
  const renderedHeight = video.videoHeight * scale;
  const offsetX = (rect.width - renderedWidth) / 2;
  const offsetY = (rect.height - renderedHeight) / 2;
  const [x1, y1, x2, y2] = bbox;

  return {
    left: `${offsetX + x1 * scale}px`,
    top: `${offsetY + y1 * scale}px`,
    width: `${Math.max(1, (x2 - x1) * scale)}px`,
    height: `${Math.max(1, (y2 - y1) * scale)}px`,
  };
}

export default function App() {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedVideoUrl, setSelectedVideoUrl] = useState<string | null>(null);
  const [stats, setStats] = useState<DrivingStats | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [analysisText, setAnalysisText] = useState('');
  const [latestResult, setLatestResult] = useState<AnalysisResult | null>(null);
  const [currentBehavior, setCurrentBehavior] = useState<BehaviorSummary | null>(null);
  const [currentFatigue, setCurrentFatigue] = useState<FatigueSummary | null>(null);
  const [behaviorBoxStyle, setBehaviorBoxStyle] = useState<CSSProperties | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [statusMessage, setStatusMessage] = useState('等待真实视频或相机输入');

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const captureTimerRef = useRef<number | null>(null);
  const streamModeRef = useRef<'camera' | 'video' | null>(null);
  const detectionHistoryRef = useRef<Detection[]>([]);

  const resetResults = useCallback(() => {
    detectionHistoryRef.current = [];
    setDetections([]);
    setStats(null);
    setLatestResult(null);
    setCurrentBehavior(null);
    setCurrentFatigue(null);
    setBehaviorBoxStyle(null);
    setAnalysisText('');
  }, []);

  const updateFromResult = useCallback((result: AnalysisResult) => {
    setLatestResult(result);
    setStats(result.stats);
    const merged = mergeDetections(detectionHistoryRef.current, result.detections || []);
    detectionHistoryRef.current = merged;
    setDetections(merged);
    setCurrentBehavior(result.current_behavior ?? null);
    setCurrentFatigue(result.current_fatigue ?? null);
    setAnalysisText(result.llm_analysis || '');
    setStatusMessage(result.error ? result.error : `真实数据已更新${result.frame_id !== null ? ` · frame ${result.frame_id}` : ''}`);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const checkBackend = () => {
      fetch(`${BACKEND_URL}/health`)
        .then((response) => {
          if (!response.ok) throw new Error('backend offline');
          return response.json();
        })
        .then(() => !cancelled && setBackendOnline(true))
        .catch(() => !cancelled && setBackendOnline(false));
    };
    checkBackend();
    const timer = window.setInterval(checkBackend, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (selectedVideoUrl) URL.revokeObjectURL(selectedVideoUrl);
    };
  }, [selectedVideoUrl]);

  useEffect(() => {
    const updateBox = () => setBehaviorBoxStyle(videoBoxStyle(videoRef.current, currentBehavior?.bbox ?? null));
    updateBox();
    window.addEventListener('resize', updateBox);
    return () => window.removeEventListener('resize', updateBox);
  }, [currentBehavior?.bbox, selectedVideoUrl, isCameraActive]);

  const stopFrameStream = useCallback(() => {
    if (captureTimerRef.current) {
      window.clearInterval(captureTimerRef.current);
      captureTimerRef.current = null;
    }
    websocketRef.current?.close();
    websocketRef.current = null;
    if (streamModeRef.current === 'camera') {
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setIsCameraActive(false);
    }
    streamModeRef.current = null;
    setIsAnalyzing(false);
  }, []);

  const startFrameWebSocket = useCallback(
    async (mode: 'camera' | 'video') => {
      const ws = new WebSocket(websocketUrl('/ws/camera'));
      ws.binaryType = 'arraybuffer';
      websocketRef.current = ws;
      streamModeRef.current = mode;

      ws.onopen = () => {
        setIsAnalyzing(true);
        setStatusMessage(mode === 'camera' ? '实时相机逐帧检测中' : '上传视频模拟相机逐帧检测中');
        captureTimerRef.current = window.setInterval(() => {
          const video = videoRef.current;
          const canvas = canvasRef.current;
          const socket = websocketRef.current;
          if (!video || !canvas || !socket || socket.readyState !== WebSocket.OPEN || video.videoWidth === 0) return;
          if (mode === 'video' && (video.paused || video.ended)) return;
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const context = canvas.getContext('2d');
          if (!context) return;
          context.drawImage(video, 0, 0, canvas.width, canvas.height);
          canvas.toBlob(
            (blob) => {
              if (blob && socket.readyState === WebSocket.OPEN) {
                socket.send(blob);
              }
            },
            'image/jpeg',
            0.72,
          );
        }, Math.max(200, 1000 / CAMERA_FPS));
      };

      ws.onmessage = (event) => {
        const result = JSON.parse(event.data) as AnalysisResult;
        if (result.error) {
          setStatusMessage(result.error);
          return;
        }
        updateFromResult(result);
      };

      ws.onerror = () => {
        setStatusMessage('实时检测连接异常');
      };

      ws.onclose = () => {
        if (websocketRef.current === ws) {
          stopFrameStream();
        }
      };
    },
    [stopFrameStream, updateFromResult],
  );

  const startCamera = useCallback(async () => {
    if (!backendOnline) {
      setStatusMessage('后端服务未连接，无法启动实时检测');
      return;
    }
    stopFrameStream();
    setIsCameraActive(true);
    setStatusMessage('正在请求本地相机权限');
    setSelectedFile(null);
    if (selectedVideoUrl) {
      URL.revokeObjectURL(selectedVideoUrl);
      setSelectedVideoUrl(null);
    }
    resetResults();

    try {
      await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540 }, audio: false });
      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        await videoRef.current.play();
      }
      await startFrameWebSocket('camera');
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : '无法打开本地相机');
      stopFrameStream();
    }
  }, [backendOnline, resetResults, selectedVideoUrl, startFrameWebSocket, stopFrameStream]);

  const handleToggleCamera = useCallback(() => {
    if (isCameraActive) {
      stopFrameStream();
      setStatusMessage('实时相机已停止');
    } else {
      void startCamera();
    }
  }, [isCameraActive, startCamera, stopFrameStream]);

  const handleFileConfirm = (file: File) => {
    stopFrameStream();
    if (selectedVideoUrl) URL.revokeObjectURL(selectedVideoUrl);
    setSelectedFile(file);
    setSelectedVideoUrl(URL.createObjectURL(file));
    setIsModalOpen(false);
    resetResults();
    setStatusMessage(`已选择真实视频：${file.name}`);
  };

  const handleStartUploadAnalysis = async () => {
    if (!selectedFile || !backendOnline) return;
    stopFrameStream();
    resetResults();
    setStatusMessage('上传视频按相机模式实时逐帧检测中');

    try {
      const video = videoRef.current;
      if (!video) throw new Error('视频播放器未准备好');
      video.currentTime = 0;
      await video.play();
      await startFrameWebSocket('video');
      video.onended = () => {
        stopFrameStream();
        setStatusMessage('上传视频实时检测完成，可下载结果报告');
      };
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : '视频实时检测启动失败');
      stopFrameStream();
    }
  };

  const handleCreateOrDownloadReport = async () => {
    if (!latestResult) return;
    if (latestResult.report_url) {
      window.open(`${BACKEND_URL}${latestResult.report_url}`, '_blank', 'noopener,noreferrer');
      return;
    }
    try {
      const response = await fetch(`${BACKEND_URL}/api/reports`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...latestResult,
          detections,
          source: latestResult.source.kind === 'camera' && selectedFile
            ? { kind: 'upload-live', name: selectedFile.name }
            : latestResult.source,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || '报告生成失败');
      window.open(`${BACKEND_URL}${payload.report_url}`, '_blank', 'noopener,noreferrer');
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : '报告生成失败');
    }
  };

  const videoLabel = useMemo(() => {
    if (isCameraActive) return '本地相机实时流';
    if (selectedFile) return selectedFile.name;
    return '';
  }, [isCameraActive, selectedFile]);

  const behaviorBadgeClass =
    currentBehavior?.severity === 'none'
      ? 'border-green-400/50 bg-green-500/20 text-green-50'
      : currentBehavior?.severity === 'critical' || currentBehavior?.severity === 'high'
        ? 'border-red-400/60 bg-red-500/25 text-red-50'
        : 'border-orange-300/60 bg-orange-500/25 text-orange-50';

  const fatigueBadgeClass =
    currentFatigue?.risk_level === 'low'
      ? 'border-green-400/50 bg-green-500/20 text-green-50'
      : currentFatigue?.risk_level === 'high'
        ? 'border-red-400/60 bg-red-500/25 text-red-50'
        : 'border-orange-300/60 bg-orange-500/25 text-orange-50';

  // 当前帧所有带 bbox 的检测都画框（不再只画 current_behavior 一个），
  // 否则像 smoking(medium) 这种被 high 行为盖过、又没框的 top 顶掉，框就永远不弹。
  const sevBoxClass = (sev?: string): string => {
    if (sev === 'critical' || sev === 'high') return 'border-red-400 shadow-[0_0_18px_rgba(248,113,113,0.65)]';
    if (sev === 'medium') return 'border-amber-400 shadow-[0_0_18px_rgba(251,191,36,0.6)]';
    return 'border-sky-400 shadow-[0_0_18px_rgba(56,189,248,0.6)]';
  };
  const sevTagClass = (sev?: string): string => {
    if (sev === 'critical' || sev === 'high') return 'bg-red-500';
    if (sev === 'medium') return 'bg-amber-500';
    return 'bg-sky-500';
  };
  const overlayBoxes = (latestResult?.detections || [])
    .filter((d) => Array.isArray(d.bbox) && (d.bbox as number[]).length === 4)
    .map((d, i) => ({
      key: `${d.id ?? d.type}-${i}`,
      style: videoBoxStyle(videoRef.current, d.bbox as number[]),
      label: d.type,
      severity: d.severity,
    }))
    .filter((b) => b.style);

  const realtimeOverlays = latestResult ? (
    <div className="absolute left-4 right-4 top-16 grid grid-cols-2 gap-3 pointer-events-none">
      <div className={`rounded-lg border px-4 py-3 backdrop-blur-md shadow-lg ${behaviorBadgeClass}`}>
        <div className="text-[11px] font-bold uppercase tracking-wider opacity-80">{currentBehavior?.algorithm_label || '行为识别'}</div>
        <div className="mt-1 text-lg font-bold truncate">{currentBehavior?.label || '等待行为识别结果'}</div>
        <div className="mt-1 flex items-center justify-between text-xs opacity-90">
          <span>置信度 {currentBehavior ? `${(currentBehavior.confidence * 100).toFixed(0)}%` : '--'}</span>
          <span>{currentBehavior?.severity || '--'}</span>
        </div>
      </div>
      <div className={`rounded-lg border px-4 py-3 backdrop-blur-md shadow-lg ${fatigueBadgeClass}`}>
        <div className="text-[11px] font-bold uppercase tracking-wider opacity-80">{currentFatigue?.algorithm_label || '疲劳检测'}</div>
        <div className="mt-1 text-lg font-bold truncate">{currentFatigue?.label || '等待疲劳检测结果'}</div>
        <div className="mt-1 flex items-center justify-between text-xs opacity-90">
          <span>置信度 {currentFatigue ? `${(currentFatigue.confidence * 100).toFixed(0)}%` : '--'}</span>
          <span>{currentFatigue?.risk_level || '--'}</span>
        </div>
      </div>
    </div>
  ) : null;

  return (
    <div className="flex items-center justify-center min-h-screen bg-slate-100 py-4 font-sans antialiased">
      <div className="w-[1280px] h-[1065px] bg-surface text-on-surface flex flex-col shadow-2xl overflow-hidden rounded-xl border border-outline-variant relative">
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
          <main className="flex-1 p-margin overflow-y-auto bg-surface-container-low">
            <div className="flex gap-gutter h-full">
              <div className="flex-1 flex flex-col gap-gutter h-full min-w-0">
                <div className="card flex-1 flex flex-col overflow-hidden relative group">
                  <div className="w-full flex-1 bg-surface-container-high relative flex items-center justify-center min-h-0">
                    <AnimatePresence mode="wait">
                      {!selectedVideoUrl && !isCameraActive ? (
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
                          <h3 className="font-display text-2xl font-bold text-slate-800 mb-2">等待真实输入</h3>
                          <p className="font-medium text-slate-500 max-w-sm">上传视频或打开本地相机后，系统会调用后端算法返回真实检测数据。</p>
                        </motion.div>
                      ) : (
                        <motion.div key="player" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="absolute inset-0 bg-slate-950">
                          <video
                            ref={videoRef}
                            src={selectedVideoUrl ?? undefined}
                            className="w-full h-full object-contain bg-black"
                            controls={!isCameraActive}
                            autoPlay={isCameraActive}
                            onLoadedMetadata={() => setBehaviorBoxStyle(videoBoxStyle(videoRef.current, currentBehavior?.bbox ?? null))}
                            playsInline
                          />
                          {overlayBoxes.map((b) => (
                            <div
                              key={b.key}
                              className={`absolute pointer-events-none border-2 ${sevBoxClass(b.severity)}`}
                              style={b.style as CSSProperties}
                            >
                              <div className={`absolute left-0 top-0 -translate-y-full px-2 py-0.5 text-[11px] font-bold text-white shadow ${sevTagClass(b.severity)}`}>
                                {b.label}
                              </div>
                            </div>
                          ))}
                          {realtimeOverlays}
                          <div className="absolute top-4 left-4 flex gap-2">
                            <span className="px-3 py-1 bg-black/60 backdrop-blur-sm text-white text-xs font-bold rounded flex items-center gap-2">
                              {isAnalyzing && <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />}
                              {isCameraActive ? 'LIVE' : isAnalyzing ? 'VIDEO LIVE' : 'VIDEO'}
                            </span>
                            <span className="px-3 py-1 bg-black/60 backdrop-blur-sm text-white text-xs font-bold rounded max-w-[420px] truncate">
                              {videoLabel}
                            </span>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    <div className="absolute bottom-base left-base right-base flex items-center justify-between text-[11px] font-mono text-outline-variant uppercase tracking-widest opacity-80">
                      <span>SIGNAL STATE: {isCameraActive || isAnalyzing ? 'STREAMING' : selectedFile ? 'READY' : 'AWAITING DATA'}</span>
                      <span className="normal-case tracking-normal text-right flex items-center gap-1">
                        {!backendOnline && <AlertCircle size={13} className="text-error" />}
                        {statusMessage}
                      </span>
                    </div>
                    <canvas ref={canvasRef} className="hidden" />
                  </div>

                  <ControlPanel
                    onUpload={() => setIsModalOpen(true)}
                    onStartUploadAnalysis={handleStartUploadAnalysis}
                    onToggleCamera={handleToggleCamera}
                    isAnalyzing={isAnalyzing}
                    isCameraActive={isCameraActive}
                    hasSelectedFile={Boolean(selectedFile)}
                    backendOnline={backendOnline}
                  />
                </div>
              </div>

              <Sidebar
                stats={stats}
                detections={detections}
                analysisText={analysisText}
                isAnalyzing={isAnalyzing}
                latestResult={latestResult}
                onDownloadReport={handleCreateOrDownloadReport}
              />
            </div>
          </main>
        </div>

        <UploadModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} onConfirm={handleFileConfirm} />
      </div>
    </div>
  );
}
