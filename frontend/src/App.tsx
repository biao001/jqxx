import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, Bell, Film, HelpCircle, Loader2, UserCircle } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import Sidebar from './components/Sidebar';
import ControlPanel from './components/ControlPanel';
import UploadModal from './components/UploadModal';
import { AnalysisResult, Detection, DrivingStats } from './types';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
const CAMERA_FPS = Number(import.meta.env.VITE_CAMERA_FPS || 3);

function websocketUrl(path: string) {
  const url = new URL(BACKEND_URL);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  url.pathname = path;
  return url.toString();
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
  const [backendOnline, setBackendOnline] = useState(false);
  const [statusMessage, setStatusMessage] = useState('等待真实视频或相机输入');

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const captureTimerRef = useRef<number | null>(null);

  const updateFromResult = useCallback((result: AnalysisResult) => {
    setLatestResult(result);
    setStats(result.stats);
    setDetections(result.detections || []);
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

  const stopCamera = useCallback(() => {
    if (captureTimerRef.current) {
      window.clearInterval(captureTimerRef.current);
      captureTimerRef.current = null;
    }
    websocketRef.current?.close();
    websocketRef.current = null;
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
    setIsAnalyzing(false);
  }, []);

  const startCamera = useCallback(async () => {
    if (!backendOnline) {
      setStatusMessage('后端服务未连接，无法启动实时检测');
      return;
    }
    setIsCameraActive(true);
    setStatusMessage('正在请求本地相机权限');
    setSelectedFile(null);
    if (selectedVideoUrl) {
      URL.revokeObjectURL(selectedVideoUrl);
      setSelectedVideoUrl(null);
    }
    setDetections([]);
    setStats(null);
    setLatestResult(null);
    setAnalysisText('');

    try {
      await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540 }, audio: false });
      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        await videoRef.current.play();
      }

      const ws = new WebSocket(websocketUrl('/ws/camera'));
      ws.binaryType = 'arraybuffer';
      websocketRef.current = ws;

      ws.onopen = () => {
        setIsAnalyzing(true);
        setStatusMessage('实时相机逐帧检测中');
        captureTimerRef.current = window.setInterval(() => {
          const video = videoRef.current;
          const canvas = canvasRef.current;
          const socket = websocketRef.current;
          if (!video || !canvas || !socket || socket.readyState !== WebSocket.OPEN || video.videoWidth === 0) return;
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
          stopCamera();
        }
      };
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : '无法打开本地相机');
      stopCamera();
    }
  }, [backendOnline, selectedVideoUrl, stopCamera, updateFromResult]);

  const handleToggleCamera = useCallback(() => {
    if (isCameraActive) {
      stopCamera();
    } else {
      void startCamera();
    }
  }, [isCameraActive, startCamera, stopCamera]);

  const handleFileConfirm = (file: File) => {
    stopCamera();
    if (selectedVideoUrl) URL.revokeObjectURL(selectedVideoUrl);
    setSelectedFile(file);
    setSelectedVideoUrl(URL.createObjectURL(file));
    setIsModalOpen(false);
    setDetections([]);
    setStats(null);
    setLatestResult(null);
    setAnalysisText('');
    setStatusMessage(`已选择真实视频：${file.name}`);
  };

  const handleStartUploadAnalysis = async () => {
    if (!selectedFile || !backendOnline) return;
    setIsAnalyzing(true);
    setStatusMessage('正在上传真实视频并调用本地算法分析');
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${BACKEND_URL}/api/videos/analyze`, {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || '视频分析失败');
      }
      updateFromResult(result as AnalysisResult);
      setStatusMessage('视频分析完成');
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : '视频分析失败');
    } finally {
      setIsAnalyzing(false);
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
        body: JSON.stringify(latestResult),
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
                            playsInline
                          />
                          {isAnalyzing && (
                            <div className="absolute inset-0 pointer-events-none bg-black/10 flex items-center justify-center">
                              <div className="rounded-full bg-black/60 text-white px-4 py-2 flex items-center gap-2 text-sm font-semibold">
                                <Loader2 size={18} className="animate-spin" />
                                {isCameraActive ? '实时逐帧检测' : '视频分析中'}
                              </div>
                            </div>
                          )}
                          <div className="absolute top-4 left-4 flex gap-2">
                            <span className="px-3 py-1 bg-black/60 backdrop-blur-sm text-white text-xs font-bold rounded flex items-center gap-2">
                              {isCameraActive && <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />}
                              {isCameraActive ? 'LIVE' : 'VIDEO'}
                            </span>
                            <span className="px-3 py-1 bg-black/60 backdrop-blur-sm text-white text-xs font-bold rounded max-w-[420px] truncate">
                              {videoLabel}
                            </span>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    <div className="absolute bottom-base left-base right-base flex items-center justify-between text-[11px] font-mono text-outline-variant uppercase tracking-widest opacity-80">
                      <span>SIGNAL STATE: {isCameraActive ? 'STREAMING' : selectedFile ? (isAnalyzing ? 'ANALYZING' : 'READY') : 'AWAITING DATA'}</span>
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
