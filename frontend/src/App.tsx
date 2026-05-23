import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, Bell, BellOff, Film, History as HistoryIcon, Settings as SettingsIcon, UserCircle } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import Sidebar from './components/Sidebar';
import ControlPanel from './components/ControlPanel';
import UploadModal from './components/UploadModal';
import AlertOverlay from './components/AlertOverlay';
import EventTimeline, { TimelineEvent } from './components/EventTimeline';
import VideoHud from './components/VideoHud';
import DriverStatusBar, { DriverStatus } from './components/DriverStatusBar';
import Toasts from './components/Toasts';
import HistoryModal from './components/HistoryModal';
import { useDriverAlerts } from './lib/useDriverAlerts';
import { AnalysisResult, BehaviorSummary, Detection, DrivingStats, FatigueSummary, Severity } from './types';

const SEV_RANK: Record<Severity, number> = { none: 0, low: 1, medium: 2, high: 3, critical: 4 };

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
const CAMERA_FPS = Number(import.meta.env.VITE_CAMERA_FPS || 10); // 上限帧率；实际由后端处理速度自适应

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
  const [settings, setSettings] = useState({ voice: true, hud: true, scanline: true, statusBar: true });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const alertsMuted = !settings.voice;
  const [videoTime, setVideoTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [sessionEvents, setSessionEvents] = useState<TimelineEvent[]>([]);
  const [toasts, setToasts] = useState<{ id: string; label: string; severity: Severity }[]>([]);

  const sessionStartRef = useRef(0);
  const activeTypesRef = useRef<Set<string>>(new Set());
  const eventSeqRef = useRef(0);
  const [boxesFresh, setBoxesFresh] = useState(false);
  const staleTimerRef = useRef<number | null>(null);
  const scoreSeriesRef = useRef<{ t: number; score: number; behavior: number; fatigue: number }[]>([]);
  const sessionEventsRef = useRef<TimelineEvent[]>([]);
  const sessionSourceRef = useRef('');
  const [historyOpen, setHistoryOpen] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const captureTimerRef = useRef<number | null>(null);
  const streamModeRef = useRef<'camera' | 'video' | null>(null);
  const streamingRef = useRef(false);
  const lastSentRef = useRef(0);
  const detectionHistoryRef = useRef<Detection[]>([]);

  const resetResults = useCallback(() => {
    detectionHistoryRef.current = [];
    activeTypesRef.current = new Set();
    eventSeqRef.current = 0;
    sessionStartRef.current = 0;
    scoreSeriesRef.current = [];
    sessionEventsRef.current = [];
    setDetections([]);
    setStats(null);
    setLatestResult(null);
    setCurrentBehavior(null);
    setCurrentFatigue(null);
    setBehaviorBoxStyle(null);
    setAnalysisText('');
    setSessionEvents([]);
    setVideoTime(0);
  }, []);

  const updateFromResult = useCallback((result: AnalysisResult) => {
    setLatestResult(result);
    setStats(result.stats);

    // 当前时刻：视频模式取视频时间，相机模式取会话已用时(供列表/时间轴跳转)
    const isVideo = streamModeRef.current === 'video';
    const t = isVideo
      ? videoRef.current?.currentTime ?? 0
      : sessionStartRef.current
        ? (performance.now() - sessionStartRef.current) / 1000
        : 0;
    if (!isVideo) {
      setVideoTime(t);
      setVideoDuration(t);
    }

    // 累积评分曲线(用于行程报告)
    if (result.stats) {
      scoreSeriesRef.current.push({
        t,
        score: result.stats.score,
        behavior: result.stats.behavior_score ?? 0,
        fatigue: result.stats.fatigue_score ?? 0,
      });
      if (scoreSeriesRef.current.length > 1200) scoreSeriesRef.current.shift();
    }

    // 给每条检测打上发生时刻，列表可点击跳转
    const tagged = (result.detections || []).map((d) => ({ ...d, vt: t }));
    const merged = mergeDetections(detectionHistoryRef.current, tagged);
    detectionHistoryRef.current = merged;
    setDetections(merged);
    setCurrentBehavior(result.current_behavior ?? null);
    setCurrentFatigue(result.current_fatigue ?? null);
    setAnalysisText(result.llm_analysis || '');

    // 新鲜度：每来一帧续期；超过 300ms 没新结果(暂停/结束/卡顿)就清空识别框
    setBoxesFresh(true);
    if (staleTimerRef.current) window.clearTimeout(staleTimerRef.current);
    staleTimerRef.current = window.setTimeout(() => setBoxesFresh(false), 300);
    setStatusMessage(result.error ? result.error : `真实数据已更新${result.frame_id !== null ? ` · frame ${result.frame_id}` : ''}`);

    // 事件时间轴：记录每个"新出现"的中等及以上行为，标注其发生时刻
    const current = new Set<string>();
    for (const d of result.detections || []) {
      if ((SEV_RANK[d.severity as Severity] ?? 0) < 2) continue; // 仅 medium 及以上
      const key = `${d.type}|${d.source}`;
      current.add(key);
      if (!activeTypesRef.current.has(key)) {
        const id = `ev-${eventSeqRef.current++}`;
        const sev = d.severity as Severity;
        const ev = { id, t, label: d.type, severity: sev };
        sessionEventsRef.current.push(ev);
        setSessionEvents((prev) => [...prev, ev].slice(-200));
        // 弹出实时事件提示，3.8s 后自动消失
        setToasts((prev) => [...prev, { id, label: d.type, severity: sev }].slice(-4));
        window.setTimeout(() => setToasts((prev) => prev.filter((x) => x.id !== id)), 3800);
      }
    }
    activeTypesRef.current = current;
  }, []);

  const alert = useDriverAlerts(latestResult, !alertsMuted);

  const handleSeek = useCallback(
    (t: number) => {
      const v = videoRef.current;
      if (v && !isCameraActive) {
        v.currentTime = t;
        setVideoTime(t);
      }
    },
    [isCameraActive],
  );

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

  // 跟踪视频播放时间(播放头)与时长，供事件时间轴定位/跳转
  useEffect(() => {
    const v = videoRef.current;
    if (!v || isCameraActive) return;
    const onTime = () => setVideoTime(v.currentTime);
    const onMeta = () => setVideoDuration(isFinite(v.duration) ? v.duration : 0);
    v.addEventListener('timeupdate', onTime);
    v.addEventListener('loadedmetadata', onMeta);
    v.addEventListener('durationchange', onMeta);
    onMeta();
    return () => {
      v.removeEventListener('timeupdate', onTime);
      v.removeEventListener('loadedmetadata', onMeta);
      v.removeEventListener('durationchange', onMeta);
    };
  }, [selectedVideoUrl, isCameraActive]);

  const saveSession = useCallback(() => {
    const series = scoreSeriesRef.current;
    scoreSeriesRef.current = [];
    const events = sessionEventsRef.current;
    sessionEventsRef.current = [];
    if (series.length < 3) return; // 太短不保存
    const avg = (arr: number[]) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
    const scores = series.map((s) => s.score);
    const payload = {
      source: sessionSourceRef.current || '会话',
      duration: series[series.length - 1].t,
      score_avg: Math.round(avg(scores)),
      score_min: Math.min(...scores),
      behavior_avg: Math.round(avg(series.map((s) => s.behavior))),
      fatigue_avg: Math.round(avg(series.map((s) => s.fatigue))),
      events_count: events.length,
      events,
      score_series: series.filter((_, i) => i % 3 === 0).map((s) => ({ t: Math.round(s.t * 10) / 10, score: s.score })),
    };
    fetch(`${BACKEND_URL}/api/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).catch(() => {});
  }, []);

  const stopFrameStream = useCallback(() => {
    saveSession();
    streamingRef.current = false;
    if (captureTimerRef.current) {
      window.clearTimeout(captureTimerRef.current);
      captureTimerRef.current = null;
    }
    if (staleTimerRef.current) {
      window.clearTimeout(staleTimerRef.current);
      staleTimerRef.current = null;
    }
    setBoxesFresh(false);
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
  }, [saveSession]);

  const startFrameWebSocket = useCallback(
    async (mode: 'camera' | 'video') => {
      const ws = new WebSocket(websocketUrl('/ws/camera'));
      ws.binaryType = 'arraybuffer';
      websocketRef.current = ws;
      streamModeRef.current = mode;

      const minInterval = Math.max(60, 1000 / CAMERA_FPS);

      // 抓取并发送一帧；成功返回 true。失败(视频暂停/未就绪)返回 false。
      const sendFrame = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const socket = websocketRef.current;
        if (!video || !canvas || !socket || socket.readyState !== WebSocket.OPEN || video.videoWidth === 0) return false;
        if (mode === 'video' && (video.paused || video.ended)) return false;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        if (!context) return false;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(
          (blob) => {
            if (blob && socket.readyState === WebSocket.OPEN) socket.send(blob);
          },
          'image/jpeg',
          0.72,
        );
        lastSentRef.current = performance.now();
        return true;
      };

      // 自适应节流泵：发出一帧后等后端结果回来再发下一帧，避免定时盲发导致排队滞后。
      const pump = () => {
        if (!streamingRef.current) return;
        const sent = sendFrame();
        if (!sent) {
          // 没发出去(视频未就绪/暂停)，稍后重试，别卡死循环
          captureTimerRef.current = window.setTimeout(pump, 150);
        }
        // 发出去了：等 onmessage 收到结果后再调度下一帧
      };

      ws.onopen = () => {
        setIsAnalyzing(true);
        streamingRef.current = true;
        sessionStartRef.current = performance.now();
        setStatusMessage(mode === 'camera' ? '实时相机逐帧检测中' : '上传视频模拟相机逐帧检测中');
        pump();
      };

      ws.onmessage = (event) => {
        const result = JSON.parse(event.data) as AnalysisResult;
        if (!result.error) updateFromResult(result);
        else setStatusMessage(result.error);
        // 收到结果即调度下一帧；距上次发送不足 minInterval 时补足间隔(限上限帧率)
        if (streamingRef.current) {
          const delay = Math.max(0, minInterval - (performance.now() - lastSentRef.current));
          captureTimerRef.current = window.setTimeout(pump, delay);
        }
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
    sessionSourceRef.current = '本地相机实时流';
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
    sessionSourceRef.current = selectedFile.name || '上传视频';
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
  const overlayBoxes = (latestResult?.detections || [])
    .filter((d) => Array.isArray(d.bbox) && (d.bbox as number[]).length === 4)
    .map((d, i) => ({
      key: `${d.id ?? d.type}-${i}`,
      style: videoBoxStyle(videoRef.current, d.bbox as number[]),
      label: d.type,
      severity: d.severity,
      confidence: d.confidence,
    }))
    .filter((b) => b.style);

  // 由当前帧检测推导驾驶员状态指示灯(正常/告警)
  const driverStatus = useMemo<DriverStatus | null>(() => {
    if (!latestResult) return null;
    const types = (latestResult.detections || []).map((d) => d.type);
    const has = (kw: string) => types.some((t) => t.includes(kw));
    return {
      present: !has('无人'),
      seatbelt: !has('未系'),
      hands: !has('离开方向盘') && !has('离盘'),
      gaze: !has('视线'),
      noPhone: !has('手机') && !has('电话'),
      noSmoke: !has('吸烟'),
    };
  }, [latestResult]);

  const sevHex = (sev?: string): string =>
    sev === 'critical' || sev === 'high' ? '#f87171' : sev === 'medium' ? '#fbbf24' : '#38bdf8';

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
        {currentFatigue?.signals && (
          <div className="mt-2 pt-2 border-t border-white/20 space-y-1.5">
            <div className="flex items-center justify-between text-[11px]">
              <span>视线 · {currentFatigue.signals.gaze_zone || '--'}</span>
              <span className="tabular-nums opacity-90">
                头部 {Math.round(currentFatigue.signals.head_yaw ?? 0)}° / {Math.round(currentFatigue.signals.head_pitch ?? 0)}°
              </span>
            </div>
            <div className="flex items-center gap-2 text-[11px]">
              <span className="shrink-0 font-bold">PERCLOS</span>
              <div className="flex-1 h-1.5 rounded-full bg-white/25 overflow-hidden">
                <div
                  className={`h-full rounded-full ${(currentFatigue.signals.perclos ?? 0) >= 0.45 ? 'bg-red-300' : 'bg-white/90'}`}
                  style={{ width: `${Math.round((currentFatigue.signals.perclos ?? 0) * 100)}%`, transition: 'width 0.4s ease' }}
                />
              </div>
              <span className="tabular-nums w-9 text-right">{Math.round((currentFatigue.signals.perclos ?? 0) * 100)}%</span>
            </div>
          </div>
        )}
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
          <div className="flex items-center gap-4 text-primary relative">
            <button
              onClick={() => setSettings((s) => ({ ...s, voice: !s.voice }))}
              title={alertsMuted ? '语音告警已静音，点击开启' : '语音告警已开启，点击静音'}
              className={`p-2 rounded-full transition-colors active:scale-90 ${alertsMuted ? 'text-slate-400 hover:bg-surface-container' : 'text-primary hover:bg-surface-container'}`}
            >
              {alertsMuted ? <BellOff size={22} /> : <Bell size={22} />}
            </button>
            <button
              onClick={() => setHistoryOpen(true)}
              title="行程历史"
              className="p-2 rounded-full transition-colors active:scale-90 hover:bg-surface-container"
            >
              <HistoryIcon size={22} />
            </button>
            <button
              onClick={() => setSettingsOpen((o) => !o)}
              title="显示设置"
              className={`p-2 rounded-full transition-colors active:scale-90 ${settingsOpen ? 'bg-surface-container text-primary' : 'hover:bg-surface-container'}`}
            >
              <SettingsIcon size={22} />
            </button>
            <button className="p-2 hover:bg-surface-container rounded-full transition-colors active:scale-90">
              <UserCircle size={22} />
            </button>

            {settingsOpen && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setSettingsOpen(false)} />
                <div className="absolute right-0 top-12 z-50 w-60 rounded-xl border border-outline-variant bg-white shadow-2xl p-3">
                  <div className="text-sm font-bold text-slate-700 px-1 pb-2 mb-1 border-b border-outline-variant">显示与告警设置</div>
                  {([
                    ['voice', '语音告警播报'],
                    ['hud', '机器视觉 HUD'],
                    ['scanline', '扫描线动效'],
                    ['statusBar', '驾驶员状态灯'],
                  ] as const).map(([key, label]) => (
                    <button
                      key={key}
                      onClick={() => setSettings((s) => ({ ...s, [key]: !s[key] }))}
                      className="w-full flex items-center justify-between px-2 py-2 rounded-lg hover:bg-surface-container transition-colors text-sm font-medium text-slate-700"
                    >
                      <span>{label}</span>
                      <span className={`relative w-9 h-5 rounded-full transition-colors ${settings[key] ? 'bg-primary' : 'bg-slate-300'}`}>
                        <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all ${settings[key] ? 'left-[18px]' : 'left-0.5'}`} />
                      </span>
                    </button>
                  ))}
                </div>
              </>
            )}
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
                          {boxesFresh && overlayBoxes.map((b) => {
                            const c = sevHex(b.severity);
                            return (
                              <div key={b.key} className="absolute pointer-events-none" style={b.style as CSSProperties}>
                                <div className="absolute inset-0 rounded-sm" style={{ border: `1px solid ${c}55`, boxShadow: `0 0 16px ${c}66, inset 0 0 14px ${c}22` }} />
                                {[
                                  'top-0 left-0 border-t-2 border-l-2',
                                  'top-0 right-0 border-t-2 border-r-2',
                                  'bottom-0 left-0 border-b-2 border-l-2',
                                  'bottom-0 right-0 border-b-2 border-r-2',
                                ].map((cls, i) => (
                                  <span key={i} className={`absolute w-3.5 h-3.5 ${cls}`} style={{ borderColor: c }} />
                                ))}
                                <div className="absolute left-0 top-0 -translate-y-full flex items-center gap-1.5 px-2 py-0.5 text-[11px] font-bold text-white" style={{ background: c }}>
                                  <span>{b.label}</span>
                                  <span className="opacity-85 tabular-nums">{Math.round((b.confidence ?? 0) * 100)}%</span>
                                </div>
                              </div>
                            );
                          })}
                          {settings.hud && (
                            <VideoHud active={isAnalyzing} scan={settings.scanline} latencyMs={latestResult?.metrics?.latency_ms} frameId={latestResult?.frame_id} />
                          )}
                          {settings.statusBar && <DriverStatusBar status={driverStatus} />}
                          {realtimeOverlays}
                          <AlertOverlay level={alert.level} text={alert.text} muted={alertsMuted} />
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

                {(selectedVideoUrl || isCameraActive) && (
                  <EventTimeline
                    events={sessionEvents}
                    duration={isCameraActive ? videoTime : videoDuration}
                    currentTime={videoTime}
                    seekable={Boolean(selectedVideoUrl) && !isCameraActive}
                    onSeek={handleSeek}
                  />
                )}
              </div>

              <Sidebar
                stats={stats}
                detections={detections}
                analysisText={analysisText}
                isAnalyzing={isAnalyzing}
                latestResult={latestResult}
                onDownloadReport={handleCreateOrDownloadReport}
                currentTime={videoTime}
                seekable={Boolean(selectedVideoUrl) && !isCameraActive}
                onSeek={handleSeek}
              />
            </div>
          </main>
        </div>

        <UploadModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} onConfirm={handleFileConfirm} />
        <Toasts toasts={toasts} />
        <HistoryModal open={historyOpen} onClose={() => setHistoryOpen(false)} backendUrl={BACKEND_URL} />
      </div>
    </div>
  );
}
