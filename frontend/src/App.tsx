import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, BarChart3, Bell, BellOff, Bot, Camera, Database, Film, History as HistoryIcon, IdCard, LogOut, Maximize2, Settings as SettingsIcon } from 'lucide-react';
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
import SnapshotGallery, { Snapshot } from './components/SnapshotGallery';
import AiChatModal from './components/AiChatModal';
import DriverModal from './components/DriverModal';
import InfoSidebar from './components/InfoSidebar';
import LandingPage from './components/LandingPage';
import AuthPage from './components/AuthPage';
import StatsDashboard from './components/StatsDashboard';
import ModelDataPage from './components/ModelDataPage';
import { useDriverAlerts } from './lib/useDriverAlerts';
import { AnalysisResult, BehaviorSummary, Detection, DrivingStats, FatigueSummary, Severity } from './types';

const SEV_RANK: Record<Severity, number> = { none: 0, low: 1, medium: 2, high: 3, critical: 4 };

// 后端地址：默认空串(同源) —— 所有 /api、/health、/ws 请求走当前页面同源，
// 由 dev server 反向代理到后端。这样局域网其他机器只需访问前端这一个端口(3000)，
// 无需直连后端 8000，规避代理/防火墙导致的"后端未链接"。
// 如需直连某个后端，可显式设置 VITE_BACKEND_URL。
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || '';
const CAMERA_FPS = Number(import.meta.env.VITE_CAMERA_FPS || 13); // 上限帧率；实际由后端处理速度自适应

function websocketUrl(path: string) {
  const base = BACKEND_URL || (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:3000');
  const url = new URL(base);
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
  const [scoreHistory, setScoreHistory] = useState<number[]>([]);
  const [drivingSeconds, setDrivingSeconds] = useState(0);
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
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [driverModalOpen, setDriverModalOpen] = useState(false);
  const [view, setView] = useState<'landing' | 'login' | 'app' | 'dashboard' | 'modeldata'>('landing');
  const [user, setUser] = useState<string | null>(() => localStorage.getItem('dms_user'));
  const snapCooldownRef = useRef<Record<string, number>>({});
  const playerRef = useRef<HTMLDivElement | null>(null);

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
    snapCooldownRef.current = {};
    setSnapshots([]);
    setDetections([]);
    setStats(null);
    setLatestResult(null);
    setCurrentBehavior(null);
    setCurrentFatigue(null);
    setBehaviorBoxStyle(null);
    setAnalysisText('');
    setSessionEvents([]);
    setScoreHistory([]);
    setVideoTime(0);
    setDrivingSeconds(0);
  }, []);

  const captureSnapshot = useCallback((label: string, severity: Severity, t: number) => {
    const v = videoRef.current;
    if (!v || v.videoWidth === 0) return;
    const c = document.createElement('canvas');
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(v, 0, 0, c.width, c.height);
    const barH = Math.max(28, Math.round(c.height * 0.06));
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(0, c.height - barH, c.width, barH);
    ctx.fillStyle = '#fff';
    ctx.font = `bold ${Math.round(barH * 0.5)}px sans-serif`;
    ctx.textBaseline = 'middle';
    ctx.fillText(`${label} · ${new Date().toLocaleTimeString('zh-CN')}`, 10, c.height - barH / 2);
    const url = c.toDataURL('image/jpeg', 0.8);
    setSnapshots((prev) => [{ id: `snap-${Date.now()}`, url, label, severity, t, wall: Date.now() }, ...prev].slice(0, 24));
  }, []);

  // 采集 n 帧当前画面(用于车主登记)
  const captureFrames = useCallback(async (n: number): Promise<string[]> => {
    const v = videoRef.current;
    if (!v || v.videoWidth === 0) return [];
    const c = document.createElement('canvas');
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    const ctx = c.getContext('2d');
    if (!ctx) return [];
    const out: string[] = [];
    for (let i = 0; i < n; i++) {
      ctx.drawImage(v, 0, 0, c.width, c.height);
      out.push(c.toDataURL('image/jpeg', 0.85));
      await new Promise((r) => setTimeout(r, 250));
    }
    return out;
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
      setScoreHistory((h) => [...h, result.stats.score].slice(-60));
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
        // 高危行为(high/critical)自动抓拍证据帧，每类 5 秒冷却防刷屏
        if ((SEV_RANK[sev] ?? 0) >= 3) {
          const now = performance.now();
          if (now - (snapCooldownRef.current[key] || 0) > 5000) {
            snapCooldownRef.current[key] = now;
            captureSnapshot(d.type, sev, t);
          }
        }
        // 弹出实时事件提示，3.8s 后自动消失
        setToasts((prev) => [...prev, { id, label: d.type, severity: sev }].slice(-4));
        window.setTimeout(() => setToasts((prev) => prev.filter((x) => x.id !== id)), 3800);
      }
    }
    activeTypesRef.current = current;
  }, [captureSnapshot]);

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

  const toggleFullscreen = useCallback(() => {
    const el = playerRef.current;
    if (!el) return;
    if (document.fullscreenElement) document.exitFullscreen?.();
    else el.requestFullscreen?.();
  }, []);

  // 键盘快捷键：空格 播放/暂停，M 静音切换，F 全屏，C 抓拍
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if (e.code === 'Space' && selectedVideoUrl && !isCameraActive) {
        const v = videoRef.current;
        if (v) {
          e.preventDefault();
          if (v.paused) void v.play();
          else v.pause();
        }
      } else if (e.key === 'm' || e.key === 'M') {
        setSettings((s) => ({ ...s, voice: !s.voice }));
      } else if (e.key === 'f' || e.key === 'F') {
        toggleFullscreen();
      } else if (e.key === 'c' || e.key === 'C') {
        captureSnapshot('手动抓拍', 'none', videoRef.current?.currentTime ?? 0);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selectedVideoUrl, isCameraActive, toggleFullscreen, captureSnapshot]);

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
  const overlayBoxes = (latestResult?.frame_detections ?? latestResult?.detections ?? [])
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

  // 左侧信息栏：当前帧的行为(用于交规提醒) + 行驶计时
  const activeBehaviors = useMemo(() => {
    const types = (latestResult?.frame_detections ?? []).map((d) => d.type);
    const fat = latestResult?.current_fatigue;
    if (fat && fat.risk_level !== 'low' && fat.label) types.push(fat.label);
    return types;
  }, [latestResult]);

  useEffect(() => {
    if (!isAnalyzing) return;
    const t = window.setInterval(() => setDrivingSeconds((s) => s + 1), 1000);
    return () => window.clearInterval(t);
  }, [isAnalyzing]);

  // 提供给 AI 问答的当前驾驶数据上下文
  const chatContext = useMemo(
    () => ({
      综合评分: stats?.score,
      状态: stats?.status,
      驾驶行为评分: stats?.behavior_score,
      疲劳评分: stats?.fatigue_score,
      当前行为: currentBehavior?.label,
      当前疲劳: currentFatigue?.label,
      疲劳信号: currentFatigue?.signals,
      近期事件: sessionEvents.slice(-12).map((e) => ({ 行为: e.label, 严重度: e.severity })),
    }),
    [stats, currentBehavior, currentFatigue, sessionEvents],
  );

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

  if (view === 'landing') {
    return <LandingPage loggedIn={!!user} user={user} onEnter={() => setView(user ? 'app' : 'login')} />;
  }
  if (view === 'login') {
    return <AuthPage onSuccess={(u) => { setUser(u); setView('app'); }} onBack={() => setView('landing')} />;
  }
  if (view === 'dashboard') {
    return <StatsDashboard backendUrl={BACKEND_URL} onBack={() => setView('app')} />;
  }
  if (view === 'modeldata') {
    return <ModelDataPage backendUrl={BACKEND_URL} onBack={() => setView('app')} />;
  }

  return (
    <div className="flex items-center justify-center h-screen bg-gradient-to-br from-slate-200 via-slate-100 to-blue-100/50 p-3 font-sans antialiased">
      <div className="w-full max-w-[1720px] h-full max-h-[1100px] bg-surface text-on-surface flex flex-col shadow-2xl overflow-hidden rounded-2xl border border-white/60 relative">
        <header className="bg-gradient-to-r from-white via-white to-blue-50/60 px-6 h-16 border-b border-outline-variant/60 flex justify-between items-center z-50 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-primary-container flex items-center justify-center shadow-md shadow-primary/30">
              <Film size={18} className="text-white" />
            </div>
            <h1 className="text-2xl font-bold font-display tracking-tight bg-gradient-to-r from-slate-900 to-primary bg-clip-text text-transparent">驾驶状态推演与行为测算平台</h1>
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
              onClick={() => setGalleryOpen(true)}
              title="关键帧抓拍"
              className="relative p-2 rounded-full transition-colors active:scale-90 hover:bg-surface-container"
            >
              <Camera size={22} />
              {snapshots.length > 0 && (
                <span className="absolute -top-0.5 -right-0.5 min-w-[16px] h-4 px-1 rounded-full bg-red-500 text-white text-[10px] font-bold flex items-center justify-center">
                  {snapshots.length}
                </span>
              )}
            </button>
            <button
              onClick={() => setDriverModalOpen(true)}
              title="车主管理"
              className="p-2 rounded-full transition-colors active:scale-90 hover:bg-surface-container"
            >
              <IdCard size={22} />
            </button>
            <button
              onClick={() => setChatOpen(true)}
              title="AI 驾驶助手"
              className="p-2 rounded-full transition-colors active:scale-90 hover:bg-surface-container"
            >
              <Bot size={22} />
            </button>
            <button
              onClick={() => setHistoryOpen(true)}
              title="行程历史"
              className="p-2 rounded-full transition-colors active:scale-90 hover:bg-surface-container"
            >
              <HistoryIcon size={22} />
            </button>
            <button
              onClick={() => setView('dashboard')}
              title="数据看板"
              className="p-2 rounded-full transition-colors active:scale-90 hover:bg-surface-container"
            >
              <BarChart3 size={22} />
            </button>
            <button
              onClick={() => setView('modeldata')}
              title="模型与数据"
              className="p-2 rounded-full transition-colors active:scale-90 hover:bg-surface-container"
            >
              <Database size={22} />
            </button>
            <button
              onClick={() => setSettingsOpen((o) => !o)}
              title="显示设置"
              className={`p-2 rounded-full transition-colors active:scale-90 ${settingsOpen ? 'bg-surface-container text-primary' : 'hover:bg-surface-container'}`}
            >
              <SettingsIcon size={22} />
            </button>
            <div className="flex items-center gap-1.5 pl-2 ml-1 border-l border-outline-variant/50">
              {user && <span className="text-sm font-medium text-slate-600 max-w-[80px] truncate">{user}</span>}
              <button
                onClick={() => { localStorage.removeItem('dms_user'); setUser(null); setView('landing'); }}
                title="退出登录"
                className="p-2 rounded-full transition-colors active:scale-90 hover:bg-surface-container text-slate-500"
              >
                <LogOut size={20} />
              </button>
            </div>

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
              <InfoSidebar drivingSeconds={drivingSeconds} activeBehaviors={activeBehaviors} />
              <div className="flex-1 flex flex-col gap-gutter h-full min-w-0">
                <div className="card flex-1 flex flex-col overflow-hidden relative group">
                  <div className="w-full flex-1 bg-gradient-to-br from-slate-100 via-surface-container to-blue-50/40 relative flex items-center justify-center min-h-0">
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
                        <motion.div ref={playerRef} key="player" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="absolute inset-0 bg-slate-950">
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
                            <span className="px-3 py-1 bg-black/60 backdrop-blur-sm text-white text-xs font-bold rounded max-w-[300px] truncate">
                              {videoLabel}
                            </span>
                            {latestResult?.driver && (latestResult.driver.status === 'known' || latestResult.driver.status === 'unknown') && (
                              <span className={`px-3 py-1 backdrop-blur-sm text-xs font-bold rounded flex items-center gap-1.5 ${latestResult.driver.status === 'known' ? 'bg-emerald-500/85 text-white' : 'bg-amber-500/85 text-white'}`}>
                                <IdCard size={13} />
                                {latestResult.driver.status === 'known' ? `车主 ${latestResult.driver.name}` : '未登记驾驶员'}
                              </span>
                            )}
                            <button
                              onClick={() => captureSnapshot('手动抓拍', 'none', videoRef.current?.currentTime ?? 0)}
                              title="抓拍当前帧 (C)"
                              className="px-2 py-1 bg-black/60 backdrop-blur-sm text-white rounded hover:bg-black/80 active:scale-90 transition"
                            >
                              <Camera size={16} />
                            </button>
                            <button
                              onClick={toggleFullscreen}
                              title="全屏 (F)"
                              className="px-2 py-1 bg-black/60 backdrop-blur-sm text-white rounded hover:bg-black/80 active:scale-90 transition"
                            >
                              <Maximize2 size={16} />
                            </button>
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
                scoreHistory={scoreHistory}
              />
            </div>
          </main>
        </div>

        <UploadModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} onConfirm={handleFileConfirm} />
        <Toasts toasts={toasts} />
        <HistoryModal open={historyOpen} onClose={() => setHistoryOpen(false)} backendUrl={BACKEND_URL} />
        <SnapshotGallery open={galleryOpen} onClose={() => setGalleryOpen(false)} snapshots={snapshots} onClear={() => setSnapshots([])} />
        <AiChatModal open={chatOpen} onClose={() => setChatOpen(false)} backendUrl={BACKEND_URL} context={chatContext} />
        <DriverModal open={driverModalOpen} onClose={() => setDriverModalOpen(false)} backendUrl={BACKEND_URL} />
      </div>
    </div>
  );
}
