import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, BarChart3, Bell, BellOff, Bot, Camera, ChevronDown, Database, FileText, Film, FlaskConical, Focus, HelpCircle, History as HistoryIcon, IdCard, LogOut, Maximize2, Minimize2, Settings as SettingsIcon, WifiOff } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import Sidebar from './components/Sidebar';
import ControlPanel from './components/ControlPanel';
import UploadModal from './components/UploadModal';
import AlertOverlay from './components/AlertOverlay';
import EventTimeline, { TimelineEvent } from './components/EventTimeline';
import EdgeGlow from './components/EdgeGlow';
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
import TrainingLab from './components/TrainingLab';
import TripReportModal from './components/TripReportModal';
import AnimatedBoxes from './components/AnimatedBoxes';
import SyncedVideo, { type DetSet } from './components/SyncedVideo';
import { useDriverAlerts } from './lib/useDriverAlerts';
import { videoBoxStyle } from './lib/videoBox';
import { AnalysisResult, BehaviorSummary, Detection, DrivingStats, FatigueSummary, Severity } from './types';
import { activeBehaviorsFromCurrentFrame, driverStatusFromCurrentFrame } from './statusSelectors';

const SEV_RANK: Record<Severity, number> = { none: 0, low: 1, medium: 2, high: 3, critical: 4 };

// 后端地址：默认空串(同源) —— 所有 /api、/health、/ws 请求走当前页面同源，
// 由 dev server 反向代理到后端。这样局域网其他机器只需访问前端这一个端口(3000)，
// 无需直连后端 8000，规避代理/防火墙导致的"后端未链接"。
// 如需直连某个后端，可显式设置 VITE_BACKEND_URL。
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || '';
const CAMERA_FPS = Number(import.meta.env.VITE_CAMERA_FPS || 30); // 发帧上限(贴合相机原生30fps)；后端实测可达~77fps，余量充足
const SEND_MAX_WIDTH = 640; // 发送帧最大宽度(px)，超过则等比缩小，降低带宽和解码耗时

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

export default function App() {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  // 录制采样配置(可自由选择)：目标数据集 / 时长档位(manual=手动停) / 是否自动预标
  const [recDatasets, setRecDatasets] = useState<{ name: string; label: string }[]>([]);
  const [recDataset, setRecDataset] = useState('realtime_demo');
  const [recDuration, setRecDuration] = useState<'manual' | '10' | '20' | '30'>('10');
  const [recAutolabel, setRecAutolabel] = useState(true);
  const recordingRef = useRef(false);
  const recordedRef = useRef<string[]>([]);
  const recGrabTimerRef = useRef<number | null>(null);
  const recStopTimerRef = useRef<number | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedVideoUrl, setSelectedVideoUrl] = useState<string | null>(null);
  const [stats, setStats] = useState<DrivingStats | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [analysisText, setAnalysisText] = useState('');
  const [latestResult, setLatestResult] = useState<AnalysisResult | null>(null);
  const [liveFps, setLiveFps] = useState(0);
  const [activeModelName, setActiveModelName] = useState('');
  const recvTimesRef = useRef<number[]>([]);
  const [currentBehavior, setCurrentBehavior] = useState<BehaviorSummary | null>(null);
  const [currentFatigue, setCurrentFatigue] = useState<FatigueSummary | null>(null);
  const [behaviorBoxStyle, setBehaviorBoxStyle] = useState<CSSProperties | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [statusMessage, setStatusMessage] = useState('等待真实视频或相机输入');
  const [settings, setSettings] = useState(() => {
    const def = { voice: true, hud: true, scanline: true, statusBar: true, spotlight: true, sync: true };
    try { return { ...def, ...(JSON.parse(localStorage.getItem('dms_settings') || '{}') as Partial<typeof def>) }; } catch { return def; }
  });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [focusMode, setFocusMode] = useState(false); // 专注模式：隐藏外壳只留视频+叠层
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [moreOpen, setMoreOpen] = useState(false); // 顶栏「更多」导航下拉
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
  const [reportOpen, setReportOpen] = useState(false);
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [driverModalOpen, setDriverModalOpen] = useState(false);
  const [view, setView] = useState<'landing' | 'login' | 'app' | 'dashboard' | 'modeldata' | 'traininglab'>('landing');
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
  const inFlightRef = useRef(0); // 当前在途帧数（已发未收结果），限最大 1 避免排队滞后
  const bboxScaleRef = useRef(1); // 发送缩放比（videoWidth / sentWidth），用于还原 bbox 坐标
  const detectionHistoryRef = useRef<Detection[]>([]);
  // —— 视频↔检测同步(丝滑模式) ——
  // 后端逐帧 FIFO 回包(每收一帧回一包、保序)，故"第 N 个结果"必对应"第 N 个送出的帧"。
  // 送帧时压入送检时刻，收包时弹出 → 得到该结果对应帧的精确采集时刻，无需后端改动。
  const pendingCaptureRef = useRef<number[]>([]); // 在途帧的送检时刻队列(FIFO)
  const detSetsRef = useRef<DetSet[]>([]); // 近期每帧检测集合(带送检时刻)，供同步层按时刻配对
  const latEmaRef = useRef(70); // 端到端检测延迟 EMA(ms)
  const syncDelayRef = useRef(70); // 实际显示延迟(ms)：略大于延迟 EMA，保证对应检测已到

  const resetResults = useCallback(() => {
    detectionHistoryRef.current = [];
    pendingCaptureRef.current = [];
    detSetsRef.current = [];
    latEmaRef.current = 70;
    syncDelayRef.current = 70;
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

  // 事件时间轴缩略图：事件发生瞬间抓一张小图(160px宽，jpeg)，挂到该事件供悬停预览/回看
  const captureThumb = useCallback((): string | undefined => {
    const v = videoRef.current;
    if (!v || v.videoWidth === 0) return undefined;
    const c = document.createElement('canvas');
    const w = 160;
    c.width = w;
    c.height = Math.round((w * v.videoHeight) / v.videoWidth);
    const ctx = c.getContext('2d');
    if (!ctx) return undefined;
    ctx.drawImage(v, 0, 0, c.width, c.height);
    return c.toDataURL('image/jpeg', 0.6);
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
    // 发送时把画面缩小到 SEND_MAX_WIDTH，后端返回的所有 bbox 坐标都在缩小后的空间里。
    // 这里统一放大回视频原始尺寸，下游显示逻辑无需感知缩放。
    const sc = bboxScaleRef.current;
    if (sc !== 1) {
      const scaleBbox = (bbox: unknown) =>
        Array.isArray(bbox) ? (bbox as number[]).map((v) => v * sc) : bbox;
      for (const d of result.frame_detections || []) { if (d.bbox) d.bbox = scaleBbox(d.bbox) as number[]; }
      for (const d of result.detections || [])       { if (d.bbox) d.bbox = scaleBbox(d.bbox) as number[]; }
      if (result.current_behavior?.bbox) result.current_behavior.bbox = scaleBbox(result.current_behavior.bbox) as number[];
      if (result.driver?.box)   result.driver.box   = scaleBbox(result.driver.box)   as number[];
      if (result.driver_roi)    result.driver_roi   = scaleBbox(result.driver_roi)   as number[];
    }
    // 视频↔检测同步：记录"该帧检测集合 + 其送检时刻"，供 SyncedVideo 按显示帧时刻精确配对。
    // bbox 此时已还原到视频原始尺寸，与 SyncedVideo 绘制的原始帧坐标系一致。
    const captureT = result.client_capture_t;
    if (captureT != null) {
      detSetsRef.current.push({ t: captureT, dets: result.frame_detections ?? [] });
      if (detSetsRef.current.length > 16) detSetsRef.current.shift();
    }
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
    staleTimerRef.current = window.setTimeout(() => setBoxesFresh(false), 600);
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
        const ev = { id, t, label: d.type, severity: sev, thumb: captureThumb() };
        // 仅在(有上限的)显示状态里保留缩略图；全量 ref 存无 thumb 的轻量副本，
        // 避免长行程里上千条已滚出可视窗口的事件仍各自驻留 ~5KB base64(纯死重)。
        sessionEventsRef.current.push({ id, t, label: d.type, severity: sev });
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
  }, [captureSnapshot, captureThumb]);

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

  // 实时 FPS(近 1 秒收到的结果数) + 当前生效模型名
  useEffect(() => {
    const t = window.setInterval(() => {
      const now = performance.now();
      recvTimesRef.current = recvTimesRef.current.filter((x) => now - x < 1000);
      setLiveFps(recvTimesRef.current.length);
    }, 500);
    return () => window.clearInterval(t);
  }, []);

  useEffect(() => {
    fetch(`${BACKEND_URL}/api/model/active`).then((r) => r.json()).then((d) => setActiveModelName(d.active_model || '')).catch(() => {});
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
      } else if (e.key === 'z' || e.key === 'Z') {
        setFocusMode((f) => !f);
      } else if (e.key === 'Escape') {
        setFocusMode(false);
        setShortcutsOpen(false);
        setMoreOpen(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selectedVideoUrl, isCameraActive, toggleFullscreen, captureSnapshot]);

  // 显示设置持久化(刷新不丢)
  useEffect(() => {
    try { localStorage.setItem('dms_settings', JSON.stringify(settings)); } catch { /* 忽略隐私模式写入失败 */ }
  }, [settings]);

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
      // 入库剔除缩略图(thumb 是大 base64)：仅会话期间内存中用于悬停预览，不进数据库
      events: events.map((e) => ({ id: e.id, t: e.t, label: e.label, severity: e.severity })),
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
    pendingCaptureRef.current = [];
    detSetsRef.current = [];
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
    if (videoRef.current) videoRef.current.onended = null; // 清除上传视频结束回调，避免回放到尾再次触发停止/覆盖状态
    streamModeRef.current = null;
    setIsAnalyzing(false);
  }, [saveSession]);

  const startFrameWebSocket = useCallback(
    async (mode: 'camera' | 'video') => {
      const ws = new WebSocket(websocketUrl('/ws/camera'));
      ws.binaryType = 'arraybuffer';
      websocketRef.current = ws;
      streamModeRef.current = mode;

      const frameInterval = Math.max(20, 1000 / CAMERA_FPS); // 目标发帧间隔(ms)，最低 20ms(50fps)

      // 抓取并发送一帧。成功返回 true，视频未就绪/暂停返回 false。
      // 注意：发送分辨率必须与视频原始尺寸一致，否则后端返回的 bbox 坐标会与
      // 前端叠层坐标系错位，导致识别框位置严重偏差。
      const sendFrame = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const socket = websocketRef.current;
        if (!video || !canvas || !socket || socket.readyState !== WebSocket.OPEN || video.videoWidth === 0) return false;
        if (mode === 'video' && (video.paused || video.ended)) return false;

        // 等比缩小到 SEND_MAX_WIDTH 宽再编码，减少浏览器 JPEG 编码耗时（从 ~30ms→~12ms）。
        // bboxScaleRef 记录缩放比，updateFromResult 里用它把 bbox 坐标还原到视频尺寸。
        const scale = Math.min(1, SEND_MAX_WIDTH / video.videoWidth);
        const sw = Math.round(video.videoWidth * scale);
        const sh = Math.round(video.videoHeight * scale);
        bboxScaleRef.current = video.videoWidth / sw; // e.g., 960/640=1.5
        if (canvas.width !== sw) canvas.width = sw;
        if (canvas.height !== sh) canvas.height = sh;

        const context = canvas.getContext('2d');
        if (!context) return false;
        context.drawImage(video, 0, 0, sw, sh);
        const captureT = performance.now(); // 本帧像素采集时刻，供视频↔检测同步配对
        // 同步递增，确保控流检查在 toBlob 回调到来前就生效
        inFlightRef.current++;
        canvas.toBlob(
          (blob) => {
            if (blob && socket.readyState === WebSocket.OPEN) {
              // 在真正 send 处入队，保证队列顺序 == 发送顺序 == 后端 FIFO 回包顺序
              pendingCaptureRef.current.push(captureT);
              if (pendingCaptureRef.current.length > 32) pendingCaptureRef.current.shift(); // 异常防护(正常 ≤3 在途)
              socket.send(blob);
            } else {
              // 编码失败则回退计数
              inFlightRef.current = Math.max(0, inFlightRef.current - 1);
            }
          },
          'image/jpeg',
          0.72,
        );
        lastSentRef.current = performance.now();
        return true;
      };

      // 持续定时泵：不等后端回包，每隔 frameInterval 都尝试发一帧。
      // 允许最多 2 帧在途做流水线：上一帧还在编码/网络/推理途中就先发下一帧，
      // 让有效帧率不被单帧往返(编码+网络+推理~30-50ms)拖死。后端~14ms 不会积压。
      const pump = () => {
        if (!streamingRef.current) return;
        if (inFlightRef.current <= 2) {
          sendFrame();
        }
        captureTimerRef.current = window.setTimeout(pump, frameInterval);
      };

      ws.onopen = () => {
        setIsAnalyzing(true);
        streamingRef.current = true;
        inFlightRef.current = 0;
        sessionStartRef.current = performance.now();
        setStatusMessage(mode === 'camera' ? '实时相机逐帧检测中' : '上传视频模拟相机逐帧检测中');
        pump();
      };

      ws.onmessage = (event) => {
        inFlightRef.current = Math.max(0, inFlightRef.current - 1);
        // FIFO 配对：弹出最早在途帧的送检时刻，得到本结果对应帧的采集时刻
        const captureT = pendingCaptureRef.current.shift();
        const result = JSON.parse(event.data) as AnalysisResult;
        recvTimesRef.current.push(performance.now()); // 实时 FPS 计数
        if (captureT != null) {
          const lat = performance.now() - captureT; // 端到端延迟(编码+网络+推理+回传)
          // 拒绝瞬时尖峰(GC/切后台/抖动)：仅接受相对当前均值合理的样本，避免单次 0.5~0.9s
          // 卡顿把 EMA 拉高、显示延迟被顶到 240ms 持续约 1s。+120 下限允许稳态延迟逐步抬升。
          const cur = latEmaRef.current;
          if (lat > 0 && lat < Math.min(1000, cur * 3 + 120)) latEmaRef.current = cur * 0.85 + lat * 0.15;
          // 显示延迟略大于延迟均值(+12ms 余量)，确保显示某帧时其检测已到；限幅 45~240ms
          syncDelayRef.current = Math.min(240, Math.max(45, latEmaRef.current * 1.1 + 12));
          result.client_capture_t = captureT;
        }
        if (!result.error) updateFromResult(result);
        else setStatusMessage(result.error);
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
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      setStatusMessage('当前访问地址不是安全来源，浏览器禁止调用相机；请使用 https:// 地址或在本机用 localhost 访问');
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
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('当前为非安全上下文(HTTP)，浏览器禁用了摄像头。请改用 https:// 访问，或在本机用 http://localhost 访问。');
      }
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540, frameRate: { ideal: 30 } }, audio: false });
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

  // 录制采样：在真实座舱采多样数据是根治手机/动作漏检(模型泛化差)的正路。
  // 支持手动开始/停止(时长自定)或固定档位(10/20/30s)，可选目标数据集与是否自动预标。
  // 取数据集列表供下拉选择
  useEffect(() => {
    if (!backendOnline) return;
    fetch(`${BACKEND_URL}/api/projects`)
      .then((r) => r.json())
      .then((d: { projects?: { name: string; label: string }[] }) => {
        const ds = d.projects || [];
        setRecDatasets(ds);
        if (ds.length && !ds.some((p) => p.name === recDataset)) setRecDataset(ds[0].name);
      })
      .catch(() => {});
  }, [backendOnline]); // eslint-disable-line react-hooks/exhaustive-deps

  const uploadRecorded = useCallback(async () => {
    const frames = recordedRef.current;
    recordedRef.current = [];
    if (!frames.length) {
      setStatusMessage('未捕获到画面，录制取消');
      return;
    }
    setStatusMessage(`上传 ${frames.length} 帧入库…`);
    try {
      const res = await fetch(`${BACKEND_URL}/api/dataset/capture_frames`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset: recDataset, split: 'unassigned', frames, autolabel: recAutolabel }),
      }).then((r) => r.json());
      if (res.detail) setStatusMessage(`录制入库失败：${res.detail}`);
      else setStatusMessage(
        `✅ 已录制 ${res.saved} 帧到「${res.dataset}」` +
        (res.skipped ? `(跳过 ${res.skipped} 帧重复)` : '') +
        (recAutolabel ? `(预标注 ${res.labeled} 帧/${res.boxes} 框)` : '(未预标，全手动)') +
        `。请到「模型与数据 → 逐张浏览/编辑标注」修正后重训`,
      );
    } catch {
      setStatusMessage('录制失败：网络或后端异常');
    }
  }, [recDataset, recAutolabel]);

  const stopRecording = useCallback(() => {
    recordingRef.current = false;
    if (recGrabTimerRef.current) { window.clearTimeout(recGrabTimerRef.current); recGrabTimerRef.current = null; }
    if (recStopTimerRef.current) { window.clearTimeout(recStopTimerRef.current); recStopTimerRef.current = null; }
    setIsRecording(false);
    void uploadRecorded();
  }, [uploadRecorded]);

  const startRecording = useCallback(() => {
    if (!isCameraActive) return;
    recordedRef.current = [];
    recordingRef.current = true;
    setIsRecording(true);
    const canvas = document.createElement('canvas');
    // 每 250ms 抓一帧(~4fps)，上限 360 帧(~90s)防失控
    const grab = () => {
      if (!recordingRef.current) return;
      const v = videoRef.current;
      if (v && v.videoWidth) {
        canvas.width = v.videoWidth; canvas.height = v.videoHeight;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
          recordedRef.current.push(canvas.toDataURL('image/jpeg', 0.85));
        }
      }
      if (recordedRef.current.length >= 360) { stopRecording(); return; }
      setStatusMessage(`录制中…已 ${recordedRef.current.length} 帧${recDuration === 'manual' ? '（再次点击停止）' : ''}`);
      recGrabTimerRef.current = window.setTimeout(grab, 250);
    };
    grab();
    if (recDuration !== 'manual') {
      recStopTimerRef.current = window.setTimeout(stopRecording, Number(recDuration) * 1000);
    }
  }, [isCameraActive, recDuration, stopRecording]);

  const toggleRecord = useCallback(() => {
    if (recordingRef.current) stopRecording();
    else startRecording();
  }, [startRecording, stopRecording]);

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

  // 驾驶员区域(聚光) + 车主脸框 —— 展示"只算驾驶员"的收口
  const driverRoiStyle = settings.spotlight && isAnalyzing ? videoBoxStyle(videoRef.current, latestResult?.driver_roi ?? null) : null;
  const driverFaceStyle = settings.spotlight && isAnalyzing ? videoBoxStyle(videoRef.current, latestResult?.driver?.box ?? null) : null;
  const driverKnown = latestResult?.driver?.status === 'known';
  // 同步焊框生效条件：开启同步 + 正在分析 + (相机 或 上传视频)。上传视频也焊框，便于离线演示/检验效果。
  const useSyncedView = settings.sync && isAnalyzing && (isCameraActive || !!selectedVideoUrl);

  // 由当前帧检测推导驾驶员状态指示灯(正常/告警)
  const driverStatus = useMemo<DriverStatus | null>(() => {
    return driverStatusFromCurrentFrame(latestResult);
  }, [latestResult]);

  // 左侧信息栏：当前帧的行为(用于交规提醒) + 行驶计时
  const activeBehaviors = useMemo(() => {
    return activeBehaviorsFromCurrentFrame(latestResult);
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
    <div className={`absolute left-4 right-4 grid grid-cols-2 gap-3 pointer-events-none transition-all ${isAnalyzing && alert.level !== 'none' ? 'top-[88px]' : 'top-16'}`}>
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
            {/* EAR(眼睛纵横比)：越低越闭眼，琥珀竖线=闭眼阈值 0.18；闭眼时变红 */}
            <div className="flex items-center gap-2 text-[11px]">
              <span className="shrink-0 font-bold">EAR</span>
              <div className="relative flex-1 h-1.5 rounded-full bg-white/25 overflow-hidden">
                <div
                  className={`h-full rounded-full ${currentFatigue.signals.eyes_closed ? 'bg-red-300' : 'bg-white/90'}`}
                  style={{ width: `${Math.min(100, Math.round(((currentFatigue.signals.ear ?? 0) / 0.4) * 100))}%`, transition: 'width 0.3s ease' }}
                />
                <div className="absolute top-0 bottom-0 w-px bg-amber-300/90" style={{ left: `${(0.18 / 0.4) * 100}%` }} />
              </div>
              <span className="tabular-nums w-9 text-right">{(currentFatigue.signals.ear ?? 0).toFixed(2)}</span>
            </div>
            <div className="flex items-center justify-between text-[11px]">
              <span className="opacity-90">头部翻滚 roll {Math.round(currentFatigue.signals.head_roll ?? 0)}°</span>
              {currentFatigue.signals.eyes_closed && <span className="px-1.5 py-0.5 rounded bg-red-500/40 text-red-50 font-bold">● 闭眼</span>}
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
  if (view === 'traininglab') {
    return <TrainingLab backendUrl={BACKEND_URL} onBack={() => setView('app')} />;
  }

  return (
    <div className={focusMode ? 'h-screen bg-slate-950 font-sans antialiased' : 'flex items-center justify-center h-screen bg-gradient-to-br from-slate-200 via-slate-100 to-blue-100/50 p-3 font-sans antialiased'}>
      <div className={focusMode ? 'w-full h-full bg-surface text-on-surface flex flex-col overflow-hidden relative' : 'w-full max-w-[1720px] h-full max-h-[1100px] bg-surface text-on-surface flex flex-col shadow-2xl overflow-hidden rounded-2xl border border-white/60 relative'}>
        {!focusMode && (
        <header className="bg-gradient-to-r from-white via-white to-blue-50/60 px-6 h-16 border-b border-outline-variant/60 flex justify-between items-center z-50 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-primary-container flex items-center justify-center shadow-md shadow-primary/30">
              <Film size={18} className="text-white" />
            </div>
            <h1 className="text-2xl font-bold font-display tracking-tight bg-gradient-to-r from-slate-900 to-primary bg-clip-text text-transparent">驾驶状态推演与行为测算平台</h1>
          </div>
          <div className="flex items-center gap-1.5 text-primary relative">
            <button
              onClick={() => setSettings((s) => ({ ...s, voice: !s.voice }))}
              title={alertsMuted ? '语音告警已静音，点击开启' : '语音告警已开启，点击静音'}
              aria-label="语音告警开关"
              aria-pressed={!alertsMuted}
              className={`p-2 rounded-full transition-colors active:scale-90 ${alertsMuted ? 'text-slate-400 hover:bg-surface-container' : 'text-primary hover:bg-surface-container'}`}
            >
              {alertsMuted ? <BellOff size={22} /> : <Bell size={22} />}
            </button>
            <span className="w-px h-5 bg-outline-variant/50 mx-0.5" />
            <button
              onClick={() => setGalleryOpen(true)}
              title="关键帧抓拍"
              aria-label="关键帧抓拍"
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
              aria-label="车主管理"
              className="p-2 rounded-full transition-colors active:scale-90 hover:bg-surface-container"
            >
              <IdCard size={22} />
            </button>
            <button
              onClick={() => setChatOpen(true)}
              title="AI 驾驶助手"
              aria-label="AI 驾驶助手"
              className="p-2 rounded-full transition-colors active:scale-90 hover:bg-surface-container"
            >
              <Bot size={22} />
            </button>
            <button
              onClick={() => setReportOpen(true)}
              title="行程报告"
              aria-label="行程报告"
              className="p-2 rounded-full transition-colors active:scale-90 hover:bg-surface-container"
            >
              <FileText size={22} />
            </button>
            <button
              onClick={() => setHistoryOpen(true)}
              title="行程历史"
              aria-label="行程历史"
              className="p-2 rounded-full transition-colors active:scale-90 hover:bg-surface-container"
            >
              <HistoryIcon size={22} />
            </button>
            <span className="w-px h-5 bg-outline-variant/50 mx-0.5" />
            {/* 导航类页面收进「更多」下拉，给顶栏减负 */}
            <div className="relative">
              <button
                onClick={() => setMoreOpen((o) => !o)}
                title="更多页面：数据看板 / 模型与数据 / 训练实验室"
                aria-label="更多页面"
                aria-expanded={moreOpen}
                className={`px-2.5 py-2 rounded-full transition-colors active:scale-90 inline-flex items-center gap-1 text-sm font-medium ${moreOpen ? 'bg-surface-container text-primary' : 'hover:bg-surface-container'}`}
              >
                更多 <ChevronDown size={16} className={`transition-transform ${moreOpen ? 'rotate-180' : ''}`} />
              </button>
              {moreOpen && (
                <>
                  <div className="fixed inset-0 z-40" onClick={() => setMoreOpen(false)} />
                  <div className="absolute right-0 top-12 z-50 w-52 rounded-xl border border-outline-variant bg-white shadow-2xl p-2">
                    {([
                      [BarChart3, '数据看板', 'dashboard'],
                      [Database, '模型与数据', 'modeldata'],
                      [FlaskConical, '训练实验室', 'traininglab'],
                    ] as const).map(([Icon, label, v]) => (
                      <button
                        key={v}
                        onClick={() => { setMoreOpen(false); setView(v); }}
                        className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg hover:bg-surface-container transition-colors text-sm font-medium text-slate-700"
                      >
                        <Icon size={18} className="text-primary" /> {label}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
            <span className="w-px h-5 bg-outline-variant/50 mx-0.5" />
            <button
              onClick={() => setFocusMode(true)}
              title="专注模式：隐藏侧栏只看视频 (Z)"
              aria-label="进入专注模式"
              className="p-2 rounded-full transition-colors active:scale-90 hover:bg-surface-container"
            >
              <Focus size={22} />
            </button>
            <button
              onClick={() => setSettingsOpen((o) => !o)}
              title="显示设置"
              aria-label="显示设置"
              aria-expanded={settingsOpen}
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
                    ['sync', '视频↔检测同步 (丝滑焊框)'],
                    ['voice', '语音告警播报'],
                    ['hud', '机器视觉 HUD'],
                    ['scanline', '扫描线动效'],
                    ['statusBar', '驾驶员状态灯'],
                    ['spotlight', '聚焦驾驶员区域'],
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
        )}

        {!backendOnline && (
          <div className="shrink-0 flex items-center justify-center gap-2 px-4 py-2 bg-red-600 text-white text-sm font-medium z-40">
            <WifiOff size={16} /> 后端服务未连接 —— 实时检测 / 上传 / 训练等功能不可用，正在自动重试…
          </div>
        )}

        <div className="flex-1 flex overflow-hidden">
          <main className="flex-1 p-margin overflow-y-auto bg-surface-container-low">
            <div className="flex gap-gutter h-full">
              {!focusMode && <InfoSidebar drivingSeconds={drivingSeconds} activeBehaviors={activeBehaviors} />}
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
                            controls={!isCameraActive && !useSyncedView}
                            autoPlay={isCameraActive}
                            onLoadedMetadata={() => setBehaviorBoxStyle(videoBoxStyle(videoRef.current, currentBehavior?.bbox ?? null))}
                            playsInline
                          />
                          {/* 识别框层：同步开+分析中(相机或上传视频) → 延迟画面对齐检测(焊框，SyncedVideo)；
                              否则 → 运动外推 60fps 跟随(AnimatedBoxes)。可在显示设置一键切换对比。 */}
                          {useSyncedView ? (
                            <SyncedVideo
                              videoRef={videoRef}
                              enabled={isAnalyzing}
                              detSetsRef={detSetsRef}
                              delayMsRef={syncDelayRef}
                            />
                          ) : (
                            <AnimatedBoxes
                              videoRef={videoRef}
                              detections={latestResult?.frame_detections ?? latestResult?.detections ?? []}
                              fresh={boxesFresh}
                            />
                          )}
                          {/* 驾驶员聚光区 + 车主脸框：直观展示"只算驾驶员、忽略背景" */}
                          {driverRoiStyle && (
                            <div className="absolute pointer-events-none" style={{ ...(driverRoiStyle as CSSProperties), boxShadow: '0 0 0 9999px rgba(2,6,23,0.42)', border: '1.5px dashed #22d3ee', borderRadius: 6 }}>
                              <div className="absolute left-0 top-0 -translate-y-full px-2 py-0.5 text-[11px] font-bold text-slate-900 bg-cyan-300 rounded-t whitespace-nowrap">
                                驾驶员区域 · 仅此区域计分
                              </div>
                            </div>
                          )}
                          {driverFaceStyle && (
                            <div className="absolute pointer-events-none rounded" style={{ ...(driverFaceStyle as CSSProperties), border: `2px solid ${driverKnown ? '#34d399' : '#fbbf24'}`, boxShadow: `0 0 12px ${driverKnown ? '#34d39988' : '#fbbf2488'}` }}>
                              <div className="absolute left-0 bottom-0 translate-y-full px-1.5 text-[10px] font-bold text-white whitespace-nowrap" style={{ background: driverKnown ? '#34d399' : '#fbbf24' }}>
                                {driverKnown ? `车主 ${latestResult?.driver?.name}` : '驾驶员(未登记)'}
                              </div>
                            </div>
                          )}
                          {settings.hud && (
                            <VideoHud active={isAnalyzing} scan={settings.scanline} latencyMs={latestResult?.metrics?.latency_ms} frameId={latestResult?.frame_id} fps={liveFps} model={activeModelName} syncMs={useSyncedView ? syncDelayRef.current : null} driverLocked={!!latestResult?.driver_roi || !!latestResult?.driver?.box || !!latestResult?.driver_present} bannerActive={isAnalyzing && alert.level !== 'none'} />
                          )}
                          {settings.statusBar && <DriverStatusBar status={driverStatus} />}
                          {realtimeOverlays}
                          <EdgeGlow level={isAnalyzing ? alert.level : 'none'} active={isAnalyzing} />
                          {/* 分析停止后(上传视频播完)gate 掉告警，避免横幅/边缘光残留上一帧的陈旧告警 */}
                          <AlertOverlay level={isAnalyzing ? alert.level : 'none'} text={alert.text} muted={alertsMuted} />
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
                              aria-label="全屏"
                              className="px-2 py-1 bg-black/60 backdrop-blur-sm text-white rounded hover:bg-black/80 active:scale-90 transition"
                            >
                              <Maximize2 size={16} />
                            </button>
                            <button
                              onClick={() => setFocusMode((f) => !f)}
                              title="专注模式 (Z)"
                              aria-label="切换专注模式"
                              className="px-2 py-1 bg-black/60 backdrop-blur-sm text-white rounded hover:bg-black/80 active:scale-90 transition"
                            >
                              {focusMode ? <Minimize2 size={16} /> : <Focus size={16} />}
                            </button>
                            <div className="relative">
                              <button
                                onClick={() => setShortcutsOpen((o) => !o)}
                                title="快捷键"
                                aria-label="快捷键说明"
                                className="px-2 py-1 bg-black/60 backdrop-blur-sm text-white rounded hover:bg-black/80 active:scale-90 transition"
                              >
                                <HelpCircle size={16} />
                              </button>
                              {shortcutsOpen && (
                                <div className="absolute left-0 top-full mt-1.5 z-40 w-56 rounded-lg bg-slate-900/95 text-slate-200 text-[11px] p-2.5 shadow-xl space-y-1" onMouseLeave={() => setShortcutsOpen(false)}>
                                  <div className="font-bold text-white mb-1">快捷键</div>
                                  {[['空格', '播放/暂停(上传视频)'], ['M', '语音告警静音'], ['F', '全屏'], ['C', '抓拍当前帧'], ['Z', '专注模式'], ['Esc', '退出专注/关闭弹层']].map(([k, d]) => (
                                    <div key={k} className="flex items-center justify-between gap-3">
                                      <kbd className="px-1.5 py-0.5 rounded bg-white/15 font-mono">{k}</kbd>
                                      <span className="opacity-85">{d}</span>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
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

                  {!focusMode && (
                    <ControlPanel
                      onUpload={() => setIsModalOpen(true)}
                      onStartUploadAnalysis={handleStartUploadAnalysis}
                      onToggleCamera={handleToggleCamera}
                      onRecordSamples={toggleRecord}
                      isRecording={isRecording}
                      recDatasets={recDatasets}
                      recDataset={recDataset}
                      onRecDatasetChange={setRecDataset}
                      recDuration={recDuration}
                      onRecDurationChange={setRecDuration}
                      recAutolabel={recAutolabel}
                      onRecAutolabelChange={setRecAutolabel}
                      isAnalyzing={isAnalyzing}
                      isCameraActive={isCameraActive}
                      hasSelectedFile={Boolean(selectedFile)}
                      backendOnline={backendOnline}
                    />
                  )}
                </div>

                {!focusMode && (selectedVideoUrl || isCameraActive) && (
                  <EventTimeline
                    events={sessionEvents}
                    duration={isCameraActive ? videoTime : videoDuration}
                    currentTime={videoTime}
                    seekable={Boolean(selectedVideoUrl) && !isCameraActive}
                    onSeek={handleSeek}
                  />
                )}
              </div>

              {!focusMode && (
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
              )}
            </div>
          </main>
        </div>

        <UploadModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} onConfirm={handleFileConfirm} />
        <Toasts toasts={toasts} />
        <HistoryModal open={historyOpen} onClose={() => setHistoryOpen(false)} backendUrl={BACKEND_URL} />
        <TripReportModal
          open={reportOpen}
          onClose={() => setReportOpen(false)}
          score={stats?.score ?? 0}
          behaviorScore={stats?.behavior_score}
          fatigueScore={stats?.fatigue_score}
          drivingSeconds={drivingSeconds}
          driverName={latestResult?.driver?.name}
          scoreHistory={scoreHistory}
          events={sessionEvents}
          snapshots={snapshots}
          review={latestResult?.llm_analysis || analysisText}
          source={videoLabel}
        />
        <SnapshotGallery open={galleryOpen} onClose={() => setGalleryOpen(false)} snapshots={snapshots} onClear={() => setSnapshots([])} />
        <AiChatModal open={chatOpen} onClose={() => setChatOpen(false)} backendUrl={BACKEND_URL} context={chatContext} />
        <DriverModal open={driverModalOpen} onClose={() => setDriverModalOpen(false)} backendUrl={BACKEND_URL} />
      </div>
    </div>
  );
}
