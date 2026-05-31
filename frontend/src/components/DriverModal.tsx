import { useEffect, useRef, useState } from 'react';
import { Camera, CameraOff, IdCard, Trash2, UserPlus, X } from 'lucide-react';

interface Props {
  open: boolean;
  onClose: () => void;
  backendUrl: string;
}

/** 车主管理：弹窗内置摄像头预览，直接采集人脸登记;识别走后端 LBPH。*/
export default function DriverModal({ open, onClose, backendUrl }: Props) {
  const [drivers, setDrivers] = useState<string[]>([]);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState('');
  const [camOn, setCamOn] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const refresh = () =>
    fetch(`${backendUrl}/api/driver/list`).then((r) => r.json()).then((d) => setDrivers(d.drivers || [])).catch(() => {});

  const startCam = async () => {
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCamOn(false);
        setMsg('非安全上下文(HTTP)无法用摄像头，请用 https:// 或本机 http://localhost 访问');
        return;
      }
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCamOn(true);
      setMsg('');
    } catch {
      setCamOn(false);
      setMsg('无法打开摄像头，请检查权限');
    }
  };
  const stopCam = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCamOn(false);
  };

  useEffect(() => {
    if (open) {
      refresh();
      setMsg('');
      setName('');
      void startCam();
    } else {
      stopCam();
    }
    return stopCam;
  }, [open]);

  if (!open) return null;

  const captureFrames = async (n: number): Promise<string[]> => {
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
      await new Promise((r) => setTimeout(r, 220));
    }
    return out;
  };

  const register = async () => {
    if (!name.trim()) return setMsg('请先输入车主姓名');
    if (!camOn) return setMsg('摄像头未开启');
    setBusy(true);
    setMsg('正在采集人脸…请正对摄像头');
    try {
      const images = await captureFrames(8);
      const r = await fetch(`${backendUrl}/api/driver/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim(), images }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || '登记失败');
      setMsg(`✅ 已登记「${d.name}」，采集到 ${d.saved} 张人脸`);
      setName('');
      setDrivers(d.drivers || []);
    } catch (e) {
      setMsg(`❌ ${e instanceof Error ? e.message : '登记失败'}`);
    } finally {
      setBusy(false);
    }
  };

  const del = (n: string) =>
    fetch(`${backendUrl}/api/driver/${encodeURIComponent(n)}`, { method: 'DELETE' }).then(() => refresh());

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 backdrop-blur-sm" onClick={onClose}>
      <div className="w-[760px] bg-white rounded-2xl shadow-2xl overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <div className="px-5 h-14 flex items-center justify-between border-b border-outline-variant">
          <h3 className="font-display font-bold text-slate-800 flex items-center gap-2"><IdCard size={18} className="text-primary" /> 车主管理</h3>
          <button onClick={onClose} className="p-1.5 rounded-full hover:bg-surface-container text-slate-500"><X size={20} /></button>
        </div>

        <div className="flex">
          {/* 摄像头预览 */}
          <div className="w-[360px] shrink-0 bg-slate-900 relative flex items-center justify-center" style={{ minHeight: 300 }}>
            <video ref={videoRef} muted playsInline className="w-full h-full object-cover" />
            {!camOn && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400 gap-2">
                <CameraOff size={36} />
                <button onClick={startCam} className="text-sm text-blue-300 underline">开启摄像头</button>
              </div>
            )}
            {/* 人脸取景框引导 */}
            {camOn && (
              <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                <div className="w-40 h-48 border-2 border-cyan-300/70 rounded-[40%] shadow-[0_0_20px_rgba(34,211,238,0.4)]" />
              </div>
            )}
            {busy && <div className="absolute bottom-3 left-0 right-0 text-center text-cyan-200 text-sm animate-pulse">采集中…</div>}
          </div>

          {/* 登记 + 列表 */}
          <div className="flex-1 p-5 space-y-4">
            <div>
              <div className="text-sm font-bold text-slate-600 mb-2 flex items-center gap-1"><Camera size={15} className="text-primary" /> 登记新车主</div>
              <div className="flex gap-2">
                <input value={name} onChange={(e) => setName(e.target.value)} placeholder="车主姓名" className="flex-1 px-3 py-2 rounded-lg border border-outline-variant text-sm focus:outline-none focus:border-primary" />
                <button onClick={register} disabled={busy || !camOn} className="px-4 rounded-lg bg-primary text-white text-sm font-medium disabled:opacity-50 flex items-center gap-1">
                  <UserPlus size={16} /> {busy ? '采集中' : '登记'}
                </button>
              </div>
              <p className="text-xs text-slate-400 mt-1.5">正对镜头(置于椭圆框内)，点击登记后系统采集 8 张人脸样本。</p>
              {msg && <p className="text-xs mt-2 text-slate-600">{msg}</p>}
            </div>

            <div>
              <div className="text-sm font-bold text-slate-600 mb-2">已登记车主（{drivers.length}）</div>
              {drivers.length === 0 ? (
                <p className="text-sm text-slate-400">暂无登记车主</p>
              ) : (
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {drivers.map((d) => (
                    <div key={d} className="flex items-center justify-between px-3 py-2 rounded-lg bg-slate-50 border border-outline-variant/50">
                      <span className="text-sm font-medium text-slate-700 flex items-center gap-2"><IdCard size={15} className="text-primary" />{d}</span>
                      <button onClick={() => del(d)} className="text-red-500 hover:text-red-600"><Trash2 size={15} /></button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
