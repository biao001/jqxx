import { type ChangeEvent, type ReactNode, useCallback, useEffect, useState } from 'react';
import { AlertTriangle, ArrowLeft, CheckCircle2, Cpu, Database, Download, FolderPlus, Images, type LucideIcon, Pencil, Rocket, SlidersHorizontal, Sparkles, Trash2, Upload, Wrench } from 'lucide-react';
import DatasetBrowser from './DatasetBrowser';

interface ClassMetric { map50: number; map: number; p?: number; r?: number }
interface PerClass { epoch?: number; map50?: number; map?: number; per_class?: Record<string, ClassMetric> }
// 训练溯源：<model>.meta.json —— 这一轮模型基于哪个数据集/权重/参数训出 + 最终各类别 mAP
interface TrainMeta {
  dataset?: string; base?: string; epochs?: number; imgsz?: number;
  freeze?: number; lr0?: number; all_splits?: boolean; include_incremental?: boolean;
  finished_at?: number; map50?: number; map?: number; per_class?: Record<string, ClassMetric>;
}

interface TrainStatus {
  running: boolean;
  started: boolean;
  project?: string;
  out_model?: string;
  base?: string;
  per_class?: PerClass;
  epoch: number;
  total_epochs: number;
  best_map: number;
  finished: boolean;
  success: boolean;
  elapsed: number;
  is_active?: boolean;
  active_model?: string;
  log_tail: string;
}

interface ProjectStat {
  name: string;
  label: string;
  model: string;
  builtin: boolean;
  dataset_exists: boolean;
  images: Record<string, number>;
  model_exists: boolean;
  model_size_mb: number;
  active: boolean;
  meta?: TrainMeta | null;
}

interface ModelFile {
  model: string;
  label: string;
  registered: boolean; // 是否关联注册项目（false=手动拷入 models/ 的散装 .pt）
  size_mb: number;
  mtime: number;
  active: boolean;
  meta?: TrainMeta | null;
}

interface Props {
  backendUrl: string;
  onBack: () => void;
}

interface DatasetInfo {
  exists: boolean;
  names?: string[];
  num_classes?: number;
  classes?: { cls?: number; name: string; count: number }[];
  splits?: Record<string, { images: number; labels: number }>;
  bbox_quality?: { tiny: number; normal: number; huge: number };
  total_instances?: number;
}

const PARAM_LABELS: Record<string, string> = {
  conf: '通用检测阈值',
  smoking_conf: '吸烟阈值',
  phone_use_conf: '手机阈值',
  drink_eat_conf: '饮水/进食阈值',
};
const TRAIN_LABELS: Record<string, string> = {
  base: '基础权重',
  epochs: '训练轮数',
  imgsz: '输入尺寸',
  batch: '批大小',
  patience: '早停耐心',
};

function Card({ icon: Icon, title, desc, children }: { icon: LucideIcon; title: string; desc?: string; children: ReactNode }) {
  return (
    <div className="card p-6">
      <div className="flex items-center gap-2 mb-1">
        <Icon size={19} className="text-primary" />
        <h3 className="font-display text-lg font-bold text-slate-800">{title}</h3>
      </div>
      {desc && <p className="text-xs text-slate-400 mb-4">{desc}</p>}
      {!desc && <div className="mb-4" />}
      {children}
    </div>
  );
}

/** 分区标题：把数据 / 模型 / 训练 / 运行时阈值 这几件独立的事在视觉上分开。*/
function SectionTitle({ children }: { children: ReactNode }) {
  return <h2 className="text-[11px] font-bold tracking-[0.2em] text-slate-400 uppercase pt-5 pb-0.5 px-1">{children}</h2>;
}

function Toggle({ on, onClick, label, hint }: { on: boolean; onClick: () => void; label: string; hint?: string }) {
  return (
    <button type="button" onClick={onClick} className="flex items-center justify-between w-full gap-3 px-1 py-1.5 text-left">
      <span className="min-w-0">
        <span className="text-sm font-medium text-slate-700">{label}</span>
        {hint && <span className="block text-[11px] text-slate-400 leading-snug">{hint}</span>}
      </span>
      <span className={`relative shrink-0 w-9 h-5 rounded-full transition-colors ${on ? 'bg-primary' : 'bg-slate-300'}`}>
        <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all ${on ? 'left-[18px]' : 'left-0.5'}`} />
      </span>
    </button>
  );
}

/** 把模型的训练溯源(meta)拼成一行：基于哪个数据集/权重训出、mAP、是否演示/含增量。*/
function metaText(meta?: TrainMeta | null): string | null {
  if (!meta) return null;
  const bits: string[] = [];
  if (meta.dataset) bits.push(`基于 ${meta.dataset}`);
  if (meta.base) bits.push(String(meta.base).replace('models/', ''));
  if (meta.map50 != null) bits.push(`mAP50 ${(meta.map50 * 100).toFixed(0)}%`);
  if (meta.all_splits) bits.push('演示·过拟合');
  if (meta.include_incremental) bits.push('含增量');
  return bits.length ? bits.join(' · ') : null;
}

function ModelMeta({ meta }: { meta?: TrainMeta | null }) {
  const t = metaText(meta);
  return t ? <div className="text-[11px] text-slate-400 truncate">{t}</div> : null;
}

export default function ModelDataPage({ backendUrl, onBack }: Props) {
  const [ds, setDs] = useState<DatasetInfo | null>(null);
  const [params, setParams] = useState<Record<string, number>>({});
  const [train, setTrain] = useState<Record<string, string | number | boolean>>({});
  const [savedMsg, setSavedMsg] = useState('');
  const [uploadMsg, setUploadMsg] = useState('');
  const [browserOpen, setBrowserOpen] = useState(false);
  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null);
  const [projects, setProjects] = useState<ProjectStat[]>([]);
  const [modelFiles, setModelFiles] = useState<ModelFile[]>([]);
  const [project, setProject] = useState('reckless_mapped');
  const [mergeSrc, setMergeSrc] = useState(''); // 「并入其他数据集」的来源项目
  const [augOps, setAugOps] = useState<Record<string, boolean>>({ rotate: true, flip: true, noise: true, brightness: true, blur: false });
  const [augFactor, setAugFactor] = useState(2);
  const [augDeg, setAugDeg] = useState(15);
  const [augMsg, setAugMsg] = useState('');
  const [augBusy, setAugBusy] = useState(false);
  const [augMode, setAugMode] = useState<'fixed' | 'balance'>('fixed'); // 固定份数 / 均衡到目标
  const [augTarget, setAugTarget] = useState(100);                       // 均衡模式每类目标框数
  const [augClasses, setAugClasses] = useState<Record<number, boolean>>({}); // 选中的目标类(空=全部)
  const [augNoise, setAugNoise] = useState(18);                          // 噪声 σ 上限
  const [augBright, setAugBright] = useState(30);                        // 亮度幅度上限
  const [augBlurK, setAugBlurK] = useState(9);                           // 运动模糊最大核
  const [augPreviews, setAugPreviews] = useState<{ src: string; boxes: number; image: string }[]>([]);

  const loadProjects = useCallback(() => {
    fetch(`${backendUrl}/api/projects`).then((r) => r.json()).then((d) => setProjects(d.projects || [])).catch(() => {});
  }, [backendUrl]);

  const loadModels = useCallback(() => {
    fetch(`${backendUrl}/api/models`).then((r) => r.json()).then((d) => setModelFiles(d.models || [])).catch(() => {});
  }, [backendUrl]);

  const loadDataset = useCallback(() => {
    fetch(`${backendUrl}/api/dataset/info?dataset=${project}`).then((r) => r.json()).then((d) => setDs(d.active)).catch(() => {});
  }, [backendUrl, project]);

  useEffect(() => {
    loadProjects();
    loadModels();
    fetch(`${backendUrl}/api/detector/params`).then((r) => r.json()).then(setParams).catch(() => {});
    fetch(`${backendUrl}/api/training/config`).then((r) => r.json()).then(setTrain).catch(() => {});
  }, [backendUrl, loadProjects, loadModels]);

  useEffect(() => { loadDataset(); }, [loadDataset]);

  useEffect(() => {
    const poll = () => fetch(`${backendUrl}/api/training/status`).then((r) => r.json()).then(setTrainStatus).catch(() => {});
    poll();
    const t = setInterval(poll, 3000);
    return () => clearInterval(t);
  }, [backendUrl]);

  const startTrain = () =>
    fetch(`${backendUrl}/api/training/start`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ dataset: project }) })
      .then((r) => r.json()).then(setTrainStatus).catch(() => {});

  const createProject = () => {
    const name = window.prompt('新数据集英文名(小写字母开头，仅含小写字母/数字/下划线)：');
    if (!name) return;
    const label = window.prompt('显示名称(中文可)：', name) || name;
    fetch(`${backendUrl}/api/projects`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, label }) })
      .then((r) => r.json())
      .then((d: { detail?: string; created?: { name: string }; projects?: ProjectStat[] }) => {
        if (d.detail) { setSavedMsg(`❌ ${d.detail}`); setTimeout(() => setSavedMsg(''), 3000); return; }
        if (d.projects) setProjects(d.projects);
        if (d.created) setProject(d.created.name);
        setSavedMsg('✅ 已创建空数据集，可在上方浏览器上传图片并标注'); setTimeout(() => setSavedMsg(''), 3000);
      })
      .catch(() => { setSavedMsg('❌ 创建失败'); setTimeout(() => setSavedMsg(''), 3000); });
  };

  // 组装增强公共入参(操作/类别/强度/模式)
  const buildAugBody = () => {
    const ops = Object.keys(augOps).filter((k) => augOps[k]);
    const classes = Object.keys(augClasses).map(Number).filter((c) => augClasses[c]);
    return {
      dataset: project,
      ops,
      source_split: 'train',
      max_deg: augDeg,
      classes,
      intensity: { max_deg: augDeg, noise_sigma: augNoise, brightness: augBright, blur_ksize: augBlurK },
      ...(augMode === 'balance' ? { target_per_class: augTarget } : { factor: augFactor }),
    };
  };

  const runAugment = () => {
    const body = buildAugBody();
    if (!body.ops.length) { setAugMsg('❌ 请至少选择一种增强操作'); setTimeout(() => setAugMsg(''), 3000); return; }
    setAugBusy(true); setAugMsg('增强生成中…');
    fetch(`${backendUrl}/api/dataset/augment`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    })
      .then((r) => r.json())
      .then((d: { detail?: string; generated?: number; sources?: number; mode?: string; per_class_added?: Record<string, number> }) => {
        if (d.detail) { setAugMsg(`❌ ${d.detail}`); return; }
        if (d.mode === 'balance') {
          const detail = Object.entries(d.per_class_added || {}).map(([c, n]) => `类${c}+${n}`).join(' ');
          setAugMsg(`✅ 均衡完成，生成 ${d.generated} 张 (${detail || '各类已达标'})，已写入 train`);
        } else {
          setAugMsg(`✅ 由 ${d.sources} 张原图生成 ${d.generated} 张增强样本，已写入 train`);
        }
        loadDataset(); loadProjects();
      })
      .catch(() => setAugMsg('❌ 增强失败'))
      .finally(() => { setAugBusy(false); setTimeout(() => setAugMsg(''), 6000); });
  };

  const previewAugment = () => {
    const body = buildAugBody();
    if (!body.ops.length) { setAugMsg('❌ 请至少选择一种增强操作'); setTimeout(() => setAugMsg(''), 3000); return; }
    setAugBusy(true); setAugMsg('生成预览中…'); setAugPreviews([]);
    fetch(`${backendUrl}/api/dataset/augment_preview`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...body, count: 4 }),
    })
      .then((r) => r.json())
      .then((d: { detail?: string; previews?: { src: string; boxes: number; image: string }[]; note?: string }) => {
        if (d.detail) { setAugMsg(`❌ ${d.detail}`); return; }
        setAugPreviews(d.previews || []);
        setAugMsg(d.note || `✅ 预览 ${d.previews?.length ?? 0} 张(未写入数据集)`);
      })
      .catch(() => setAugMsg('❌ 预览失败'))
      .finally(() => { setAugBusy(false); setTimeout(() => setAugMsg(''), 6000); });
  };

  const clearAugment = () => {
    if (!window.confirm('确定删除该数据集中所有离线增强样本(<dataset>_aug_*)？原始标注图不受影响。')) return;
    setAugBusy(true);
    fetch(`${backendUrl}/api/dataset/clear_augment`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset: project }),
    })
      .then((r) => r.json())
      .then((d: { removed?: number }) => { setAugMsg(`✅ 已清除 ${d.removed ?? 0} 张增强样本`); loadDataset(); loadProjects(); })
      .catch(() => setAugMsg('❌ 清除失败'))
      .finally(() => { setAugBusy(false); setTimeout(() => setAugMsg(''), 5000); });
  };

  const switchModel = (model: string) => {
    fetch(`${backendUrl}/api/model/active`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model }) })
      .then((r) => r.json())
      .then((d: { detail?: string; projects?: ProjectStat[] }) => {
        if (d.detail) { setSavedMsg(`❌ ${d.detail}`); setTimeout(() => setSavedMsg(''), 3000); return; }
        if (d.projects) setProjects(d.projects);
        loadModels(); // 刷新散装模型的 active 标记
        setSavedMsg('✅ 已切换实时生效模型'); setTimeout(() => setSavedMsg(''), 3000);
      })
      .catch(() => {});
  };

  const flash = (msg: string) => { setSavedMsg(msg); setTimeout(() => setSavedMsg(''), 3000); };

  const deleteProject = (p: ProjectStat) => {
    if (!window.confirm(`确定删除项目「${p.label}」?\n将一并删除其数据集目录和模型文件 ${p.model}，不可恢复。`)) return;
    fetch(`${backendUrl}/api/projects/${p.name}`, { method: 'DELETE' })
      .then((r) => r.json())
      .then((d: { detail?: string; projects?: ProjectStat[] }) => {
        if (d.detail) { flash(`❌ ${d.detail}`); return; }
        if (d.projects) setProjects(d.projects);
        if (project === p.name) setProject('reckless_mapped');
        flash('✅ 已删除项目');
      })
      .catch(() => flash('❌ 删除失败'));
  };

  const renameProject = (p: ProjectStat) => {
    const new_label = window.prompt('新的显示名称：', p.label);
    if (new_label == null) return;
    const new_name = p.builtin ? undefined : (window.prompt('新的英文标识名(留空则不改;改名会迁移数据集目录+模型)：', p.name) || undefined);
    fetch(`${backendUrl}/api/projects/${p.name}/rename`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_label, new_name: new_name === p.name ? undefined : new_name }),
    }).then((r) => r.json())
      .then((d: { detail?: string; projects?: ProjectStat[]; project?: { name: string } }) => {
        if (d.detail) { flash(`❌ ${d.detail}`); return; }
        if (d.projects) setProjects(d.projects);
        if (project === p.name && d.project) setProject(d.project.name);
        flash('✅ 已重命名');
      })
      .catch(() => flash('❌ 重命名失败'));
  };

  const exportDataset = (name: string) => { window.open(`${backendUrl}/api/dataset/export?dataset=${encodeURIComponent(name)}`, '_blank'); };

  // 把增量(unassigned)并入训练集(train)，标记为「已训练」
  const promoteIncremental = () => {
    if (!window.confirm('把当前数据集的「增量(unassigned)」图片(连标注)并入训练集 train？\n并入后将作为已训练数据参与下次训练。')) return;
    fetch(`${backendUrl}/api/dataset/promote_incremental`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ dataset: project }) })
      .then((r) => r.json())
      .then((d: { moved?: number }) => { flash(`✅ 已把 ${d.moved ?? 0} 张增量并入训练集`); loadDataset(); loadProjects(); })
      .catch(() => flash('❌ 并入失败'));
  };

  const curProject = projects.find((p) => p.name === project);
  const incrementalCount = ds?.splits?.unassigned?.images ?? 0;

  // 开训前数据划分体检：拦截会让 mAP 失真/训练事故的常见问题
  const splitImg = (s: string) => ds?.splits?.[s]?.images ?? 0;
  const splitLbl = (s: string) => ds?.splits?.[s]?.labels ?? 0;
  const healthIssues: string[] = [];
  if (ds?.exists) {
    if (splitImg('train') === 0) healthIssues.push('训练集为空(train=0)，无法训练');
    if (splitImg('valid') === 0) healthIssues.push('无验证集(valid=0)：mAP 将完全不可信');
    if (splitImg('test') === 0) healthIssues.push('无测试集(test=0)');
    (['train', 'valid', 'test'] as const).forEach((s) => { if (splitImg(s) > 0 && splitLbl(s) === 0) healthIssues.push(`${s} 有图但全无标注(会被当背景)`); });
    const emptyCls = (ds.classes || []).filter((c) => (c.count ?? 0) === 0).map((c) => c.name);
    if (emptyCls.length) healthIssues.push(`类无任何样本：${emptyCls.join('、')}`);
  }

  // 把其他数据集的已标注样本并入当前数据集(防遗忘混合微调：把实时/演示数据并进现役集再训)
  const mergeInto = () => {
    if (!mergeSrc || mergeSrc === project) return;
    const srcLabel = projects.find((p) => p.name === mergeSrc)?.label ?? mergeSrc;
    if (!window.confirm(`把「${srcLabel}」中所有「已标注」样本(连标注)复制并入当前数据集「${curProject?.label ?? project}」？\n源数据集不变；未标注图会跳过；同名已存在会跳过(可重复执行不重复并入)。`)) return;
    fetch(`${backendUrl}/api/dataset/merge_into`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ src: mergeSrc, dst: project }) })
      .then((r) => r.json())
      .then((d: { detail?: string; total?: number; skipped_unlabeled?: number; added?: Record<string, number> }) => {
        if (d.detail) { flash(`❌ ${d.detail}`); return; }
        const a = d.added || {};
        flash(`✅ 已并入 ${d.total ?? 0} 张(train ${a.train ?? 0}/valid ${a.valid ?? 0}/test ${a.test ?? 0})，跳过未标注 ${d.skipped_unlabeled ?? 0}`);
        loadDataset(); loadProjects();
      })
      .catch(() => flash('❌ 并入失败'));
  };

  const maxClass = Math.max(1, ...(ds?.classes?.map((c) => c.count) || [1]));

  // 训练基底权重可选项:通用 COCO 预训练 + 已训练出的项目模型(可微调/续训)
  const baseOptions: { value: string; label: string }[] = [
    { value: 'yolov8n.pt', label: 'yolov8n.pt（通用·最快，实时优先）' },
    { value: 'yolov8s.pt', label: 'yolov8s.pt（通用·默认，均衡）' },
    { value: 'yolov8m.pt', label: 'yolov8m.pt（通用·精度高，较慢）' },
    ...projects
      .filter((p) => p.model_exists)
      .map((p) => ({ value: `models/${p.model}`, label: `models/${p.model}（基于「${p.label}」微调，少量数据即可）` })),
  ];

  const saveParams = () =>
    fetch(`${backendUrl}/api/detector/params`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params) })
      .then((r) => r.json())
      .then((d) => { setParams(d); setSavedMsg('✅ 检测阈值已保存并实时生效'); setTimeout(() => setSavedMsg(''), 3000); });

  const saveTrain = () =>
    fetch(`${backendUrl}/api/training/config`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(train) })
      .then((r) => r.json())
      .then((d) => { setTrain(d); setSavedMsg('✅ 训练参数已保存(下次训练生效)'); setTimeout(() => setSavedMsg(''), 3000); });

  const onUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setUploadMsg('上传解析中…');
    const fd = new FormData();
    fd.append('file', f);
    fetch(`${backendUrl}/api/dataset/upload`, { method: 'POST', body: fd })
      .then((r) => r.json())
      .then((d: { detail?: string; dir?: string; images?: number; labels?: number }) =>
        setUploadMsg(d.detail ? `❌ ${d.detail}` : `✅ 已上传「${d.dir}」：图片 ${d.images}、标注 ${d.labels}`),
      )
      .catch(() => setUploadMsg('❌ 上传失败'));
  };

  const bq = ds?.bbox_quality;
  const bqTotal = bq ? bq.tiny + bq.normal + bq.huge : 0;

  return (
    <div className="h-screen overflow-y-auto bg-gradient-to-br from-slate-100 to-blue-50/40">
      <header className="bg-white/80 backdrop-blur border-b border-outline-variant/60 px-8 h-16 flex items-center justify-between sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="flex items-center gap-1 text-slate-500 hover:text-primary text-sm font-medium"><ArrowLeft size={18} /> 返回系统</button>
          <span className="w-px h-5 bg-outline-variant" />
          <h1 className="font-display text-xl font-bold text-slate-900 flex items-center gap-2"><Database size={20} className="text-primary" /> 模型与数据</h1>
        </div>
        {savedMsg && <span className="text-sm text-green-600 font-medium">{savedMsg}</span>}
      </header>

      <div className="max-w-[1080px] mx-auto p-8 space-y-6">
        <SectionTitle>数据集 / 模型项目</SectionTitle>
        {/* 项目/数据集选择 */}
        <Card icon={FolderPlus} title="数据集 / 模型项目" desc="不同任务用不同的数据集与模型。选择一个项目后，下方的浏览、标注、训练都作用于它；可新建空数据集自行上传标注">
          <div className="flex flex-wrap items-center gap-3">
            <select value={project} onChange={(e) => setProject(e.target.value)} className="px-3 py-2 rounded-lg border border-outline-variant text-sm focus:outline-none focus:border-primary min-w-[220px]">
              {projects.map((p) => (
                <option key={p.name} value={p.name}>{p.label}（{p.name}）</option>
              ))}
            </select>
            <button onClick={createProject} className="btn-secondary px-4 py-2 text-sm"><FolderPlus size={15} /> 新建数据集</button>
            {(() => {
              const cur = projects.find((p) => p.name === project);
              if (!cur) return null;
              const total = (Object.values(cur.images || {}) as number[]).reduce((a, b) => a + b, 0);
              return (
                <span className="text-xs text-slate-500 ml-auto">
                  图片 <b>{total}</b> 张 · 模型 <b>{cur.model}</b> {cur.model_exists ? `(${cur.model_size_mb}MB${cur.active ? ' · 实时生效中' : ''})` : '(未训练)'}
                </span>
              );
            })()}
          </div>
        </Card>

        <SectionTitle>模型部署 · 实时生效</SectionTitle>
        {/* 模型管理 / 实时生效模型 */}
        <Card icon={Cpu} title="模型管理 · 实时生效模型" desc="切换实时检测当前使用的模型；演示不同任务时一键切换(立即热生效，无需重启)">
          <div className="space-y-2">
            {projects.map((p) => (
              <div key={p.name} className={`flex items-center gap-3 p-3 rounded-xl border ${p.active ? 'border-primary bg-primary/5' : 'border-outline-variant/60'}`}>
                <Cpu size={18} className={p.active ? 'text-primary' : 'text-slate-400'} />
                <div className="min-w-0">
                  <div className="text-sm font-medium text-slate-800 truncate">{p.label} <span className="text-xs text-slate-400">{p.model}</span></div>
                  <div className="text-[11px] text-slate-400">{p.model_exists ? `${p.model_size_mb} MB` : '尚未训练生成'}</div>
                  <ModelMeta meta={p.meta} />
                </div>
                <div className="ml-auto shrink-0 flex items-center gap-1.5">
                  {p.active ? (
                    <span className="inline-flex items-center gap-1 text-sm text-primary font-medium"><CheckCircle2 size={16} /> 实时生效中</span>
                  ) : (
                    <button onClick={() => switchModel(p.model)} disabled={!p.model_exists} className="btn-secondary px-3 py-1.5 text-xs disabled:opacity-40 disabled:cursor-not-allowed">设为当前模型</button>
                  )}
                  <button onClick={() => exportDataset(p.name)} title="导出数据集 zip" className="p-1.5 rounded-lg text-slate-400 hover:text-primary hover:bg-slate-100"><Download size={16} /></button>
                  <button onClick={() => renameProject(p)} title="重命名" className="p-1.5 rounded-lg text-slate-400 hover:text-primary hover:bg-slate-100"><Pencil size={15} /></button>
                  {!p.builtin && (
                    <button onClick={() => deleteProject(p)} title="删除项目" className="p-1.5 rounded-lg text-slate-400 hover:text-red-500 hover:bg-red-50"><Trash2 size={15} /></button>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* 手动放入 models/ 目录的散装 .pt（含外部训练的自训模型，无需注册项目即可设为实时生效） */}
          {modelFiles.filter((m) => !m.registered).length > 0 && (
            <div className="mt-4 pt-4 border-t border-outline-variant/50">
              <p className="text-xs text-slate-400 mb-2">其他模型文件（放入 behavior_algo/models/ 目录的 .pt，可直接设为实时生效）</p>
              <div className="space-y-2">
                {modelFiles.filter((m) => !m.registered).map((m) => (
                  <div key={m.model} className={`flex items-center gap-3 p-3 rounded-xl border ${m.active ? 'border-primary bg-primary/5' : 'border-outline-variant/60'}`}>
                    <Cpu size={18} className={m.active ? 'text-primary' : 'text-slate-400'} />
                    <div className="min-w-0">
                      <div className="text-sm font-medium text-slate-800 truncate">{m.model}</div>
                      <div className="text-[11px] text-slate-400">{m.size_mb} MB · 自训/外部模型</div>
                      <ModelMeta meta={m.meta} />
                    </div>
                    <div className="ml-auto shrink-0">
                      {m.active ? (
                        <span className="inline-flex items-center gap-1 text-sm text-primary font-medium"><CheckCircle2 size={16} /> 实时生效中</span>
                      ) : (
                        <button onClick={() => switchModel(m.model)} className="btn-secondary px-3 py-1.5 text-xs">设为当前模型</button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>

        <SectionTitle>数据集与标注</SectionTitle>
        {/* 数据集概览 */}
        <Card icon={Database} title="训练数据集概览" desc="当前所选项目的数据集类别分布与标注质量，便于筛查类别不均衡与异常标注">
          {ds?.exists && (
            <button onClick={() => setBrowserOpen(true)} className="btn-primary mb-4 px-5 py-2 text-sm">
              <Images size={16} /> 逐张浏览 / 编辑标注
            </button>
          )}
          {!ds?.exists ? (
            <p className="text-sm text-slate-400">未找到数据集</p>
          ) : (
            <>
              <div className="flex gap-4 text-sm mb-4">
                {(Object.entries(ds.splits || {}) as [string, { images: number; labels: number }][]).filter(([k]) => k !== 'unassigned').map(([k, v]) => (
                  <span key={k} className="px-3 py-1.5 rounded-lg bg-slate-50 border border-outline-variant/50">
                    {k}: <b>{v.images}</b> 图
                  </span>
                ))}
                <span className="px-3 py-1.5 rounded-lg bg-slate-50 border border-outline-variant/50">共 <b>{ds.num_classes}</b> 类 · <b>{ds.total_instances}</b> 标注</span>
              </div>
              {/* 已训练(train/valid/test) vs 增量(unassigned) 区分；增量可一键并入 */}
              {incrementalCount > 0 && (
                <div className="mb-4 flex flex-wrap items-center gap-3 p-3 rounded-lg bg-amber-50 border border-amber-200">
                  <span className="text-sm text-amber-800">
                    增量数据(未并入训练)：<b>{incrementalCount}</b> 张 —— 录制采样默认进这里；train/valid/test 为已训练数据
                  </span>
                  <button onClick={promoteIncremental} className="btn-secondary px-3 py-1.5 text-xs ml-auto">并入训练集</button>
                </div>
              )}
              {/* 并入其他数据集的已标注样本(防遗忘混合微调) */}
              {projects.length > 1 && (
                <div className="mb-4 flex flex-wrap items-center gap-2 p-3 rounded-lg bg-slate-50 border border-outline-variant/50">
                  <span className="text-sm text-slate-600">并入其他数据集的<b>已标注</b>样本到当前：</span>
                  <select value={mergeSrc} onChange={(e) => setMergeSrc(e.target.value)} className="px-2 py-1 rounded-lg border border-outline-variant text-sm focus:outline-none focus:border-primary">
                    <option value="">选择来源数据集…</option>
                    {projects.filter((p) => p.name !== project).map((p) => (
                      <option key={p.name} value={p.name}>{p.label}（{p.name}）</option>
                    ))}
                  </select>
                  <button onClick={mergeInto} disabled={!mergeSrc} className="btn-secondary px-3 py-1.5 text-xs disabled:opacity-40 disabled:cursor-not-allowed">并入</button>
                  <span className="text-[11px] text-slate-400 w-full">copy 不动源数据集 · 跳过未标注/同名 · 可重复执行</span>
                </div>
              )}
              <div className="space-y-2">
                {ds.classes?.map((c) => (
                  <div key={c.name} className="flex items-center gap-3">
                    <span className="w-28 text-sm text-slate-600 shrink-0 truncate">{c.name}</span>
                    <div className="flex-1 h-4 rounded-full bg-slate-100 overflow-hidden">
                      <div className="h-full rounded-full bg-gradient-to-r from-primary to-cyan-400" style={{ width: `${(c.count / maxClass) * 100}%` }} />
                    </div>
                    <span className="w-12 text-right text-sm font-bold text-slate-700">{c.count}</span>
                  </div>
                ))}
              </div>
              {bq && (
                <div className="mt-4 pt-4 border-t border-outline-variant/50">
                  <div className="text-xs font-bold text-slate-500 mb-2 flex items-center gap-1"><AlertTriangle size={13} /> 标注框尺寸质量</div>
                  <div className="grid grid-cols-3 gap-3 text-center text-sm">
                    <div className="rounded-lg bg-amber-50 border border-amber-100 py-2">
                      <div className="font-bold text-amber-700">{bq.tiny}</div>
                      <div className="text-[11px] text-amber-600">极小框(&lt;0.3%)难学</div>
                    </div>
                    <div className="rounded-lg bg-green-50 border border-green-100 py-2">
                      <div className="font-bold text-green-700">{bq.normal}</div>
                      <div className="text-[11px] text-green-600">正常({bqTotal ? Math.round((bq.normal / bqTotal) * 100) : 0}%)</div>
                    </div>
                    <div className="rounded-lg bg-red-50 border border-red-100 py-2">
                      <div className="font-bold text-red-700">{bq.huge}</div>
                      <div className="text-[11px] text-red-600">超大框(&gt;50%)疑似错标</div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </Card>

        {/* 离线数据增强 */}
        <Card icon={Sparkles} title="数据增强（离线扩充训练集）" desc="对 train 中已标注的图生成旋转/翻转/噪声/亮度/模糊变体并自动变换标注框，写回 train。适合原始样本少时扩充数据；只增强 train，valid/test 保持纯净">
          {!ds?.exists ? (
            <p className="text-sm text-slate-400">未找到数据集</p>
          ) : (
            <>
              <div className="flex flex-wrap gap-2 mb-4">
                {([
                  ['rotate', '旋转'],
                  ['flip', '水平翻转'],
                  ['noise', '高斯噪声'],
                  ['brightness', '亮度/对比度'],
                  ['blur', '运动模糊'],
                ] as [string, string][]).map(([k, label]) => (
                  <button
                    key={k}
                    onClick={() => setAugOps((o) => ({ ...o, [k]: !o[k] }))}
                    className={`px-3 py-1.5 rounded-lg text-sm border transition-colors ${augOps[k] ? 'bg-primary/10 border-primary text-primary font-medium' : 'bg-slate-50 border-outline-variant/50 text-slate-500'}`}
                  >
                    {augOps[k] ? '✓ ' : ''}{label}
                  </button>
                ))}
              </div>
              {/* 模式切换：固定份数 / 均衡到目标 */}
              <div className="flex gap-2 mb-4">
                {([['fixed', '固定份数'], ['balance', '均衡到目标']] as [typeof augMode, string][]).map(([m, label]) => (
                  <button key={m} onClick={() => setAugMode(m)}
                    className={`px-3 py-1.5 rounded-lg text-sm border transition-colors ${augMode === m ? 'bg-primary text-white border-primary font-medium' : 'bg-slate-50 border-outline-variant/50 text-slate-500'}`}>
                    {label}
                  </button>
                ))}
              </div>

              {/* 目标类别选择(空=全部类) */}
              <div className="mb-4">
                <div className="text-xs font-medium text-slate-500 mb-1.5">目标类别（不选=全部类；均衡模式只补选中类）</div>
                <div className="flex flex-wrap gap-2">
                  {ds.classes?.map((c) => {
                    const idx = c.cls ?? -1;
                    const on = !!augClasses[idx];
                    return (
                      <button key={c.name} onClick={() => setAugClasses((s) => ({ ...s, [idx]: !s[idx] }))}
                        className={`px-2.5 py-1 rounded-lg text-xs border transition-colors ${on ? 'bg-primary/10 border-primary text-primary font-medium' : 'bg-slate-50 border-outline-variant/50 text-slate-500'}`}>
                        {on ? '✓ ' : ''}{c.name} <span className="opacity-60">({c.count})</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* 数量参数 */}
              <div className="flex flex-wrap gap-6 items-end mb-4">
                {augMode === 'fixed' ? (
                  <div>
                    <label className="text-xs font-medium text-slate-500 block mb-1">每张原图生成份数</label>
                    <input type="number" min={1} max={10} value={augFactor}
                      onChange={(e) => setAugFactor(Math.max(1, Math.min(10, Number(e.target.value) || 1)))}
                      className="w-24 px-3 py-1.5 rounded-lg border border-outline-variant/50 text-sm" />
                  </div>
                ) : (
                  <div>
                    <label className="text-xs font-medium text-slate-500 block mb-1">每类目标框数</label>
                    <input type="number" min={1} max={5000} value={augTarget}
                      onChange={(e) => setAugTarget(Math.max(1, Math.min(5000, Number(e.target.value) || 1)))}
                      className="w-28 px-3 py-1.5 rounded-lg border border-outline-variant/50 text-sm" />
                  </div>
                )}
              </div>

              {/* 逐操作强度 */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-3 mb-4 pt-3 border-t border-outline-variant/50">
                <div className={augOps.rotate ? '' : 'opacity-40'}>
                  <label className="text-xs font-medium text-slate-500 block mb-1">旋转 ±{augDeg}°</label>
                  <input type="range" min={0} max={45} step={1} value={augDeg} disabled={!augOps.rotate}
                    onChange={(e) => setAugDeg(Number(e.target.value))} className="w-full accent-primary" />
                </div>
                <div className={augOps.noise ? '' : 'opacity-40'}>
                  <label className="text-xs font-medium text-slate-500 block mb-1">噪声 σ {augNoise}</label>
                  <input type="range" min={0} max={60} step={1} value={augNoise} disabled={!augOps.noise}
                    onChange={(e) => setAugNoise(Number(e.target.value))} className="w-full accent-primary" />
                </div>
                <div className={augOps.brightness ? '' : 'opacity-40'}>
                  <label className="text-xs font-medium text-slate-500 block mb-1">亮度幅度 ±{augBright}</label>
                  <input type="range" min={0} max={80} step={1} value={augBright} disabled={!augOps.brightness}
                    onChange={(e) => setAugBright(Number(e.target.value))} className="w-full accent-primary" />
                </div>
                <div className={augOps.blur ? '' : 'opacity-40'}>
                  <label className="text-xs font-medium text-slate-500 block mb-1">模糊核 ≤{augBlurK}</label>
                  <input type="range" min={3} max={21} step={2} value={augBlurK} disabled={!augOps.blur}
                    onChange={(e) => setAugBlurK(Number(e.target.value))} className="w-full accent-primary" />
                </div>
              </div>

              {/* 操作按钮 */}
              <div className="flex flex-wrap gap-3 items-center">
                <button onClick={previewAugment} disabled={augBusy} className="btn-secondary px-4 py-2 text-sm disabled:opacity-50">
                  <Images size={15} /> 预览效果
                </button>
                <button onClick={runAugment} disabled={augBusy} className="btn-primary px-5 py-2 text-sm disabled:opacity-50">
                  <Sparkles size={15} /> {augMode === 'balance' ? '均衡生成' : '生成增强样本'}
                </button>
                <button onClick={clearAugment} disabled={augBusy} className="btn-secondary px-4 py-2 text-sm disabled:opacity-50">
                  <Trash2 size={15} /> 清除增强样本
                </button>
                {augMsg && <span className="text-sm text-slate-600">{augMsg}</span>}
              </div>

              {/* 预览缩略图 */}
              {augPreviews.length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
                  {augPreviews.map((p, i) => (
                    <div key={i} className="rounded-lg overflow-hidden border border-outline-variant/50">
                      <img src={p.image} alt={p.src} className="w-full block" />
                      <div className="text-[11px] text-slate-500 px-2 py-1 truncate">{p.src} · {p.boxes} 框</div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </Card>

        <SectionTitle>模型训练 · 重训配置</SectionTitle>
        {/* 训练参数 */}
        <Card icon={Wrench} title="训练参数" desc="模型重训配置，保存后下次运行 train_unified.py 时生效">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {Object.keys(TRAIN_LABELS).map((k) =>
              k === 'base' ? (
                <div key={k} className="col-span-2 md:col-span-3">
                  <label className="text-xs font-medium text-slate-500 block mb-1">{TRAIN_LABELS[k]}（决定从哪个权重开始训练）</label>
                  <select
                    value={String(train[k] ?? 'yolov8s.pt')}
                    onChange={(e) => setTrain((t) => ({ ...t, base: e.target.value }))}
                    className="w-full px-3 py-2 rounded-lg border border-outline-variant text-sm focus:outline-none focus:border-primary bg-white"
                  >
                    {/* 当前值若不在列表(如自定义)也保留可见 */}
                    {!baseOptions.some((o) => o.value === train[k]) && train[k] && (
                      <option value={String(train[k])}>{String(train[k])}（当前）</option>
                    )}
                    {baseOptions.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                  </select>
                </div>
              ) : (
                <div key={k}>
                  <label className="text-xs font-medium text-slate-500 block mb-1">{TRAIN_LABELS[k]}</label>
                  <input
                    value={(train[k] as string | number | undefined) ?? ''}
                    onChange={(e) => setTrain((t) => ({ ...t, [k]: Number(e.target.value) || 0 }))}
                    className="w-full px-3 py-2 rounded-lg border border-outline-variant text-sm focus:outline-none focus:border-primary"
                  />
                </div>
              ),
            )}
          </div>

          {/* 增量微调旋钮 + 演示模式 */}
          <div className="mt-5 pt-4 border-t border-outline-variant/50 grid grid-cols-2 gap-x-6 gap-y-1">
            <div>
              <label className="text-xs font-medium text-slate-500 block mb-1">冻结层数 freeze</label>
              <input type="number" min={0} max={24} value={Number(train.freeze ?? 0)}
                onChange={(e) => setTrain((t) => ({ ...t, freeze: Math.max(0, Number(e.target.value) || 0) }))}
                className="w-full px-3 py-2 rounded-lg border border-outline-variant text-sm focus:outline-none focus:border-primary" />
              <p className="text-[11px] text-slate-400 mt-1">0=不冻结；冻 backbone 填 <b>10</b>(增量少、防遗忘/过拟合)</p>
            </div>
            <div>
              <label className="text-xs font-medium text-slate-500 block mb-1">初始学习率 lr0</label>
              <input type="number" min={0} step={0.0001} value={Number(train.lr0 ?? 0.001)}
                onChange={(e) => setTrain((t) => ({ ...t, lr0: Math.max(0, Number(e.target.value) || 0) }))}
                className="w-full px-3 py-2 rounded-lg border border-outline-variant text-sm focus:outline-none focus:border-primary" />
              <p className="text-[11px] text-slate-400 mt-1">从已训权重微调建议更低：<b>1e-4 ~ 5e-4</b></p>
            </div>
          </div>

          <div className="mt-4 space-y-1 max-w-xl">
            <Toggle on={!!train.include_incremental} onClick={() => setTrain((t) => ({ ...t, include_incremental: !t.include_incremental }))}
              label="纳入增量数据 (unassigned)" hint="把尚未并入训练的增量也加入本次训练" />
            <Toggle on={!!train.full_data} onClick={() => setTrain((t) => ({ ...t, full_data: !t.full_data }))}
              label="演示模式 (过拟合)" hint="valid/test 并入训练、并在 train 上验证 —— 识别看着极准，仅现场演示用，mAP 不代表泛化" />
            {!!train.full_data && curProject?.builtin && (
              <p className="text-[11px] text-amber-600 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 mt-1 flex items-start gap-1.5">
                <AlertTriangle size={13} className="shrink-0 mt-0.5" />
                <span>演示模式会把现役通用模型 <b>{curProject.model}</b> 训成过拟合。建议<b>新建一个单独的演示数据集</b>来训，别覆盖通用模型 unified.pt。</span>
              </p>
            )}
          </div>
          <button onClick={saveTrain} className="btn-primary mt-5 px-6">保存训练配置</button>
        </Card>

        {/* 模型训练 */}
        <Card icon={Rocket} title="模型训练" desc={`后台训练所选项目「${projects.find((p) => p.name === project)?.label ?? project}」→ 产出 ${projects.find((p) => p.name === project)?.model ?? 'unified.pt'}；完成后到上方“模型管理”一键设为实时生效`}>
          <div className="text-xs text-slate-500 mb-3">
            将训练：数据集 <b>{curProject?.label ?? project}</b> · 基础权重 <b>{String(train.base ?? 'yolov8s.pt').replace('models/', '')}</b>
            {!!train.full_data && <span className="ml-2 px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 text-[11px] font-medium">演示·过拟合</span>}
            {!!train.include_incremental && <span className="ml-1.5 px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 text-[11px]">含增量</span>}
          </div>
          {/* 开训前划分体检：用已算好的 per-split 数据拦掉最常见训练事故 */}
          {healthIssues.length > 0 && (
            <div className="mb-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
              <div className="text-[12px] font-bold text-amber-700 flex items-center gap-1.5 mb-1"><AlertTriangle size={13} /> 数据划分体检（建议先修再训）</div>
              <ul className="text-[11px] text-amber-700 list-disc pl-4 space-y-0.5">
                {healthIssues.map((h) => <li key={h}>{h}</li>)}
              </ul>
              {!!train.full_data && <div className="text-[11px] text-amber-600 mt-1">注：演示模式用 train 当验证集，以上问题会被"虚高 mAP"掩盖。</div>}
            </div>
          )}
          {trainStatus?.running ? (
            <div>
              <div className="flex items-center justify-between text-sm mb-1">
                <span className="text-slate-700 font-medium flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-primary animate-pulse" /> 训练中 · Epoch {trainStatus.epoch}/{trainStatus.total_epochs}
                </span>
                <span className="text-outline">最佳 mAP50 {trainStatus.best_map} · 已用 {Math.floor(trainStatus.elapsed / 60)}分{trainStatus.elapsed % 60}秒</span>
              </div>
              <div className="h-2 rounded-full bg-slate-200 overflow-hidden">
                <div className="h-full rounded-full bg-gradient-to-r from-primary to-cyan-400 transition-all" style={{ width: `${trainStatus.total_epochs ? (trainStatus.epoch / trainStatus.total_epochs) * 100 : 0}%` }} />
              </div>
              <pre className="mt-3 text-[10px] leading-relaxed text-slate-500 bg-slate-900/95 text-slate-300 rounded-lg p-3 max-h-32 overflow-auto whitespace-pre-wrap">{trainStatus.log_tail || '启动中…'}</pre>
            </div>
          ) : (
            <div>
              {trainStatus?.finished && trainStatus.success && (
                <p className="text-sm text-green-600 mb-3">✅ 训练完成：基于「{trainStatus.project}」+ {String(trainStatus.base ?? '').replace('models/', '')}，最佳 mAP50 {trainStatus.best_map}，已产出 {trainStatus.out_model}{trainStatus.is_active ? '(实时生效中)' : '；可在上方“模型管理”设为生效'}</p>
              )}
              {trainStatus?.finished && !trainStatus.success && (
                <p className="text-sm text-orange-500 mb-3">⚠️ 上次训练未成功完成(中止或出错)</p>
              )}
              <button onClick={startTrain} className="btn-primary px-6"><Rocket size={16} /> 开始训练</button>
              <p className="text-xs text-slate-400 mt-2">训练在后台进行(约数十分钟~数小时,视轮数),期间可继续使用系统;本页实时显示进度。</p>
            </div>
          )}

          {/* 各类别 mAP：逐 epoch 刷新，训练时即可看到每个标签的精度(不只总 mAP50) */}
          {trainStatus?.per_class?.per_class && Object.keys(trainStatus.per_class.per_class).length > 0 && (
            <div className="mt-4 pt-4 border-t border-outline-variant/50">
              <div className="text-xs font-bold text-slate-500 mb-2">
                各类别 mAP@50（epoch {trainStatus.per_class.epoch ?? '--'}）
                <span className="font-normal text-slate-400 ml-2">
                  总 mAP50 {((trainStatus.per_class.map50 ?? 0) * 100).toFixed(1)}% · mAP50-95 {((trainStatus.per_class.map ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="space-y-1.5">
                {Object.entries(trainStatus.per_class.per_class as Record<string, ClassMetric>).map(([name, v]) => (
                  <div key={name} className="flex items-center gap-3">
                    <span className="w-32 text-sm text-slate-600 shrink-0 truncate">{name}</span>
                    <div className="flex-1 h-3.5 rounded-full bg-slate-100 overflow-hidden">
                      <div className="h-full rounded-full bg-gradient-to-r from-primary to-cyan-400" style={{ width: `${Math.min(100, Math.round((v.map50 ?? 0) * 100))}%` }} />
                    </div>
                    <span className="w-12 text-right text-sm font-bold text-slate-700 tabular-nums">{((v.map50 ?? 0) * 100).toFixed(1)}</span>
                  </div>
                ))}
              </div>
              <p className="text-[11px] text-slate-400 mt-2">逐 epoch 刷新；某类偏低=该行为样本不足/标注差，可针对性补数据再训。</p>
            </div>
          )}
        </Card>

        {/* 上传数据集 */}
        <Card icon={Upload} title="上传自定义数据集" desc="上传 YOLO 格式数据集压缩包(images + labels + data.yaml)，用于扩充/替换训练数据">
          <label className="inline-flex items-center gap-2 px-5 py-3 rounded-xl border-2 border-dashed border-outline-variant hover:border-primary cursor-pointer text-sm text-slate-600 transition-colors">
            <Upload size={18} /> 选择 .zip 数据集
            <input type="file" accept=".zip" className="hidden" onChange={onUpload} />
          </label>
          {uploadMsg && <p className="text-sm mt-3 text-slate-600">{uploadMsg}</p>}
        </Card>

        <SectionTitle>运行时阈值 · 全局（作用于当前生效模型）</SectionTitle>
        {/* 检测阈值：运行时·全局，与数据集/项目无关 */}
        <Card icon={SlidersHorizontal} title="检测阈值（实时可调 · 全局）" desc="作用于「当前生效模型」、与数据集/项目无关；调整后立即热生效。阈值越高越严格、误报越少但可能漏检">
          <div className="grid grid-cols-2 gap-x-8 gap-y-4">
            {Object.keys(PARAM_LABELS).map((k) => (
              <div key={k}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-600">{PARAM_LABELS[k]}</span>
                  <span className="font-bold text-primary tabular-nums">{(params[k] ?? 0).toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  value={params[k] ?? 0.4}
                  onChange={(e) => setParams((p) => ({ ...p, [k]: Number(e.target.value) }))}
                  className="w-full accent-primary"
                />
              </div>
            ))}
          </div>
          <button onClick={saveParams} className="btn-primary mt-5 px-6">保存并生效</button>
        </Card>
      </div>

      <DatasetBrowser open={browserOpen} onClose={() => { setBrowserOpen(false); loadDataset(); }} backendUrl={backendUrl} dataset={project} />
    </div>
  );
}
