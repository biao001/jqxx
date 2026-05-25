import { type ChangeEvent, type ReactNode, useCallback, useEffect, useState } from 'react';
import { AlertTriangle, ArrowLeft, CheckCircle2, Cpu, Database, Download, FolderPlus, Images, type LucideIcon, Pencil, Rocket, SlidersHorizontal, Trash2, Upload, Wrench } from 'lucide-react';
import DatasetBrowser from './DatasetBrowser';

interface TrainStatus {
  running: boolean;
  started: boolean;
  project?: string;
  out_model?: string;
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
}

interface Props {
  backendUrl: string;
  onBack: () => void;
}

interface DatasetInfo {
  exists: boolean;
  names?: string[];
  num_classes?: number;
  classes?: { name: string; count: number }[];
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

export default function ModelDataPage({ backendUrl, onBack }: Props) {
  const [ds, setDs] = useState<DatasetInfo | null>(null);
  const [params, setParams] = useState<Record<string, number>>({});
  const [train, setTrain] = useState<Record<string, string | number>>({});
  const [savedMsg, setSavedMsg] = useState('');
  const [uploadMsg, setUploadMsg] = useState('');
  const [browserOpen, setBrowserOpen] = useState(false);
  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null);
  const [projects, setProjects] = useState<ProjectStat[]>([]);
  const [project, setProject] = useState('reckless_mapped');

  const loadProjects = useCallback(() => {
    fetch(`${backendUrl}/api/projects`).then((r) => r.json()).then((d) => setProjects(d.projects || [])).catch(() => {});
  }, [backendUrl]);

  const loadDataset = useCallback(() => {
    fetch(`${backendUrl}/api/dataset/info?dataset=${project}`).then((r) => r.json()).then((d) => setDs(d.active)).catch(() => {});
  }, [backendUrl, project]);

  useEffect(() => {
    loadProjects();
    fetch(`${backendUrl}/api/detector/params`).then((r) => r.json()).then(setParams).catch(() => {});
    fetch(`${backendUrl}/api/training/config`).then((r) => r.json()).then(setTrain).catch(() => {});
  }, [backendUrl, loadProjects]);

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

  const switchModel = (model: string) => {
    fetch(`${backendUrl}/api/model/active`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model }) })
      .then((r) => r.json())
      .then((d: { detail?: string; projects?: ProjectStat[] }) => {
        if (d.detail) { setSavedMsg(`❌ ${d.detail}`); setTimeout(() => setSavedMsg(''), 3000); return; }
        if (d.projects) setProjects(d.projects);
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

        {/* 模型管理 / 实时生效模型 */}
        <Card icon={Cpu} title="模型管理 · 实时生效模型" desc="切换实时检测当前使用的模型；演示不同任务时一键切换(立即热生效，无需重启)">
          <div className="space-y-2">
            {projects.map((p) => (
              <div key={p.name} className={`flex items-center gap-3 p-3 rounded-xl border ${p.active ? 'border-primary bg-primary/5' : 'border-outline-variant/60'}`}>
                <Cpu size={18} className={p.active ? 'text-primary' : 'text-slate-400'} />
                <div className="min-w-0">
                  <div className="text-sm font-medium text-slate-800 truncate">{p.label} <span className="text-xs text-slate-400">{p.model}</span></div>
                  <div className="text-[11px] text-slate-400">{p.model_exists ? `${p.model_size_mb} MB` : '尚未训练生成'}</div>
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
        </Card>

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
                {(Object.entries(ds.splits || {}) as [string, { images: number; labels: number }][]).map(([k, v]) => (
                  <span key={k} className="px-3 py-1.5 rounded-lg bg-slate-50 border border-outline-variant/50">
                    {k}: <b>{v.images}</b> 图
                  </span>
                ))}
                <span className="px-3 py-1.5 rounded-lg bg-slate-50 border border-outline-variant/50">共 <b>{ds.num_classes}</b> 类 · <b>{ds.total_instances}</b> 标注</span>
              </div>
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

        {/* 检测参数(实时可调) */}
        <Card icon={SlidersHorizontal} title="检测参数（实时可调）" desc="调整各行为的置信度阈值，立即生效；阈值越高越严格、误报越少但可能漏检">
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
                    value={train[k] ?? ''}
                    onChange={(e) => setTrain((t) => ({ ...t, [k]: Number(e.target.value) || 0 }))}
                    className="w-full px-3 py-2 rounded-lg border border-outline-variant text-sm focus:outline-none focus:border-primary"
                  />
                </div>
              ),
            )}
          </div>
          <button onClick={saveTrain} className="btn-primary mt-5 px-6">保存训练配置</button>
        </Card>

        {/* 模型训练 */}
        <Card icon={Rocket} title="模型训练" desc={`后台训练所选项目「${projects.find((p) => p.name === project)?.label ?? project}」→ 产出 ${projects.find((p) => p.name === project)?.model ?? 'unified.pt'}；完成后到上方“模型管理”一键设为实时生效`}>
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
                <p className="text-sm text-green-600 mb-3">✅ 训练完成，最佳 mAP50 {trainStatus.best_map}，已产出 {trainStatus.out_model}{trainStatus.is_active ? '(实时生效中)' : '；可在上方“模型管理”设为生效'}</p>
              )}
              {trainStatus?.finished && !trainStatus.success && (
                <p className="text-sm text-orange-500 mb-3">⚠️ 上次训练未成功完成(中止或出错)</p>
              )}
              <button onClick={startTrain} className="btn-primary px-6"><Rocket size={16} /> 开始训练</button>
              <p className="text-xs text-slate-400 mt-2">训练在后台进行(约数十分钟~数小时,视轮数),期间可继续使用系统;本页实时显示进度。</p>
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
      </div>

      <DatasetBrowser open={browserOpen} onClose={() => { setBrowserOpen(false); loadDataset(); }} backendUrl={backendUrl} dataset={project} />
    </div>
  );
}
