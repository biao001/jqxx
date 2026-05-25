import { motion } from 'motion/react';
import { Activity, ArrowRight, BarChart3, Bot, Camera, Eye, IdCard, ShieldCheck } from 'lucide-react';

interface Props {
  onEnter: () => void;
  loggedIn: boolean;
  user: string | null;
}

const FEATURES = [
  { icon: Activity, title: '驾驶行为识别', desc: '手机/吸烟/饮食/安全带/双手离盘等多类危险行为实时检测' },
  { icon: Eye, title: '疲劳监测', desc: 'PERCLOS 闭眼率、头部姿态、视线区域，精准识别疲劳与分心' },
  { icon: ShieldCheck, title: '分级告警', desc: '语音播报 + 全屏脉冲分级预警，关键帧自动抓拍取证' },
  { icon: Bot, title: 'AI 智能点评', desc: '大模型实时点评驾驶状态、生成改进建议、支持对话问答' },
  { icon: IdCard, title: '车主识别', desc: '人脸登记与实时身份识别，防顶替、按驾驶员归档' },
  { icon: BarChart3, title: '数据看板', desc: '综合/行为/疲劳多维评分，行程趋势与违规统计分析' },
];

/** 平台首页/落地介绍页。*/
export default function LandingPage({ onEnter, loggedIn, user }: Props) {
  return (
    <div className="h-screen overflow-y-auto bg-gradient-to-br from-slate-900 via-slate-800 to-blue-950 text-white">
      <div className="max-w-[1100px] mx-auto px-8 py-16">
        {/* Hero */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }} className="text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/10 border border-white/15 text-sm text-blue-200 mb-6">
            <Camera size={15} /> 基于深度学习的驾驶员监测系统 · DMS
          </div>
          <h1 className="font-display text-5xl md:text-6xl font-bold tracking-tight leading-tight">
            驾驶状态推演<br />与<span className="bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">行为测算平台</span>
          </h1>
          <p className="mt-6 text-lg text-slate-300 max-w-2xl mx-auto leading-relaxed">
            实时识别危险驾驶行为与疲劳状态，多维评分 + AI 点评 + 分级预警，
            为行车安全提供全方位智能监测与决策支持。
          </p>
          <div className="mt-9 flex items-center justify-center gap-4">
            <button
              onClick={onEnter}
              className="group px-8 py-4 rounded-xl bg-gradient-to-r from-blue-500 to-cyan-400 text-white font-bold text-lg shadow-xl shadow-blue-500/30 hover:brightness-110 active:scale-95 transition-all flex items-center gap-2"
            >
              {loggedIn ? '进入系统' : '开始体验'}
              <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
            </button>
            {loggedIn && <span className="text-sm text-slate-400">已登录：{user}</span>}
          </div>
        </motion.div>

        {/* 数据条 */}
        <div className="mt-14 grid grid-cols-3 gap-6 max-w-2xl mx-auto text-center">
          {[['8+', '检测行为类型'], ['5', '驾驶状态维度'], ['实时', '逐帧 AI 分析']].map(([n, l]) => (
            <div key={l}>
              <div className="font-display text-3xl font-bold text-blue-300">{n}</div>
              <div className="text-sm text-slate-400 mt-1">{l}</div>
            </div>
          ))}
        </div>

        {/* 功能卡 */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-5">
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + i * 0.06 }}
              className="rounded-2xl bg-white/5 border border-white/10 p-6 hover:bg-white/10 hover:border-blue-400/30 transition-all"
            >
              <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-blue-500/30 to-cyan-400/20 border border-blue-400/20 flex items-center justify-center mb-4">
                <f.icon size={22} className="text-blue-300" />
              </div>
              <h3 className="font-display text-lg font-bold mb-2">{f.title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{f.desc}</p>
            </motion.div>
          ))}
        </div>

        <div className="mt-16 text-center text-xs text-slate-500">驾驶状态推演与行为测算平台 · 深度学习驱动的车载安全监测</div>
      </div>
    </div>
  );
}
