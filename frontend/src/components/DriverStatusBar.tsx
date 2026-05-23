import { motion } from 'motion/react';
import { Eye, Flame, Hand, Shield, Smartphone, UserCheck, type LucideIcon } from 'lucide-react';

export interface DriverStatus {
  present: boolean; // 驾驶员在位
  seatbelt: boolean; // 已系安全带
  hands: boolean; // 双手在盘
  gaze: boolean; // 视线正常
  noPhone: boolean; // 未使用手机
  noSmoke: boolean; // 未吸烟
}

interface Props {
  status: DriverStatus | null;
}

const ITEMS: { key: keyof DriverStatus; icon: LucideIcon; ok: string; bad: string }[] = [
  { key: 'present', icon: UserCheck, ok: '驾驶员在位', bad: '驾驶位无人' },
  { key: 'seatbelt', icon: Shield, ok: '已系安全带', bad: '未系安全带' },
  { key: 'hands', icon: Hand, ok: '双手在盘', bad: '双手离盘' },
  { key: 'gaze', icon: Eye, ok: '视线正常', bad: '视线偏离' },
  { key: 'noPhone', icon: Smartphone, ok: '无手机', bad: '使用手机' },
  { key: 'noSmoke', icon: Flame, ok: '无吸烟', bad: '检测到吸烟' },
];

/** 驾驶员状态指示灯组：像车载警示集群，正常绿、告警红并脉冲。*/
export default function DriverStatusBar({ status }: Props) {
  return (
    <div className="absolute bottom-9 left-1/2 -translate-x-1/2 z-20 flex gap-2 pointer-events-none">
      {ITEMS.map(({ key, icon: Icon, ok, bad }) => {
        const good = status ? status[key] : true;
        const idle = !status;
        return (
          <motion.div
            key={key}
            animate={!good && !idle ? { scale: [1, 1.06, 1] } : { scale: 1 }}
            transition={{ duration: 0.9, repeat: Infinity }}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border backdrop-blur-md text-[11px] font-bold shadow-lg ${
              idle
                ? 'border-white/15 bg-black/40 text-slate-300'
                : good
                  ? 'border-emerald-400/40 bg-emerald-500/15 text-emerald-100'
                  : 'border-red-400/60 bg-red-500/25 text-red-50'
            }`}
          >
            <Icon size={14} />
            <span className="whitespace-nowrap">{idle ? ok : good ? ok : bad}</span>
          </motion.div>
        );
      })}
    </div>
  );
}
