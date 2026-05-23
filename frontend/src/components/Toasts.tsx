import { AnimatePresence, motion } from 'motion/react';
import { AlertTriangle, Bell, Info } from 'lucide-react';
import { Severity } from '../types';

export interface ToastItem {
  id: string;
  label: string;
  severity: Severity;
}

interface Props {
  toasts: ToastItem[];
}

function style(sev: Severity) {
  if (sev === 'critical' || sev === 'high') return { cls: 'border-red-400/60 bg-red-500/90', Icon: AlertTriangle, tag: '高危行为' };
  if (sev === 'medium') return { cls: 'border-amber-400/60 bg-amber-500/90', Icon: Bell, tag: '行为提醒' };
  return { cls: 'border-sky-400/60 bg-sky-500/90', Icon: Info, tag: '提示' };
}

/** 右下角实时事件 Toast：检测到新行为时滑入，自动消失。*/
export default function Toasts({ toasts }: Props) {
  return (
    <div className="absolute bottom-5 right-5 z-50 flex flex-col gap-2 items-end pointer-events-none">
      <AnimatePresence>
        {toasts.map((t) => {
          const { cls, Icon, tag } = style(t.severity);
          return (
            <motion.div
              key={t.id}
              initial={{ opacity: 0, x: 60, scale: 0.9 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 60, scale: 0.9 }}
              transition={{ type: 'spring', stiffness: 380, damping: 28 }}
              className={`flex items-center gap-3 rounded-xl border px-4 py-2.5 text-white shadow-2xl backdrop-blur-md ${cls}`}
            >
              <Icon size={18} className="shrink-0" />
              <div className="leading-tight">
                <div className="text-[10px] font-bold uppercase tracking-wider opacity-85">{tag}</div>
                <div className="text-sm font-bold">{t.label}</div>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
