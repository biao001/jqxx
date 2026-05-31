import { AnimatePresence, motion } from 'motion/react';
import { AlertTriangle, VolumeX } from 'lucide-react';
import { AlertLevel } from '../lib/useDriverAlerts';

interface Props {
  level: AlertLevel;
  text: string;
  muted: boolean;
}

/** 分级告警的顶部横幅(critical 红/warning 橙)。边缘呼吸光由 EdgeGlow 负责(分级升级)。*/
export default function AlertOverlay({ level, text, muted }: Props) {
  const isCritical = level === 'critical';
  return (
    <>
      {/* 顶部告警横幅 */}
      <AnimatePresence>
        {level !== 'none' && (
          <motion.div
            key="banner"
            initial={{ y: -48, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -48, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 320, damping: 26 }}
            className={`absolute top-0 left-0 right-0 z-40 flex items-center justify-center gap-3 px-5 py-3 text-white font-bold shadow-lg ${
              isCritical ? 'bg-red-600/95' : 'bg-orange-500/95'
            }`}
          >
            <motion.span
              animate={isCritical ? { scale: [1, 1.18, 1] } : { scale: 1 }}
              transition={{ duration: 0.7, repeat: Infinity }}
            >
              <AlertTriangle size={20} />
            </motion.span>
            <span className="tracking-wide">
              {isCritical ? '危险预警' : '分心预警'} · {text}
            </span>
            {muted && (
              <span className="ml-2 inline-flex items-center gap-1 text-xs font-medium opacity-80">
                <VolumeX size={14} /> 已静音
              </span>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
