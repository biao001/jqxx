import { AnimatePresence, motion } from 'motion/react';
import { AlertTriangle, VolumeX } from 'lucide-react';
import { AlertLevel } from '../lib/useDriverAlerts';

interface Props {
  level: AlertLevel;
  text: string;
  muted: boolean;
}

/** 分级告警的视觉层：critical 全屏红色边缘脉冲 + 顶部横幅；warning 顶部橙色横幅。*/
export default function AlertOverlay({ level, text, muted }: Props) {
  const isCritical = level === 'critical';
  return (
    <>
      {/* critical：边缘红色脉冲 */}
      <AnimatePresence>
        {isCritical && (
          <motion.div
            key="pulse"
            className="absolute inset-0 pointer-events-none z-30"
            initial={{ opacity: 0 }}
            animate={{ opacity: [0.45, 1, 0.45] }}
            exit={{ opacity: 0 }}
            transition={{ duration: 1.1, repeat: Infinity, ease: 'easeInOut' }}
            style={{ boxShadow: 'inset 0 0 0 6px rgba(239,68,68,0.95), inset 0 0 70px rgba(239,68,68,0.45)' }}
          />
        )}
      </AnimatePresence>

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
