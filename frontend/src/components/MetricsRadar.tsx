import { DrivingStats } from '../types';

const AXES: { key: keyof DrivingStats; label: string }[] = [
  { key: 'focus', label: '专注度' },
  { key: 'reaction', label: '反应速度' },
  { key: 'compliance', label: '操作规范' },
  { key: 'fatigue', label: '清醒程度' },
  { key: 'stability', label: '状态稳定' },
];

interface Props {
  stats: DrivingStats | null;
  size?: number;
}

/** 五维状态雷达图（真实数据，替代原占位装饰）。*/
export default function MetricsRadar({ stats, size = 200 }: Props) {
  const cx = size / 2;
  const cy = size / 2;
  const R = size / 2 - 26;
  const n = AXES.length;
  const angle = (i: number) => (-90 + (i * 360) / n) * (Math.PI / 180);
  const pt = (i: number, r: number): [number, number] => [cx + r * Math.cos(angle(i)), cy + r * Math.sin(angle(i))];
  const poly = (pts: [number, number][]) => pts.map((p) => p.join(',')).join(' ');

  const gridLevels = [0.25, 0.5, 0.75, 1];
  const valuePts = AXES.map(({ key }, i) => pt(i, R * (((stats?.[key] as number) ?? 0) / 100)));

  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="w-full" style={{ maxWidth: size }}>
      {/* 网格 */}
      {gridLevels.map((lv) => (
        <polygon
          key={lv}
          points={poly(AXES.map((_, i) => pt(i, R * lv)))}
          fill="none"
          stroke="#e2e8f0"
          strokeWidth={1}
        />
      ))}
      {/* 轴线 */}
      {AXES.map((_, i) => {
        const [x, y] = pt(i, R);
        return <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke="#e2e8f0" strokeWidth={1} />;
      })}
      {/* 数值多边形 */}
      <polygon
        points={poly(valuePts)}
        fill="rgba(59,130,246,0.22)"
        stroke="#3b82f6"
        strokeWidth={2}
        style={{ transition: 'all 0.5s ease' }}
      />
      {valuePts.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={2.6} fill="#3b82f6" />
      ))}
      {/* 标签 */}
      {AXES.map(({ key, label }, i) => {
        const [x, y] = pt(i, R + 14);
        const anchor = Math.abs(x - cx) < 6 ? 'middle' : x > cx ? 'start' : 'end';
        return (
          <text key={key} x={x} y={y} textAnchor={anchor} dominantBaseline="middle" style={{ fontSize: 10, fontWeight: 700, fill: '#64748b' }}>
            {label}
          </text>
        );
      })}
    </svg>
  );
}
