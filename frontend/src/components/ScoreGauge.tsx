interface Props {
  score: number | null;
  size?: number;
}

/** 半圆评分仪表盘：0-100，按分数分绿/黄/红三区上色。*/
export default function ScoreGauge({ score, size = 240 }: Props) {
  const w = size;
  const h = size * 0.62;
  const cx = w / 2;
  const cy = w / 2;
  const R = w / 2 - 18;
  const sw = 16;
  const len = Math.PI * R;
  const v = score ?? 0;
  const frac = Math.max(0, Math.min(1, v / 100));
  const color = score == null ? '#cbd5e1' : v >= 80 ? '#22c55e' : v >= 60 ? '#f59e0b' : '#ef4444';
  const arc = `M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ maxWidth: w }}>
      <path d={arc} fill="none" stroke="#e8edf3" strokeWidth={sw} strokeLinecap="round" />
      <path
        d={arc}
        fill="none"
        stroke={color}
        strokeWidth={sw}
        strokeLinecap="round"
        strokeDasharray={`${frac * len} ${len}`}
        style={{ transition: 'stroke-dasharray 0.6s ease, stroke 0.4s ease' }}
      />
      <text x={cx} y={cy - 14} textAnchor="middle" style={{ fontSize: w * 0.2, fontWeight: 800, fill: color }}>
        {score ?? '--'}
      </text>
      <text x={cx} y={cy + 8} textAnchor="middle" style={{ fontSize: w * 0.058, fill: '#94a3b8', fontWeight: 600 }}>
        / 100
      </text>
    </svg>
  );
}
