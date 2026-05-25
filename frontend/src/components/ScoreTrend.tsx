interface Props {
  history: number[];
}

const color = (v: number) => (v >= 80 ? '#22c55e' : v >= 60 ? '#f59e0b' : '#ef4444');

/** 综合评分实时趋势折线图(带渐变填充)。*/
export default function ScoreTrend({ history }: Props) {
  const w = 300;
  const h = 72;
  if (history.length < 2) {
    return <div className="h-[72px] flex items-center justify-center text-xs text-slate-400">开始分析后显示评分趋势…</div>;
  }
  const n = history.length;
  const x = (i: number) => (i / (n - 1)) * w;
  const y = (s: number) => h - (Math.max(0, Math.min(100, s)) / 100) * (h - 6) - 3;
  const line = history.map((s, i) => `${x(i).toFixed(1)},${y(s).toFixed(1)}`).join(' ');
  const area = `0,${h} ${line} ${w},${h}`;
  const last = history[n - 1];
  const c = color(last);
  const avg = Math.round(history.reduce((a, b) => a + b, 0) / n);
  const min = Math.min(...history);

  return (
    <div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ height: 72 }} preserveAspectRatio="none">
        <defs>
          <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={c} stopOpacity="0.28" />
            <stop offset="100%" stopColor={c} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {[80, 60].map((g) => (
          <line key={g} x1="0" y1={y(g)} x2={w} y2={y(g)} stroke="#e2e8f0" strokeWidth="1" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
        ))}
        <polygon points={area} fill="url(#trendFill)" />
        <polyline
          points={line}
          fill="none"
          stroke={c}
          strokeWidth="2"
          strokeLinejoin="round"
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
        />
        <circle cx={x(n - 1)} cy={y(last)} r="3" fill={c} />
      </svg>
      <div className="flex justify-between mt-1.5 text-[11px] text-outline">
        <span>当前 <b style={{ color: c }}>{last}</b></span>
        <span>均值 {avg}</span>
        <span>最低 {min}</span>
      </div>
    </div>
  );
}
