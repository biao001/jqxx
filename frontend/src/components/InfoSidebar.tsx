import { type ReactNode, useEffect, useState } from 'react';
import { AlertTriangle, CloudSun, Gauge, type LucideIcon, ScrollText, Timer } from 'lucide-react';

interface Props {
  drivingSeconds: number;
  activeBehaviors: string[];
}

const REST_LIMIT = 2 * 3600; // 建议每 2 小时休息

const RULES: { match: string; text: string }[] = [
  { match: '手机', text: '开车刷/接打手机：记 3 分、罚 200 元，事故风险陡增' },
  { match: '电话', text: '驾驶中接打电话：记 3 分、罚 200 元' },
  { match: '安全带', text: '未系安全带：记 1–2 分、罚 50 元，碰撞伤亡风险极高' },
  { match: '吸烟', text: '驾驶中吸烟分散注意力、烟灰易致慌乱，请勿吸烟' },
  { match: '离', text: '双手离开方向盘将失去车辆控制，严禁脱手驾驶' },
  { match: '饮水', text: '驾驶中饮水会分心，建议停车后再饮用' },
  { match: '进食', text: '驾驶中进食分散注意力，请勿边开车边吃' },
  { match: '疲劳', text: '疲劳驾驶反应迟钝；连续驾驶 4 小时以上记 9 分、罚款' },
  { match: '视线', text: '视线长时间偏离前方极危险，请专注路况' },
  { match: '无人', text: '驾驶位异常，请确认驾驶员在位' },
];
const SAFE_TIPS = ['保持车距，礼让行人，文明驾驶。', '系好安全带是对生命的基本保障。', '雨雾天减速慢行，及时开灯。', '疲劳/酒后切勿驾驶。'];

function fmt(s: number) {
  const hh = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = Math.floor(s % 60);
  return `${hh.toString().padStart(2, '0')}:${mm.toString().padStart(2, '0')}:${ss.toString().padStart(2, '0')}`;
}

interface Weather {
  temp?: number;
  wind?: number;
  desc?: string;
  tip?: string;
  error?: string;
}

function weatherFromCode(code: number): { desc: string; tip: string } {
  if (code === 0) return { desc: '晴', tip: '光线良好，注意强光时备好遮阳。' };
  if (code <= 3) return { desc: '多云', tip: '能见度尚可，保持正常车距。' };
  if (code === 45 || code === 48) return { desc: '雾', tip: '能见度低，开雾灯、减速、加大车距。' };
  if (code >= 51 && code <= 67) return { desc: '雨', tip: '路面湿滑，减速慢行、防止侧滑。' };
  if (code >= 71 && code <= 77) return { desc: '雪', tip: '注意结冰打滑，平稳驾驶、加大车距。' };
  if (code >= 80 && code <= 82) return { desc: '阵雨', tip: '雨势变化大，控制车速、开启雨刷。' };
  if (code >= 95) return { desc: '雷雨', tip: '恶劣天气，尽量避免出行、远离积水。' };
  return { desc: '未知', tip: '关注实时路况，安全驾驶。' };
}

function Card({ icon: Icon, title, children, accent }: { icon: LucideIcon; title: string; children: ReactNode; accent?: string }) {
  return (
    <div className="card p-4 shrink-0">
      <div className="flex items-center gap-2.5 mb-3 pb-2.5 border-b border-outline-variant/60">
        <span className={`w-1 h-5 rounded-full ${accent ?? 'bg-gradient-to-b from-primary to-primary-container'}`} />
        <Icon size={17} className="text-primary" />
        <h3 className="font-display text-base font-bold text-on-surface">{title}</h3>
      </div>
      {children}
    </div>
  );
}

/** 左侧信息栏：行驶时长 / 交规提醒 / 天气 / 打分机制。*/
export default function InfoSidebar({ drivingSeconds, activeBehaviors }: Props) {
  const [weather, setWeather] = useState<Weather | null>(null);

  useEffect(() => {
    if (!navigator.geolocation) {
      setWeather({ error: '浏览器不支持定位' });
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords;
        fetch(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,weather_code,wind_speed_10m`)
          .then((r) => r.json())
          .then((d) => {
            const cur = d.current || {};
            const wc = weatherFromCode(Number(cur.weather_code ?? -1));
            setWeather({ temp: cur.temperature_2m, wind: cur.wind_speed_10m, desc: wc.desc, tip: wc.tip });
          })
          .catch(() => setWeather({ error: '天气获取失败' }));
      },
      () => setWeather({ error: '未授权定位，无法获取天气' }),
      { timeout: 8000 },
    );
  }, []);

  const overRest = drivingSeconds >= REST_LIMIT;
  const restPct = Math.min(100, (drivingSeconds / REST_LIMIT) * 100);
  const rules = RULES.filter((r) => activeBehaviors.some((b) => b.includes(r.match)));
  const tip = SAFE_TIPS[Math.floor(drivingSeconds / 8) % SAFE_TIPS.length];

  return (
    <div className="w-[300px] shrink-0 flex flex-col gap-gutter h-full overflow-y-auto overflow-x-hidden">
      {/* 行驶时长 */}
      <Card icon={Timer} title="行驶时长" accent={overRest ? 'bg-red-500' : undefined}>
        <div className={`font-display text-4xl font-bold tabular-nums text-center ${overRest ? 'text-red-500' : 'text-slate-800'}`}>{fmt(drivingSeconds)}</div>
        <div className="mt-3 h-2 rounded-full bg-slate-200 overflow-hidden">
          <div className={`h-full rounded-full transition-all ${overRest ? 'bg-red-500' : 'bg-primary'}`} style={{ width: `${restPct}%` }} />
        </div>
        <p className={`mt-2 text-xs ${overRest ? 'text-red-500 font-bold' : 'text-outline'}`}>
          {overRest ? '⚠️ 已连续驾驶超 2 小时，请尽快休息！' : '建议每 2 小时休息一次'}
        </p>
      </Card>

      {/* 交规提醒 */}
      <Card icon={ScrollText} title="交规提醒">
        {rules.length > 0 ? (
          <div className="space-y-2">
            {rules.map((r) => (
              <div key={r.match} className="flex gap-2 text-xs text-red-600 bg-red-50 border border-red-100 rounded-lg p-2.5">
                <AlertTriangle size={15} className="shrink-0 mt-0.5" />
                <span className="leading-relaxed">{r.text}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-outline leading-relaxed">{tip}</p>
        )}
      </Card>

      {/* 天气提醒 */}
      <Card icon={CloudSun} title="行驶天气">
        {weather == null ? (
          <p className="text-xs text-outline">正在获取天气…</p>
        ) : weather.error ? (
          <p className="text-xs text-outline">{weather.error}</p>
        ) : (
          <div>
            <div className="flex items-baseline gap-2">
              <span className="font-display text-3xl font-bold text-slate-800">{weather.temp}°C</span>
              <span className="text-sm font-medium text-slate-600">{weather.desc}</span>
              <span className="text-xs text-outline ml-auto">风速 {weather.wind} km/h</span>
            </div>
            <p className="mt-2 text-xs text-primary bg-blue-50 border border-blue-100 rounded-lg p-2.5 leading-relaxed">🚗 {weather.tip}</p>
          </div>
        )}
      </Card>

      {/* 打分机制 */}
      <Card icon={Gauge} title="打分机制">
        <ul className="text-xs text-on-surface-variant space-y-1.5 leading-relaxed list-disc pl-4">
          <li><b>综合评分</b> = 100 − 行为风险×0.42 − 疲劳风险×0.36 − 违规扣分</li>
          <li><b>违规扣分</b>按严重度：低 4 / 中 9 / 高 16 / 紧急 30</li>
          <li><b>行为评分</b>只看行为风险与违规；<b>疲劳评分</b>由 PERCLOS、打哈欠、视线等决定</li>
          <li>无人/遮挡额外扣分；分数越高越安全</li>
        </ul>
        <div className="mt-3 grid grid-cols-4 gap-1 text-center text-[10px] font-bold">
          <span className="rounded bg-green-100 text-green-700 py-1">≥80 正常</span>
          <span className="rounded bg-yellow-100 text-yellow-700 py-1">≥60 注意</span>
          <span className="rounded bg-orange-100 text-orange-700 py-1">≥40 警告</span>
          <span className="rounded bg-red-100 text-red-700 py-1">&lt;40 危险</span>
        </div>
      </Card>
    </div>
  );
}
