import { useState } from 'react';
import { ArrowLeft, Lock, LogIn, User, UserPlus } from 'lucide-react';

interface Props {
  onSuccess: (user: string) => void;
  onBack: () => void;
}

// 展示用：账号存浏览器 localStorage，不接后端
function loadUsers(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem('dms_users') || '{}');
  } catch {
    return {};
  }
}

/** 登录 / 注册(展示用，localStorage)。*/
export default function AuthPage({ onSuccess, onBack }: Props) {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [msg, setMsg] = useState('');

  const submit = () => {
    const u = username.trim();
    if (!u || !password) return setMsg('请输入用户名和密码');
    const users = loadUsers();
    if (mode === 'register') {
      if (users[u]) return setMsg('该用户名已存在');
      users[u] = password;
      localStorage.setItem('dms_users', JSON.stringify(users));
      localStorage.setItem('dms_user', u);
      onSuccess(u);
    } else {
      if (!users[u]) return setMsg('用户不存在，请先注册');
      if (users[u] !== password) return setMsg('密码错误');
      localStorage.setItem('dms_user', u);
      onSuccess(u);
    }
  };

  return (
    <div className="h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-slate-800 to-blue-950 p-4">
      <div className="w-[400px] bg-white rounded-2xl shadow-2xl p-8 relative">
        <button onClick={onBack} className="absolute left-4 top-4 text-slate-400 hover:text-slate-600 flex items-center gap-1 text-sm">
          <ArrowLeft size={16} /> 返回
        </button>

        <div className="text-center mb-6 mt-4">
          <div className="w-14 h-14 mx-auto rounded-2xl bg-gradient-to-br from-primary to-primary-container flex items-center justify-center shadow-lg shadow-primary/30 mb-3">
            {mode === 'login' ? <LogIn size={26} className="text-white" /> : <UserPlus size={26} className="text-white" />}
          </div>
          <h2 className="font-display text-2xl font-bold text-slate-900">{mode === 'login' ? '登录' : '注册'}</h2>
          <p className="text-sm text-slate-400 mt-1">驾驶状态推演与行为测算平台</p>
        </div>

        <div className="space-y-3">
          <div className="flex items-center gap-2 px-3 py-2.5 rounded-xl border border-outline-variant focus-within:border-primary transition-colors">
            <User size={18} className="text-slate-400" />
            <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="用户名" className="flex-1 outline-none text-sm bg-transparent" />
          </div>
          <div className="flex items-center gap-2 px-3 py-2.5 rounded-xl border border-outline-variant focus-within:border-primary transition-colors">
            <Lock size={18} className="text-slate-400" />
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && submit()}
              placeholder="密码"
              className="flex-1 outline-none text-sm bg-transparent"
            />
          </div>
          {msg && <p className="text-xs text-red-500">{msg}</p>}
          <button onClick={submit} className="w-full btn-primary py-3 text-base">
            {mode === 'login' ? '登 录' : '注 册'}
          </button>
        </div>

        <div className="mt-5 text-center text-sm text-slate-500">
          {mode === 'login' ? '还没有账号？' : '已有账号？'}
          <button onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setMsg(''); }} className="text-primary font-semibold ml-1 hover:underline">
            {mode === 'login' ? '去注册' : '去登录'}
          </button>
        </div>
        <p className="mt-4 text-center text-[11px] text-slate-300">演示账号体系，数据仅存于本地浏览器</p>
      </div>
    </div>
  );
}
