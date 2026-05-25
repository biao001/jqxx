import { useRef, useState, useEffect } from 'react';
import { Bot, Send, User, X } from 'lucide-react';

interface Msg {
  role: 'user' | 'assistant';
  content: string;
}

interface Props {
  open: boolean;
  onClose: () => void;
  backendUrl: string;
  context: Record<string, unknown>;
}

const SUGGESTIONS = ['我这趟开得怎么样？', '我有哪些需要改进的？', '现在该注意什么？'];

/** AI 驾驶问答：就当前驾驶监测数据与 AI 对话。*/
export default function AiChatModal({ open, onClose, backendUrl, context }: Props) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, loading]);

  if (!open) return null;

  const send = (q: string) => {
    const question = q.trim();
    if (!question || loading) return;
    const history = messages.slice(-6);
    setMessages((m) => [...m, { role: 'user', content: question }]);
    setInput('');
    setLoading(true);
    fetch(`${backendUrl}/api/llm/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, context, history }),
    })
      .then((r) => r.json())
      .then((d) => setMessages((m) => [...m, { role: 'assistant', content: d.answer || '(无回答)' }]))
      .catch(() => setMessages((m) => [...m, { role: 'assistant', content: '请求失败，请检查后端/网络。' }]))
      .finally(() => setLoading(false));
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 backdrop-blur-sm" onClick={onClose}>
      <div className="w-[560px] h-[600px] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <div className="px-5 h-14 flex items-center justify-between border-b border-outline-variant">
          <h3 className="font-display font-bold text-slate-800 flex items-center gap-2"><Bot size={18} className="text-primary" /> AI 驾驶助手</h3>
          <button onClick={onClose} className="p-1.5 rounded-full hover:bg-surface-container text-slate-500"><X size={20} /></button>
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
          {messages.length === 0 && (
            <div className="text-center text-slate-400 text-sm mt-8">
              <Bot size={40} className="mx-auto mb-2 opacity-50" />
              <p>就当前驾驶状态向我提问</p>
              <div className="flex flex-col items-center gap-2 mt-4">
                {SUGGESTIONS.map((s) => (
                  <button key={s} onClick={() => send(s)} className="px-3 py-1.5 rounded-full bg-slate-100 hover:bg-slate-200 text-xs text-slate-600">
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`flex gap-2 ${m.role === 'user' ? 'flex-row-reverse' : ''}`}>
              <div className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 ${m.role === 'user' ? 'bg-primary text-white' : 'bg-slate-200 text-slate-600'}`}>
                {m.role === 'user' ? <User size={15} /> : <Bot size={15} />}
              </div>
              <div className={`max-w-[75%] px-3 py-2 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${m.role === 'user' ? 'bg-primary text-white rounded-tr-sm' : 'bg-slate-100 text-slate-700 rounded-tl-sm'}`}>
                {m.content}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex gap-2">
              <div className="w-7 h-7 rounded-full bg-slate-200 text-slate-600 flex items-center justify-center"><Bot size={15} /></div>
              <div className="px-3 py-2 rounded-2xl bg-slate-100 text-slate-400 text-sm">思考中…</div>
            </div>
          )}
        </div>

        <div className="p-3 border-t border-outline-variant flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && send(input)}
            placeholder="输入问题，回车发送…"
            className="flex-1 px-3 py-2 rounded-lg border border-outline-variant text-sm focus:outline-none focus:border-primary"
          />
          <button onClick={() => send(input)} disabled={loading || !input.trim()} className="px-4 rounded-lg bg-primary text-white disabled:opacity-40">
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
