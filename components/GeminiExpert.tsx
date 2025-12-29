
import React, { useState, useRef, useEffect } from 'react';
import { getAIAssistance } from '../services/geminiService';
import { ChatMessage } from '../types';

const GeminiExpert: React.FC<{ framework: string }> = ({ framework }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMsg: ChatMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    const responseText = await getAIAssistance(input, framework);
    const aiMsg: ChatMessage = { role: 'assistant', content: responseText || 'No response.' };
    
    setMessages(prev => [...prev, aiMsg]);
    setLoading(false);
  };

  return (
    <div className="flex flex-col h-full bg-slate-900/80 border-l border-slate-700 w-full max-w-md">
      <div className="p-6 border-b border-slate-700 flex items-center justify-between">
        <div>
          <h3 className="text-xl font-bold text-white flex items-center gap-2">
            <span className="text-indigo-400">✨</span> Gemini Expert
          </h3>
          <p className="text-xs text-slate-500">Optimizing {framework} for Edge AI</p>
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 && (
          <div className="text-center py-10 opacity-50">
            <p className="text-sm">Ask me for a specific C++ boilerplate or optimization tip!</p>
          </div>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
              msg.role === 'user' 
                ? 'bg-indigo-600 text-white rounded-tr-none' 
                : 'bg-slate-800 text-slate-200 rounded-tl-none border border-slate-700'
            }`}>
              {msg.content.split('\n').map((line, i) => (
                <p key={i} className={line.trim().startsWith('`') ? 'code-font bg-black/30 p-1 rounded mt-1' : ''}>
                  {line}
                </p>
              ))}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-800 rounded-2xl px-4 py-3 text-slate-400 animate-pulse text-sm">
              Thinking...
            </div>
          </div>
        )}
      </div>

      <div className="p-6 bg-slate-900 border-t border-slate-700">
        <div className="relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask for custom code..."
            className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 pr-12 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 text-slate-200 transition-all"
          />
          <button
            onClick={handleSend}
            disabled={loading}
            className="absolute right-2 top-2 p-1.5 text-indigo-400 hover:text-white transition-colors"
          >
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default GeminiExpert;
