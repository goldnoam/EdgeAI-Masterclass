
import React, { useMemo } from 'react';

interface CodeBlockProps {
  code: string;
  language: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ code, language }) => {
  const [copied, setCopied] = React.useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Simple Regex-based Syntax Highlighting
  const highlightedCode = useMemo(() => {
    let html = code
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Keywords
    const keywords = /\b(import|from|class|def|return|if|else|elif|for|while|try|except|with|as|nullptr|int|float|void|auto|std|vector|const|char|return|using|namespace|include|public|private)\b/g;
    html = html.replace(keywords, '<span class="token keyword">$1</span>');

    // Functions
    const functions = /\b([a-zA-Z_][a-zA-Z0-9_]*)(?=\s*\()/g;
    html = html.replace(functions, '<span class="token function">$1</span>');

    // Strings
    const strings = /("[^"]*")|('[^']*')/g;
    html = html.replace(strings, '<span class="token string">$1</span>');

    // Numbers
    const numbers = /\b(\d+\.?\d*)\b/g;
    html = html.replace(numbers, '<span class="token number">$1</span>');

    // Comments
    const comments = /(#.*)|(\/\/.*)/g;
    html = html.replace(comments, '<span class="token comment">$1</span>');

    return html;
  }, [code]);

  return (
    <div className="relative group rounded-2xl overflow-hidden border border-slate-800 bg-slate-950/80 shadow-2xl">
      <div className="flex justify-between items-center px-5 py-3 bg-slate-900/50 border-b border-slate-800">
        <div className="flex items-center gap-2">
            <div className="flex gap-1.5 mr-2">
                <div className="w-2.5 h-2.5 rounded-full bg-slate-700"></div>
                <div className="w-2.5 h-2.5 rounded-full bg-slate-700"></div>
                <div className="w-2.5 h-2.5 rounded-full bg-slate-700"></div>
            </div>
            <span className="text-[10px] font-black uppercase tracking-widest text-slate-500">{language}</span>
        </div>
        <button 
          onClick={copyToClipboard}
          className="text-xs font-bold text-slate-400 hover:text-white transition-all flex items-center gap-2"
        >
          {copied ? (
            <span className="text-emerald-400 flex items-center gap-1.5">
               <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" /></svg>
               Copied!
            </span>
          ) : (
            <>
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              Copy
            </>
          )}
        </button>
      </div>
      <pre className="p-5 overflow-x-auto text-sm leading-relaxed code-font text-slate-300 custom-scrollbar">
        <code dangerouslySetInnerHTML={{ __html: highlightedCode }} />
      </pre>
    </div>
  );
};

export default CodeBlock;
