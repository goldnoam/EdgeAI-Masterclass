
import React from 'react';

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

  return (
    <div className="relative group rounded-xl overflow-hidden border border-slate-700 bg-slate-900/50 shadow-inner">
      <div className="flex justify-between items-center px-4 py-2 bg-slate-800 border-b border-slate-700">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">{language}</span>
        <button 
          onClick={copyToClipboard}
          className="text-xs text-slate-400 hover:text-white transition-colors flex items-center gap-1"
        >
          {copied ? (
            <><span className="text-green-400">Copied!</span></>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
              </svg>
              Copy
            </>
          )}
        </button>
      </div>
      <pre className="p-4 overflow-x-auto text-sm leading-relaxed code-font text-slate-200">
        <code>{code}</code>
      </pre>
    </div>
  );
};

export default CodeBlock;
