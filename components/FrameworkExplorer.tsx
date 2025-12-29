
import React, { useState, useEffect } from 'react';
import { FrameworkData, CodeExample, VersionedSnippet } from '../types';
import CodeBlock from './CodeBlock';

interface FrameworkExplorerProps {
  framework: FrameworkData;
}

const FrameworkExplorer: React.FC<FrameworkExplorerProps> = ({ framework }) => {
  const [activeTab, setActiveTab] = useState<'python' | 'cpp' | 'go' | 'yaml'>(
    framework.id === 'golang' ? 'go' : framework.id === 'kubernetes' ? 'yaml' : 'python'
  );

  // Reset tab if framework changes and current tab is not supported
  useEffect(() => {
    if (framework.id === 'golang') {
      setActiveTab('go');
    } else if (framework.id === 'kubernetes') {
      setActiveTab('yaml');
    } else if ((activeTab === 'go' && !framework.goInstall) || (activeTab === 'yaml' && !framework.yamlInstall)) {
      setActiveTab('python');
    }
  }, [framework.id]);

  // Track selected version index for each example
  const [selectedVersions, setSelectedVersions] = useState<Record<number, number>>(
    Object.fromEntries(framework.examples.map((_, i) => [i, 0]))
  );

  const copyAllCode = (example: CodeExample, versionIdx: number) => {
    const v = example.versions[versionIdx];
    let combined = `### PYTHON SNIPPET ###\n\n${v.python}\n\n### C++ IMPLEMENTATION ###\n\n${v.cpp}`;
    if (v.go) combined += `\n\n### GOLANG IMPLEMENTATION ###\n\n${v.go}`;
    if (v.yaml) combined += `\n\n### YAML MANIFEST ###\n\n${v.yaml}`;
    navigator.clipboard.writeText(combined);
    alert('Code bundle copied to clipboard!');
  };

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 w-full">
      {/* Hero Section */}
      <div className={`p-6 lg:p-10 rounded-[2rem] bg-gradient-to-br ${framework.color} mb-8 lg:mb-12 shadow-2xl relative overflow-hidden`}>
        <div className="absolute top-0 right-0 -mt-10 -mr-10 w-48 h-48 bg-white/10 rounded-full blur-3xl" />
        <div className="flex flex-col md:flex-row items-center gap-6 lg:gap-10 text-center md:text-left relative z-10">
          <div className="w-20 h-20 lg:w-28 lg:h-28 bg-white/10 backdrop-blur-md rounded-3xl flex items-center justify-center text-5xl lg:text-7xl shadow-xl ring-1 ring-white/20">
            {framework.icon}
          </div>
          <div>
            <h1 className="text-3xl lg:text-6xl font-black text-white mb-3 tracking-tighter">{framework.name}</h1>
            <p className="text-white/80 max-w-2xl text-base lg:text-xl leading-relaxed font-medium">
              {framework.description}
            </p>
          </div>
        </div>
      </div>

      {/* Installation Section */}
      <section className="mb-12 lg:mb-16">
        <div className="flex items-center gap-2.5 mb-8 px-1">
          <div className="w-2 h-7 bg-indigo-500 rounded-full"></div>
          <h2 className="text-xl lg:text-3xl font-black text-slate-100 tracking-tight">Installation</h2>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
          <div className="bg-slate-900/40 border border-slate-800/60 rounded-[1.5rem] p-6 lg:p-8 hover:border-indigo-500/30 transition-all shadow-lg">
            <h3 className="text-sm font-black text-indigo-400 mb-5 flex items-center gap-2 uppercase tracking-widest">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" /></svg>
              {framework.id === 'golang' ? 'Go Env' : framework.id === 'kubernetes' ? 'Kubectl' : 'Python Env'}
            </h3>
            <CodeBlock 
              code={framework.id === 'golang' ? framework.goInstall || '' : framework.id === 'kubernetes' ? framework.yamlInstall || '' : framework.pythonInstall} 
              language="bash" 
            />
          </div>
          <div className="bg-slate-900/40 border border-slate-800/60 rounded-[1.5rem] p-6 lg:p-8 hover:border-emerald-500/30 transition-all shadow-lg">
            <h3 className="text-sm font-black text-emerald-400 mb-5 flex items-center gap-2 uppercase tracking-widest">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" /></svg>
              Inference Bindings
            </h3>
            <div className="bg-slate-950/60 p-5 rounded-2xl border border-slate-800/40 text-[13px] text-slate-300 leading-relaxed font-mono whitespace-pre-wrap shadow-inner h-full min-h-[100px]">
              {framework.cppInstall}
            </div>
          </div>
        </div>
      </section>

      {/* Production Optimization Section */}
      <section className="mb-12 lg:mb-16">
        <div className="flex items-center gap-2.5 mb-8 px-1">
          <div className="w-2 h-7 bg-amber-500 rounded-full"></div>
          <h2 className="text-xl lg:text-3xl font-black text-slate-100 tracking-tight">Production Tuning</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {framework.optimizationTips.map((tip, i) => (
            <div key={i} className="flex items-start gap-4 p-5 rounded-2xl bg-slate-900/30 border border-slate-800/50 hover:bg-slate-800/40 transition-colors">
              <div className="shrink-0 w-6 h-6 rounded-full bg-amber-500/20 flex items-center justify-center text-amber-500">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" /></svg>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed font-medium">{tip}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Examples Section */}
      <section className="pb-20 lg:pb-0">
        <div className="flex items-center gap-2.5 mb-10 px-1">
          <div className="w-2 h-7 bg-indigo-500 rounded-full"></div>
          <h2 className="text-xl lg:text-3xl font-black text-slate-100 tracking-tight">Production Boilerplate</h2>
        </div>

        <div className="space-y-10 lg:space-y-16">
          {framework.examples.map((example, idx) => {
            const currentVIdx = selectedVersions[idx] || 0;
            const activeVersion = example.versions[currentVIdx];
            
            return (
              <div key={idx} className="bg-slate-900/30 rounded-[2rem] p-6 lg:p-10 border border-slate-800/50 relative group shadow-xl">
                <div className="flex flex-col lg:flex-row lg:justify-between lg:items-start gap-6 mb-8">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-3">
                      <h3 className="text-2xl lg:text-3xl font-bold text-slate-100 tracking-tight group-hover:text-indigo-400 transition-colors">
                        {example.title}
                      </h3>
                      <span className="px-2 py-0.5 bg-indigo-500/20 text-indigo-400 text-[10px] font-black rounded uppercase border border-indigo-500/20">
                        {activeVersion.label}
                      </span>
                    </div>
                    <p className="text-sm lg:text-base text-slate-400 leading-relaxed max-w-3xl">{example.description}</p>
                  </div>
                  <button 
                    onClick={() => copyAllCode(example, currentVIdx)}
                    className="w-full lg:w-auto flex items-center justify-center gap-2.5 px-6 py-3.5 bg-slate-800 hover:bg-slate-700 text-white rounded-2xl text-xs font-black uppercase tracking-widest transition-all shadow-lg active:scale-95"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                    Copy Bundle
                  </button>
                </div>

                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                  <div className="flex gap-2 p-1 bg-slate-950/80 border border-slate-800/50 rounded-xl w-full sm:w-fit overflow-x-auto scrollbar-hide">
                    {activeVersion.yaml && (
                       <button
                       onClick={() => setActiveTab('yaml')}
                       className={`flex-1 lg:flex-none px-6 py-2.5 rounded-lg text-xs font-black uppercase tracking-tight transition-all ${
                         activeTab === 'yaml' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-200'
                       }`}
                     >
                       Manifest
                     </button>
                    )}
                    <button
                      onClick={() => setActiveTab('python')}
                      className={`flex-1 lg:flex-none px-6 py-2.5 rounded-lg text-xs font-black uppercase tracking-tight transition-all ${
                        activeTab === 'python' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-200'
                      }`}
                    >
                      Python
                    </button>
                    <button
                      onClick={() => setActiveTab('cpp')}
                      className={`flex-1 lg:flex-none px-6 py-2.5 rounded-lg text-xs font-black uppercase tracking-tight transition-all ${
                        activeTab === 'cpp' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-200'
                      }`}
                    >
                      C++
                    </button>
                    {activeVersion.go && (
                      <button
                        onClick={() => setActiveTab('go')}
                        className={`flex-1 lg:flex-none px-6 py-2.5 rounded-lg text-xs font-black uppercase tracking-tight transition-all ${
                          activeTab === 'go' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-200'
                        }`}
                      >
                        Go
                      </button>
                    )}
                  </div>

                  {/* Version Picker */}
                  {example.versions.length > 1 && (
                    <div className="flex items-center gap-3 bg-slate-800/40 p-1 rounded-xl border border-slate-700/50">
                      <span className="text-[10px] font-black text-slate-500 uppercase ml-3 tracking-widest">Version</span>
                      {example.versions.map((v, vidx) => (
                        <button
                          key={vidx}
                          onClick={() => setSelectedVersions(prev => ({ ...prev, [idx]: vidx }))}
                          className={`px-3 py-1.5 rounded-lg text-[10px] font-bold transition-all ${
                            currentVIdx === vidx ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          {v.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                <div className="relative group/code">
                  <CodeBlock 
                    code={activeTab === 'python' ? activeVersion.python : activeTab === 'cpp' ? activeVersion.cpp : activeTab === 'go' ? activeVersion.go || '' : activeVersion.yaml || ''} 
                    language={activeTab} 
                  />
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
};

export default FrameworkExplorer;
