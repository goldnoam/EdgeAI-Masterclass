
import React from 'react';
import { COMPARISON_FEATURES, BENCHMARK_METRICS, TRAINING_BENCHMARKS, FRAMEWORKS } from '../constants';

const ComparisonView: React.FC = () => {
  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 w-full">
      <div className="p-6 lg:p-10 rounded-[2rem] bg-gradient-to-br from-indigo-600 to-indigo-900 mb-8 shadow-2xl shadow-indigo-900/20 relative overflow-hidden">
        <div className="absolute top-0 right-0 -mt-10 -mr-10 w-64 h-64 bg-white/10 rounded-full blur-3xl" />
        <h1 className="text-3xl lg:text-5xl font-black text-white mb-3 tracking-tight">Ecosystem Stats</h1>
        <p className="text-indigo-100 max-w-2xl text-base lg:text-xl leading-relaxed opacity-90">
          Raw performance data for a standard Image Classification task (ResNet-50) to guide your deployment strategy.
        </p>
      </div>

      {/* Feature Comparison */}
      <div className="mb-12">
        <div className="flex items-center justify-between mb-6 px-1">
          <h2 className="text-xl lg:text-2xl font-bold text-slate-100">Capability Matrix</h2>
          <div className="lg:hidden text-[10px] text-slate-500 font-bold uppercase flex items-center gap-1">
            Swipe Table <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M14 5l7 7m0 0l-7 7m7-7H3" strokeWidth="3" /></svg>
          </div>
        </div>
        
        <div className="bg-slate-900/40 rounded-2xl lg:rounded-3xl border border-slate-800/50 overflow-hidden shadow-xl">
          <div className="overflow-x-auto scrollbar-hide lg:scrollbar-default">
            <table className="w-full text-left border-collapse min-w-[900px]">
              <thead>
                <tr className="bg-slate-950/40 border-b border-slate-800">
                  <th className="py-5 px-6 text-slate-500 font-black uppercase tracking-tighter text-[10px] lg:text-xs sticky left-0 bg-slate-900/90 backdrop-blur-md z-10 shadow-[4px_0_12px_rgba(0,0,0,0.1)]">Feature</th>
                  {FRAMEWORKS.map(f => (
                    <th key={f.id} className="py-5 px-4 text-center min-w-[100px]">
                      <div className="flex flex-col items-center gap-1.5 group">
                        <span className="text-xl lg:text-2xl transition-transform group-hover:scale-125 duration-300">{f.icon}</span>
                        <span className="text-slate-300 font-black text-[10px] lg:text-xs tracking-tight">{f.name.split(' ')[0]}</span>
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/40">
                {COMPARISON_FEATURES.map((feature, i) => (
                  <tr key={i} className="hover:bg-slate-800/30 transition-colors group text-center">
                    <td className="py-5 px-6 font-bold text-slate-300 text-sm border-r border-slate-800/30 text-left sticky left-0 bg-slate-900/90 backdrop-blur-md z-10 shadow-[4px_0_12px_rgba(0,0,0,0.1)] group-hover:text-white">
                      {feature.name}
                    </td>
                    {FRAMEWORKS.map(f => (
                      <td key={f.id} className="py-5 px-4 text-xs lg:text-sm text-slate-400">
                        {(feature as any)[f.id]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Inference Benchmarks */}
      <h2 className="text-xl lg:text-2xl font-bold text-slate-100 mb-8 px-1">Inference Performance</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 lg:gap-8 mb-12">
        {BENCHMARK_METRICS.map((benchmark, i) => (
          <BenchmarkCard key={i} benchmark={benchmark} />
        ))}
      </div>

      {/* Training Benchmarks */}
      <div className="flex items-center gap-3 mb-8 px-1">
        <h2 className="text-xl lg:text-2xl font-bold text-slate-100">Training Performance</h2>
        <span className="bg-indigo-500/10 text-indigo-400 text-[10px] font-black uppercase px-2 py-0.5 rounded border border-indigo-500/20">Task: Image Classification</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8">
        {TRAINING_BENCHMARKS.map((benchmark, i) => (
          <BenchmarkCard key={i} benchmark={benchmark} />
        ))}
      </div>
    </div>
  );
};

const BenchmarkCard: React.FC<{ benchmark: any }> = ({ benchmark }) => (
  <div className="bg-slate-900/40 border border-slate-800/60 rounded-3xl p-6 lg:p-8 hover:border-slate-700/80 transition-all shadow-lg flex flex-col">
    <div className="flex items-start justify-between mb-2">
      <h3 className="text-slate-100 font-bold lg:text-lg tracking-tight leading-tight">{benchmark.metric}</h3>
      <span className="px-2 py-1 bg-slate-800 rounded-lg text-[10px] font-black text-slate-400 uppercase tracking-widest">{benchmark.unit}</span>
    </div>
    <p className="text-[11px] text-slate-500 mb-8 italic">Validated on NVIDIA RTX 3080 Desktop / LibTorch backend</p>
    
    <div className="space-y-4 flex-1">
      {FRAMEWORKS.map(f => {
        const val = (benchmark as any)[f.id];
        const max = Math.max(...FRAMEWORKS.map(fw => (benchmark as any)[fw.id]));
        const percentage = max > 0 ? (val / max) * 100 : 0;
        
        return (
          <div key={f.id} className="group">
            <div className="flex justify-between text-[11px] font-bold mb-1.5 uppercase tracking-wide">
              <span className="text-slate-400 flex items-center gap-1.5 group-hover:text-slate-200 transition-colors">
                <span className="text-sm">{f.icon}</span> {f.name.split(' ')[0]}
              </span>
              <span className="font-mono text-indigo-400 group-hover:scale-110 transition-transform">
                {val === 0 ? 'N/A' : val.toFixed(3).replace(/\.?0+$/, '')}
              </span>
            </div>
            <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden shadow-inner">
              <div 
                className={`h-full bg-gradient-to-r ${f.color} transition-all duration-1000 ease-out`} 
                style={{ width: `${percentage}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  </div>
);

export default ComparisonView;
