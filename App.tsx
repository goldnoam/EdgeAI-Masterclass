
import React, { useState } from 'react';
import { FRAMEWORKS } from './constants';
import { Framework } from './types';
import FrameworkExplorer from './components/FrameworkExplorer';
import ComparisonView from './components/ComparisonView';
import DeploymentWizard from './components/DeploymentWizard';
import MLIntroduction from './components/MLIntroduction';

const App: React.FC = () => {
  const [selectedFrameworkId, setSelectedFrameworkId] = useState<Framework>('pytorch');
  const [currentView, setCurrentView] = useState<'explorer' | 'comparison' | 'wizard' | 'ml-intro'>('ml-intro');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const activeFramework = FRAMEWORKS.find(f => f.id === selectedFrameworkId)!;

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'EdgeAI Masterclass',
          text: `Master Deep Learning Deployment!`,
          url: window.location.href,
        });
      } catch (err) {
        console.error('Error sharing:', err);
      }
    } else {
      navigator.clipboard.writeText(window.location.href);
      alert('Link copied to clipboard!');
    }
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-slate-950 text-slate-200">
      {/* Mobile Sidebar Overlay */}
      {isMobileMenuOpen && (
        <div 
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[60] lg:hidden animate-in fade-in duration-300"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}

      {/* Main Sidebar */}
      <aside className={`
        fixed inset-y-0 left-0 z-[70] w-[85vw] max-w-[320px] bg-slate-900/95 lg:bg-slate-900 border-r border-slate-800/60 flex flex-col 
        transition-transform duration-300 lg:static lg:translate-x-0
        ${isMobileMenuOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="p-8 pt-10 lg:pt-8 border-b border-slate-800 flex items-center justify-between">
          <button 
            onClick={() => {
              setCurrentView('ml-intro');
              setIsMobileMenuOpen(false);
            }}
            className="flex items-center gap-3 group"
          >
            <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center text-xl font-black shadow-lg shadow-indigo-500/30 group-hover:rotate-6 transition-transform">
              EA
            </div>
            <h1 className="text-xl font-black tracking-tighter text-white">
              EdgeAI <span className="text-indigo-400">Hub</span>
            </h1>
          </button>
          <button 
            onClick={() => setIsMobileMenuOpen(false)} 
            className="lg:hidden p-2 text-slate-400 hover:text-white"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto p-4 space-y-1 custom-scrollbar">
          <button
            onClick={() => {
              setCurrentView('ml-intro');
              setIsMobileMenuOpen(false);
            }}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 mb-2 ${
              currentView === 'ml-intro'
                ? 'bg-slate-800 text-white shadow-lg ring-1 ring-white/10'
                : 'text-slate-400 hover:bg-slate-800/50'
            }`}
          >
            <span className="text-xl">🎓</span>
            <span className="font-bold text-sm tracking-tight">Introduction</span>
          </button>

          <button
            onClick={() => {
              setCurrentView('wizard');
              setIsMobileMenuOpen(false);
            }}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 mb-6 ${
              currentView === 'wizard'
                ? 'bg-amber-500 text-slate-950 shadow-lg font-black'
                : 'bg-slate-800/30 text-amber-500/80 hover:bg-slate-800 hover:text-amber-400'
            }`}
          >
            <span className="text-xl">🧙‍♂️</span>
            <span className="uppercase text-[10px] font-black tracking-[0.2em]">Deployment Wizard</span>
          </button>

          <p className="px-4 text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4 mt-8 opacity-50">
            Framework Documentation
          </p>
          {FRAMEWORKS.map((f) => (
            <button
              key={f.id}
              onClick={() => {
                setSelectedFrameworkId(f.id);
                setCurrentView('explorer');
                setIsMobileMenuOpen(false);
              }}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 ${
                currentView === 'explorer' && selectedFrameworkId === f.id
                  ? 'bg-indigo-600/10 text-white ring-1 ring-indigo-500/40'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
            >
              <span className="text-xl shrink-0 opacity-80">{f.icon}</span>
              <span className="font-bold text-sm tracking-tight truncate">{f.name}</span>
            </button>
          ))}
        </nav>

        <div className="p-6 pb-12 lg:pb-8 border-t border-slate-800">
           <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800/50">
             <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1.5">Version</p>
             <p className="text-xs text-slate-400 font-medium">v1.2.0 Production-Ready</p>
           </div>
        </div>
      </aside>

      {/* Content Main Area */}
      <main className="flex-1 flex flex-col relative overflow-hidden bg-slate-950">
        {/* Global Header */}
        <header className="h-16 lg:h-20 flex-shrink-0 flex items-center justify-between px-6 lg:px-12 border-b border-slate-800/50 bg-slate-950/90 backdrop-blur-xl z-50 sticky top-0">
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setIsMobileMenuOpen(true)}
              className="lg:hidden p-2 -ml-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-xl transition-all"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M4 6h16M4 12h16M4 18h16" /></svg>
            </button>
            <div className="flex flex-col">
              <span className="text-[10px] text-indigo-400 font-black uppercase tracking-widest">
                {currentView === 'comparison' ? 'Statistics' : currentView === 'wizard' ? 'Wizard' : currentView === 'ml-intro' ? 'Overview' : 'Documentation'}
              </span>
              <h2 className="text-sm lg:text-lg font-black text-white tracking-tight">
                {currentView === 'comparison' ? 'Benchmarking' : currentView === 'wizard' ? 'Stack Generator' : currentView === 'ml-intro' ? 'Introduction' : activeFramework.name}
              </h2>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button 
              onClick={handleShare}
              className="px-5 py-2 rounded-full border border-slate-700 bg-slate-800 text-slate-200 text-xs font-black uppercase tracking-tight hover:bg-slate-700 transition-all active:scale-95"
            >
              Share
            </button>
          </div>
        </header>

        {/* Scrollable Body */}
        <div className="flex-1 overflow-y-auto custom-scrollbar relative">
          <div className="max-w-6xl mx-auto w-full px-6 lg:px-12 py-8 lg:py-16 pb-40 lg:pb-24">
            {currentView === 'explorer' ? (
              <FrameworkExplorer framework={activeFramework} />
            ) : currentView === 'comparison' ? (
              <ComparisonView />
            ) : currentView === 'wizard' ? (
              <DeploymentWizard />
            ) : (
              <MLIntroduction />
            )}
          </div>

          {/* Persistent Desktop Tabs Overlay */}
          <div className="hidden lg:flex fixed bottom-10 left-1/2 -translate-x-1/2 gap-2 bg-slate-900/90 backdrop-blur-2xl border border-slate-800/80 p-1.5 rounded-2xl shadow-2xl z-40">
            <button 
              onClick={() => setCurrentView('ml-intro')}
              className={`px-6 py-3 rounded-xl text-xs font-black uppercase tracking-widest transition-all ${currentView === 'ml-intro' ? 'bg-white text-slate-950' : 'text-slate-400 hover:bg-slate-800'}`}
            >
              Intro
            </button>
            <button 
              onClick={() => setCurrentView('explorer')}
              className={`px-6 py-3 rounded-xl text-xs font-black uppercase tracking-widest transition-all ${currentView === 'explorer' ? 'bg-white text-slate-950' : 'text-slate-400 hover:bg-slate-800'}`}
            >
              Docs
            </button>
            <button 
              onClick={() => setCurrentView('comparison')}
              className={`px-6 py-3 rounded-xl text-xs font-black uppercase tracking-widest transition-all ${currentView === 'comparison' ? 'bg-white text-slate-950' : 'text-slate-400 hover:bg-slate-800'}`}
            >
              Stats
            </button>
          </div>
        </div>

        {/* Mobile Nav Bar */}
        <div className="lg:hidden fixed bottom-0 left-0 right-0 z-50 px-6 pb-safe bg-gradient-to-t from-slate-950 via-slate-950 to-transparent pt-8">
          <div className="bg-slate-900/95 backdrop-blur-xl border border-slate-800 rounded-3xl flex items-center justify-around p-2 shadow-2xl mb-4">
            <button 
              onClick={() => setCurrentView('ml-intro')}
              className={`flex-1 py-3 rounded-2xl transition-all ${currentView === 'ml-intro' ? 'bg-indigo-600 text-white' : 'text-slate-500'}`}
            >
              <span className="text-[10px] font-black uppercase tracking-widest">Home</span>
            </button>
            <button 
              onClick={() => setCurrentView('explorer')}
              className={`flex-1 py-3 rounded-2xl transition-all ${currentView === 'explorer' ? 'bg-indigo-600 text-white' : 'text-slate-500'}`}
            >
              <span className="text-[10px] font-black uppercase tracking-widest">Docs</span>
            </button>
            <button 
              onClick={() => setCurrentView('comparison')}
              className={`flex-1 py-3 rounded-2xl transition-all ${currentView === 'comparison' ? 'bg-indigo-600 text-white' : 'text-slate-500'}`}
            >
              <span className="text-[10px] font-black uppercase tracking-widest">Stats</span>
            </button>
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
