
import React, { useState } from 'react';
import { FRAMEWORKS } from './constants';
import { Framework } from './types';
import FrameworkExplorer from './components/FrameworkExplorer';
import ComparisonView from './components/ComparisonView';
import DeploymentWizard from './components/DeploymentWizard';
import GeminiExpert from './components/GeminiExpert';
import MLIntroduction from './components/MLIntroduction';
import ObjectDetectionSandbox from './components/ObjectDetectionSandbox';

const App: React.FC = () => {
  const [selectedFrameworkId, setSelectedFrameworkId] = useState<Framework>('pytorch');
  const [currentView, setCurrentView] = useState<'explorer' | 'comparison' | 'wizard' | 'ml-intro' | 'sandbox'>('ml-intro');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);

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
          className="fixed inset-0 bg-black/70 backdrop-blur-md z-[60] lg:hidden animate-in fade-in duration-300"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}

      {/* Main Sidebar */}
      <aside className={`
        fixed inset-y-0 left-0 z-[70] w-[85vw] max-w-[320px] bg-slate-900 border-r border-slate-800 flex flex-col 
        transition-transform duration-300 cubic-bezier(0.4, 0, 0.2, 1) lg:static lg:translate-x-0
        ${isMobileMenuOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="p-6 pt-10 lg:pt-8 border-b border-slate-800 flex items-center justify-between">
          <button 
            onClick={() => {
              setCurrentView('ml-intro');
              setIsMobileMenuOpen(false);
            }}
            className="flex items-center gap-3 group"
          >
            <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center text-xl font-bold shadow-lg shadow-indigo-500/20 group-hover:scale-105 transition-transform">
              EA
            </div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
              EdgeAI Master
            </h1>
          </button>
          <button 
            onClick={() => setIsMobileMenuOpen(false)} 
            className="lg:hidden p-2 text-slate-400 hover:text-white transition-colors"
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
                ? 'bg-indigo-600 text-white shadow-lg'
                : 'text-slate-400 hover:bg-slate-800/50'
            }`}
          >
            <span className="text-xl">🎓</span>
            <span className="font-bold text-sm">ML Introduction</span>
          </button>

          <button
            onClick={() => {
              setCurrentView('sandbox');
              setIsMobileMenuOpen(false);
            }}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 mb-2 ${
              currentView === 'sandbox'
                ? 'bg-rose-600 text-white shadow-lg'
                : 'text-slate-400 hover:bg-slate-800/50'
            }`}
          >
            <span className="text-xl">🎯</span>
            <span className="font-bold text-sm">Vision Sandbox</span>
          </button>

          <button
            onClick={() => {
              setCurrentView('wizard');
              setIsMobileMenuOpen(false);
            }}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 mb-4 ${
              currentView === 'wizard'
                ? 'bg-amber-500 text-slate-950 shadow-lg'
                : 'bg-slate-800/50 text-amber-400 hover:bg-slate-800 hover:text-amber-300'
            }`}
          >
            <span className="text-xl">🧙‍♂️</span>
            <span className="font-black uppercase text-xs tracking-widest">Setup Wizard</span>
          </button>

          <p className="px-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-4 mt-4">
            Deployment Frameworks
          </p>
          {FRAMEWORKS.map((f) => (
            <button
              key={f.id}
              onClick={() => {
                setSelectedFrameworkId(f.id);
                setCurrentView('explorer');
                setIsMobileMenuOpen(false);
              }}
              className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all duration-200 ${
                currentView === 'explorer' && selectedFrameworkId === f.id
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
            >
              <span className="text-xl shrink-0">{f.icon}</span>
              <span className="font-medium truncate">{f.name}</span>
            </button>
          ))}
        </nav>

        <div className="p-6 pb-12 lg:pb-6 border-t border-slate-800">
          <div className="bg-slate-800/40 rounded-2xl p-4 border border-slate-700/30">
            <p className="text-[11px] font-bold text-slate-500 uppercase mb-2">AI Expert</p>
            <button 
              onClick={() => setIsChatOpen(true)}
              className="w-full py-2 bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-400 border border-indigo-500/20 rounded-xl text-xs font-bold transition-all"
            >
              Ask AI Assistant ✨
            </button>
          </div>
        </div>
      </aside>

      {/* Content Main Area */}
      <main className="flex-1 flex flex-col relative overflow-hidden h-screen w-full bg-slate-950">
        {/* Global Header */}
        <header className="h-16 lg:h-20 flex-shrink-0 flex items-center justify-between px-4 lg:px-10 border-b border-slate-800 bg-slate-950/80 backdrop-blur-xl z-50 sticky top-0">
          <div className="flex items-center gap-3">
            <button 
              onClick={() => setIsMobileMenuOpen(true)}
              className="lg:hidden p-2 -ml-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-all"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" /></svg>
            </button>
            <div className="flex flex-col lg:flex-row lg:items-center lg:gap-2">
              <span className="text-[10px] lg:text-xs text-indigo-400 font-bold uppercase tracking-wider">
                {currentView === 'comparison' ? 'Global Stats' : currentView === 'wizard' ? 'Setup' : currentView === 'ml-intro' ? 'Education' : currentView === 'sandbox' ? 'Sandbox' : 'Framework'}
              </span>
              <h2 className="text-sm lg:text-base font-bold text-slate-100 truncate max-w-[150px] lg:max-w-none">
                {currentView === 'comparison' ? 'Performance Benchmarks' : currentView === 'wizard' ? 'Deployment Wizard' : currentView === 'ml-intro' ? 'Machine Learning 101' : currentView === 'sandbox' ? 'Object Detection Lab' : activeFramework.name}
              </h2>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button 
              onClick={handleShare}
              className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-full border border-slate-700 bg-slate-800/50 text-slate-200 text-xs font-medium hover:bg-slate-700 transition-all"
            >
              Share Link
            </button>
            <button 
              onClick={() => setIsChatOpen(!isChatOpen)}
              className={`p-2 lg:px-4 lg:py-2 rounded-full border transition-all flex items-center gap-2 text-xs font-bold ${
                isChatOpen 
                  ? 'bg-indigo-600 border-indigo-500 text-white' 
                  : 'bg-slate-800 border-slate-700 text-indigo-400 hover:bg-slate-700'
              }`}
            >
              <span className="hidden lg:inline">AI Expert</span>
              <span className="text-base lg:text-xs">✨</span>
            </button>
          </div>
        </header>

        {/* Scrollable Body */}
        <div className="flex-1 overflow-y-auto custom-scrollbar overflow-x-hidden relative flex">
          <div className="flex-1 h-full overflow-y-auto">
            <div className="max-w-6xl mx-auto w-full px-4 lg:px-10 py-6 lg:py-10 pb-32 lg:pb-12">
              {currentView === 'explorer' ? (
                <FrameworkExplorer framework={activeFramework} />
              ) : currentView === 'comparison' ? (
                <ComparisonView />
              ) : currentView === 'wizard' ? (
                <DeploymentWizard />
              ) : currentView === 'ml-intro' ? (
                <MLIntroduction />
              ) : (
                <ObjectDetectionSandbox />
              )}
            </div>
          </div>
          
          {/* Chat Sidebar Overlay for Desktop */}
          {isChatOpen && (
            <div className="hidden xl:block w-[400px] border-l border-slate-800 animate-in slide-in-from-right duration-300">
               <GeminiExpert framework={currentView === 'explorer' ? activeFramework.name : "Deep Learning Deployment"} />
            </div>
          )}
        </div>

        {/* Mobile Chat Overlay */}
        {isChatOpen && (
          <div className="xl:hidden fixed inset-0 z-[100] flex animate-in fade-in duration-300">
             <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setIsChatOpen(false)} />
             <div className="relative ml-auto w-[90vw] max-w-md h-full bg-slate-900 shadow-2xl animate-in slide-in-from-right duration-300">
               <GeminiExpert framework={currentView === 'explorer' ? activeFramework.name : "Deep Learning Deployment"} />
             </div>
          </div>
        )}

        {/* Mobile Bottom Navigation */}
        <div className="lg:hidden fixed bottom-0 left-0 right-0 z-50 px-4 pb-safe bg-gradient-to-t from-slate-950 via-slate-950 to-transparent pt-6 pointer-events-none">
          <div className="bg-slate-900/90 backdrop-blur-xl border border-slate-800 rounded-2xl flex items-center justify-around p-1.5 shadow-2xl pointer-events-auto max-w-sm mx-auto">
            <button 
              onClick={() => setCurrentView('explorer')}
              className={`flex-1 flex flex-col items-center gap-1 py-2 rounded-xl transition-all ${
                currentView === 'explorer' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-400'
              }`}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" strokeWidth="2" /></svg>
              <span className="text-[9px] font-black uppercase">Explorer</span>
            </button>
            <button 
              onClick={() => setCurrentView('comparison')}
              className={`flex-1 flex flex-col items-center gap-1 py-2 rounded-xl transition-all ${
                currentView === 'comparison' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-400'
              }`}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2" strokeWidth="2" /></svg>
              <span className="text-[9px] font-black uppercase">Stats</span>
            </button>
            <button 
              onClick={() => setCurrentView('sandbox')}
              className={`flex-1 flex flex-col items-center gap-1 py-2 rounded-xl transition-all ${
                currentView === 'sandbox' ? 'bg-rose-600 text-white shadow-lg' : 'text-rose-500'
              }`}
            >
              <span className="text-lg">🎯</span>
              <span className="text-[9px] font-black uppercase">Sandbox</span>
            </button>
          </div>
        </div>

        {/* Desktop Quick Nav Overlay */}
        <div className="hidden lg:flex fixed bottom-8 left-1/2 -translate-x-1/2 gap-3 bg-slate-900/90 backdrop-blur-xl border border-slate-700/50 p-2 rounded-2xl shadow-2xl z-40 animate-in slide-in-from-bottom-8">
          <button 
            onClick={() => setCurrentView('ml-intro')}
            className={`px-5 py-2.5 rounded-xl text-xs font-bold transition-all ${currentView === 'ml-intro' ? 'bg-indigo-600 text-white shadow-lg' : 'hover:bg-slate-800 text-slate-300'}`}
          >
            ML 101
          </button>
          <button 
            onClick={() => setCurrentView('explorer')}
            className={`px-5 py-2.5 rounded-xl text-xs font-bold transition-all ${currentView === 'explorer' ? 'bg-indigo-600 text-white shadow-lg' : 'hover:bg-slate-800 text-slate-300'}`}
          >
            Explorer
          </button>
          <button 
            onClick={() => setCurrentView('comparison')}
            className={`px-5 py-2.5 rounded-xl text-xs font-bold transition-all ${currentView === 'comparison' ? 'bg-indigo-600 text-white shadow-lg' : 'hover:bg-slate-800 text-slate-300'}`}
          >
            Stats
          </button>
          <div className="w-px bg-slate-700/50 mx-1 self-stretch" />
          <button 
            className={`px-5 py-2.5 rounded-xl text-xs font-bold transition-all flex items-center gap-2 ${currentView === 'sandbox' ? 'bg-rose-600 text-white shadow-lg' : 'bg-rose-600/10 text-rose-500 border border-rose-500/20 hover:bg-rose-600/20'}`}
            onClick={() => setCurrentView('sandbox')}
          >
            Vision Lab
          </button>
        </div>
      </main>
    </div>
  );
};

export default App;
