
import React from 'react';

const MLIntroduction: React.FC = () => {
  const concepts = [
    { title: "Supervised Learning", desc: "Training with labeled data (e.g., this is a cat, this is a dog).", icon: "🏷️" },
    { title: "Neural Networks", desc: "Layers of interconnected nodes inspired by the human brain.", icon: "🧠" },
    { title: "Backpropagation", desc: "The algorithm used to update model weights based on errors.", icon: "📉" },
    { title: "Inference", desc: "Using a trained model to make predictions on new, unseen data.", icon: "🚀" }
  ];

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 w-full max-w-5xl mx-auto py-6">
      <div className="p-8 lg:p-12 rounded-[2.5rem] bg-slate-900 border border-slate-800 shadow-2xl relative overflow-hidden mb-12">
        <div className="absolute top-0 right-0 -mt-20 -mr-20 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl" />
        <h1 className="text-4xl lg:text-6xl font-black text-white mb-6 tracking-tighter">What is Machine Learning?</h1>
        <p className="text-slate-400 text-lg lg:text-xl leading-relaxed max-w-3xl mb-10">
          At its core, Machine Learning is about building systems that learn patterns from data rather than being explicitly programmed for every scenario.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {concepts.map((c, i) => (
            <div key={i} className="p-6 rounded-3xl bg-slate-800/40 border border-slate-700/50 hover:border-indigo-500/50 transition-all group">
              <div className="text-4xl mb-4 group-hover:scale-110 transition-transform">{c.icon}</div>
              <h3 className="text-xl font-bold text-slate-100 mb-2">{c.title}</h3>
              <p className="text-slate-400 text-sm leading-relaxed">{c.desc}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-8">
          <section>
            <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
              <span className="w-1.5 h-6 bg-indigo-500 rounded-full" />
              The Training Process
            </h2>
            <div className="prose prose-invert max-w-none text-slate-300">
              <p>Training involves feeding a model thousands of examples. Each time, the model makes a guess, calculates the loss (error), and tweaks its internal parameters to minimize that error next time.</p>
              <div className="bg-slate-950 p-6 rounded-2xl border border-slate-800 font-mono text-xs text-indigo-400">
                while model_not_converged:<br/>
                &nbsp;&nbsp;batch = data_loader.next()<br/>
                &nbsp;&nbsp;prediction = model(batch)<br/>
                &nbsp;&nbsp;loss = criterion(prediction, target)<br/>
                &nbsp;&nbsp;loss.backward()<br/>
                &nbsp;&nbsp;optimizer.step()
              </div>
            </div>
          </section>
        </div>
        
        <div className="bg-indigo-600/10 border border-indigo-500/20 rounded-3xl p-8 h-fit">
          <h3 className="text-xl font-bold text-indigo-400 mb-4">Why Edge AI?</h3>
          <ul className="space-y-4 text-sm text-slate-300">
            <li className="flex gap-3">
              <span className="text-indigo-500">⚡</span>
              <span><strong>Latency:</strong> Instant response without cloud round-trips.</span>
            </li>
            <li className="flex gap-3">
              <span className="text-indigo-500">🔒</span>
              <span><strong>Privacy:</strong> Data never leaves the device.</span>
            </li>
            <li className="flex gap-3">
              <span className="text-indigo-500">📶</span>
              <span><strong>Offline:</strong> Works in remote areas without internet.</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default MLIntroduction;
