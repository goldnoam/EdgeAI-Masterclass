
import React, { useState } from 'react';
import { WIZARD_STEPS, FRAMEWORKS } from '../constants';

const DeploymentWizard: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [showResult, setShowResult] = useState(false);

  const handleSelect = (optionId: string) => {
    const newAnswers = { ...answers, [WIZARD_STEPS[currentStep].id]: optionId };
    setAnswers(newAnswers);

    if (currentStep < WIZARD_STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      setShowResult(true);
    }
  };

  const reset = () => {
    setCurrentStep(0);
    setAnswers({});
    setShowResult(false);
  };

  const getRecommendation = () => {
    const hw = answers['hardware'];
    const prio = answers['priority'];

    if (hw === 'nvidia' && prio === 'latency') return {
      path: "PyTorch → ONNX → TensorRT",
      desc: "Highest GPU throughput with lowest latency.",
      tool: "NVIDIA TensorRT",
      frameworkId: 'tensorrt'
    };
    if (hw === 'mobile') return {
      path: "TensorFlow → TFLite",
      desc: "Best compatibility for Android and iOS devices.",
      tool: "TF-Lite Runtime",
      frameworkId: 'tensorflow'
    };
    if (hw === 'web') return {
      path: "ONNX → ONNX Runtime Web",
      desc: "High performance WebGL/WebGPU inference.",
      tool: "ORT Web",
      frameworkId: 'onnx'
    };
    return {
      path: "PyTorch → TorchScript → LibTorch",
      desc: "Balanced performance and C++ integration.",
      tool: "LibTorch Runtime",
      frameworkId: 'pytorch'
    };
  };

  const recommendation = showResult ? getRecommendation() : null;

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 w-full max-w-4xl mx-auto py-10 px-4">
      {!showResult ? (
        <div className="bg-slate-900/50 border border-slate-800 rounded-[2rem] p-8 lg:p-12 shadow-2xl">
          <div className="flex justify-between items-center mb-10">
            <div>
              <p className="text-xs font-black text-indigo-400 uppercase tracking-widest mb-2">Step {currentStep + 1} of {WIZARD_STEPS.length}</p>
              <h2 className="text-2xl lg:text-4xl font-black text-white tracking-tight">{WIZARD_STEPS[currentStep].question}</h2>
            </div>
            <button onClick={reset} className="text-slate-500 hover:text-slate-200 text-xs font-bold uppercase">Restart</button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {WIZARD_STEPS[currentStep].options.map(opt => (
              <button
                key={opt.id}
                onClick={() => handleSelect(opt.id)}
                className="group p-6 rounded-2xl bg-slate-800/40 border border-slate-700/50 text-left hover:bg-indigo-600/10 hover:border-indigo-500/50 transition-all active:scale-95"
              >
                <div className="flex items-center gap-4 mb-3">
                  <span className="text-3xl group-hover:scale-125 transition-transform duration-300">{opt.icon}</span>
                  <h3 className="font-bold text-lg text-slate-100">{opt.label}</h3>
                </div>
                <p className="text-sm text-slate-400 leading-relaxed">{opt.description}</p>
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="bg-slate-900 border border-indigo-500/30 rounded-[2rem] p-8 lg:p-12 shadow-2xl shadow-indigo-500/10 text-center animate-in zoom-in-95 duration-500">
          <div className="w-20 h-20 bg-indigo-600 rounded-3xl flex items-center justify-center text-4xl mx-auto mb-8 shadow-xl shadow-indigo-600/30">
            ✅
          </div>
          <h2 className="text-3xl lg:text-5xl font-black text-white mb-4 tracking-tight">Optimal Stack Found</h2>
          <div className="inline-block px-4 py-2 bg-indigo-500/20 text-indigo-400 rounded-full font-black text-xs uppercase tracking-widest mb-8">
            Recommendation
          </div>
          
          <div className="bg-slate-950 p-8 rounded-3xl border border-slate-800 mb-10 text-center">
            <h3 className="text-2xl lg:text-4xl font-mono text-indigo-400 font-bold mb-4">{recommendation?.path}</h3>
            <p className="text-slate-400 text-lg mb-0">{recommendation?.desc}</p>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button 
              onClick={reset}
              className="px-8 py-4 rounded-2xl bg-slate-800 hover:bg-slate-700 text-white font-bold transition-all"
            >
              Start Over
            </button>
            <button 
              className="px-8 py-4 rounded-2xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold shadow-lg shadow-indigo-600/20 transition-all active:scale-95"
              onClick={() => {
                // In a real app, this would navigate to the framework's page
                alert(`Navigate to ${recommendation?.tool} detailed guide!`);
              }}
            >
              Get Started Guide
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default DeploymentWizard;
