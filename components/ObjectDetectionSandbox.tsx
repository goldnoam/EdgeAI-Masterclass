
import React, { useState } from 'react';
import { GoogleGenAI } from "@google/genai";

const ObjectDetectionSandbox: React.FC = () => {
  const [image, setImage] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('Detect and describe all unique tools on the workbench.');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const runDetection = async () => {
    if (!image) return;
    setLoading(true);
    setResult('');

    try {
      // Create fresh instance right before call using process.env.API_KEY directly
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const base64Data = image.split(',')[1];
      
      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: {
          parts: [
            { text: `You are a high-performance computer vision assistant. ${prompt}. Provide bounding box coordinates if possible in [ymin, xmin, ymax, xmax] format (0-1000 scale) and identify what makes the objects unique.` },
            { inlineData: { mimeType: 'image/jpeg', data: base64Data } }
          ]
        },
      });

      // Directly access .text property from response
      setResult(response.text || 'No response from model.');
    } catch (err) {
      console.error(err);
      setResult('Error running visual analysis.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 w-full max-w-5xl mx-auto py-6">
      <div className="p-8 lg:p-12 rounded-[2.5rem] bg-slate-900 border border-slate-800 shadow-2xl mb-8">
        <h2 className="text-3xl font-black text-white mb-2 tracking-tight">Unique Object Prompting</h2>
        <p className="text-slate-400 mb-8">Demonstrating Gemini-3-Flash's native ability to detect and reason about specific objects in a scene.</p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="space-y-6">
            <div 
              className="aspect-video w-full rounded-2xl border-2 border-dashed border-slate-700 bg-slate-800/30 flex flex-col items-center justify-center relative overflow-hidden group cursor-pointer hover:border-indigo-500 transition-colors"
              onClick={() => document.getElementById('file-upload')?.click()}
            >
              {image ? (
                <img src={image} className="w-full h-full object-cover" alt="Preview" />
              ) : (
                <div className="text-center p-6">
                  <span className="text-4xl mb-4 block">📸</span>
                  <p className="text-sm font-bold text-slate-400">Click to upload target image</p>
                </div>
              )}
              <input id="file-upload" type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
            </div>

            <div>
              <label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Detection Prompt</label>
              <textarea 
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 text-slate-200 h-24"
                placeholder="What should the model find?"
              />
            </div>

            <button 
              onClick={runDetection}
              disabled={loading || !image}
              className={`w-full py-4 rounded-xl font-black uppercase text-xs tracking-widest transition-all ${
                loading || !image ? 'bg-slate-800 text-slate-600' : 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20 hover:bg-indigo-500 active:scale-95'
              }`}
            >
              {loading ? 'Analyzing Scene...' : 'Run Visual Analysis'}
            </button>
          </div>

          <div className="flex flex-col">
            <label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Analysis Result</label>
            <div className="flex-1 bg-slate-950 rounded-2xl border border-slate-800 p-6 overflow-y-auto max-h-[400px] lg:max-h-none custom-scrollbar">
              {loading ? (
                <div className="space-y-3">
                  <div className="h-4 bg-slate-800 rounded w-3/4 animate-pulse" />
                  <div className="h-4 bg-slate-800 rounded w-1/2 animate-pulse" />
                  <div className="h-4 bg-slate-800 rounded w-5/6 animate-pulse" />
                </div>
              ) : result ? (
                <div className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap font-mono">
                  {result}
                </div>
              ) : (
                <div className="text-center py-20 opacity-30 text-sm italic">
                  Results will appear here after analysis.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ObjectDetectionSandbox;
