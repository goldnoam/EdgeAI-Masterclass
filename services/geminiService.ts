
import { GoogleGenAI } from "@google/genai";

// Initialize GoogleGenAI with API key directly from environment as per @google/genai guidelines
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export const getAIAssistance = async (prompt: string, framework: string) => {
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: prompt,
      config: {
        systemInstruction: `You are an expert in Deep Learning deployment and high-performance computing. 
        You specialize in ${framework}, C++, and Python. Provide concise, production-ready code snippets 
        focused on optimization (TensorRT, ONNX Runtime, LibTorch, OpenVINO). 
        Always include both Python and C++ counterparts if possible.`,
        temperature: 0.7,
      },
    });
    return response.text;
  } catch (error) {
    console.error("Gemini Error:", error);
    return "Sorry, I encountered an error while generating code. Please try again.";
  }
};
