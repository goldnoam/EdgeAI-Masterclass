
export type Framework = 'pytorch' | 'onnx' | 'tensorflow' | 'tensorrt' | 'roboflow' | 'ultralytics' | 'sam3' | 'golang';

export interface VersionedSnippet {
  label: string;
  python: string;
  cpp: string;
  go?: string;
}

export interface CodeExample {
  title: string;
  description: string;
  versions: VersionedSnippet[];
}

export interface FrameworkData {
  id: Framework;
  name: string;
  icon: string;
  description: string;
  color: string;
  examples: CodeExample[];
  pythonInstall: string;
  cppInstall: string;
  goInstall?: string;
  optimizationTips: string[];
  trainingGuide: {
    description: string;
    code: string;
  };
}

export interface ComparisonFeature {
  name: string;
  pytorch: string | number;
  onnx: string | number;
  tensorflow: string | number;
  tensorrt: string | number;
  ultralytics: string | number;
  roboflow: string | number;
  sam3: string | number;
  golang: string | number;
}

export interface BenchmarkData {
  metric: string;
  unit: string;
  pytorch: number;
  onnx: number;
  tensorflow: number;
  tensorrt: number;
  ultralytics: number;
  roboflow: number;
  sam3: number;
  golang: number;
}

export interface WizardStep {
  id: string;
  question: string;
  options: {
    id: string;
    label: string;
    description: string;
    icon: string;
  }[];
}

// Added missing ChatMessage interface to resolve compilation error in GeminiExpert.tsx
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}
