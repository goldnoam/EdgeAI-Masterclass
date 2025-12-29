import { FrameworkData, ComparisonFeature, BenchmarkData, WizardStep } from './types';

export const COMPARISON_FEATURES: ComparisonFeature[] = [
  {
    name: "Primary Strength",
    pytorch: "Research & Flexibility",
    onnx: "Interoperability",
    tensorflow: "Mobile & Production",
    tensorrt: "Maximum GPU Speed",
    ultralytics: "Vision Ease-of-use",
    roboflow: "Data Pipeline",
    sam3: "Zero-shot Seg",
    golang: "Cloud Concurrency"
  },
  {
    name: "C++ API Quality",
    pytorch: "Excellent",
    onnx: "Good",
    tensorflow: "Moderate",
    tensorrt: "Professional",
    ultralytics: "Via ONNX/TRT",
    roboflow: "REST",
    sam3: "Via ONNX",
    golang: "via CGO"
  },
  {
    name: "Learning Curve",
    pytorch: "Medium",
    onnx: "Medium",
    tensorflow: "High",
    tensorrt: "High",
    ultralytics: "Very Low",
    roboflow: "Very Low",
    sam3: "Low",
    golang: "Medium"
  }
];

export const BENCHMARK_METRICS: BenchmarkData[] = [
  {
    metric: "Inference Latency (ResNet-50)",
    unit: "ms",
    pytorch: 45,
    onnx: 18,
    tensorflow: 25,
    tensorrt: 4,
    ultralytics: 12,
    roboflow: 150,
    sam3: 28,
    golang: 22
  }
];

export const TRAINING_BENCHMARKS: BenchmarkData[] = [
  {
    metric: "Training Speed (ImageNet)",
    unit: "epochs/sec",
    pytorch: 0.045,
    onnx: 0.032,
    tensorflow: 0.041,
    tensorrt: 0,
    ultralytics: 0.058,
    roboflow: 0.012,
    sam3: 0.015,
    golang: 0.008
  },
  {
    metric: "Training Throughput",
    unit: "img/sec",
    pytorch: 425,
    onnx: 390,
    tensorflow: 410,
    tensorrt: 0,
    ultralytics: 480,
    roboflow: 110,
    sam3: 140,
    golang: 95
  },
  {
    metric: "VRAM Efficiency",
    unit: "GB",
    pytorch: 6.2,
    onnx: 7.4,
    tensorflow: 8.1,
    tensorrt: 0,
    ultralytics: 4.2,
    roboflow: 9.5,
    sam3: 11.2,
    golang: 8.5
  }
];

export const WIZARD_STEPS: WizardStep[] = [
  {
    id: 'hardware',
    question: "What is your target hardware?",
    options: [
      { id: 'nvidia', label: 'NVIDIA GPU', description: 'Maximum performance via CUDA/TensorRT', icon: '⚡' },
      { id: 'mobile', label: 'Mobile/Edge', description: 'Deployment on Android, iOS, or ARM', icon: '📱' },
      { id: 'cpu', label: 'Intel/AMD CPU', description: 'Standard server or desktop inference', icon: '💻' },
      { id: 'web', label: 'Web Browser', description: 'Running models directly in the browser', icon: '🌐' }
    ]
  },
  {
    id: 'priority',
    question: "What is your top priority?",
    options: [
      { id: 'latency', label: 'Ultra Low Latency', description: 'Real-time performance is critical', icon: '⏱️' },
      { id: 'flexibility', label: 'Deployment Ease', description: 'Quick to set up and maintain', icon: '🛠️' },
      { id: 'size', label: 'Model Size', description: 'Small binary/model footprint', icon: '📦' }
    ]
  }
];

export const FRAMEWORKS: FrameworkData[] = [
  {
    id: 'pytorch',
    name: 'PyTorch / LibTorch',
    icon: '🔥',
    color: 'from-orange-500 to-red-600',
    description: 'The preferred framework for research and dynamic graph development, with robust C++ support via LibTorch.',
    pythonInstall: 'pip install torch torchvision torchaudio',
    cppInstall: '1. Download LibTorch (Pre-built) from pytorch.org\n2. Unzip to path/to/libtorch\n3. Use find_package(Torch REQUIRED) in CMakeLists.txt\n4. Set -DCMAKE_PREFIX_PATH="path/to/libtorch" during build.',
    optimizationTips: [
      "Use Auto Mixed Precision (AMP) to speed up training and inference by 2-3x on modern GPUs.",
      "Convert models to TorchScript (Tracing/Scripting) to bypass the Python Global Interpreter Lock (GIL).",
      "Utilize 'Channels Last' memory format for up to 20% performance boost on vision models."
    ],
    trainingGuide: {
      description: "Standard training workflow using DistributedDataParallel (DDP).",
      code: "import torch.nn as nn\nfrom torch.nn.parallel import DistributedDataParallel as DDP\nddp_model = DDP(model, device_ids=[rank])"
    },
    examples: [
      {
        title: 'Model Loading in C++',
        description: 'Loading a scripted model from LibTorch.',
        versions: [
          {
            label: 'v2.5',
            python: `import torch\nmodel = torch.jit.script(MyModel())\nmodel.save("model.pt")`,
            cpp: `#include <torch/script.h>\n#include <iostream>\n\nint main() {\n  torch::jit::script::Module module;\n  try {\n    module = torch::jit::load("model.pt");\n  } catch (const c10::Error& e) {\n    return -1;\n  }\n  std::cout << "Model loaded!\\n";\n  return 0;\n}`
          }
        ]
      }
    ]
  },
  {
    id: 'onnx',
    name: 'ONNX Runtime',
    icon: '💠',
    color: 'from-blue-500 to-indigo-600',
    description: 'Universal inference engine for high-performance cross-platform deployment.',
    pythonInstall: 'pip install onnxruntime-gpu',
    cppInstall: 'Download binaries from GitHub or use vcpkg install onnxruntime',
    optimizationTips: ["Use CUDA/TensorRT Execution Providers", "Enable Graph Optimizations"],
    trainingGuide: { description: "N/A", code: "" },
    examples: [
      {
        title: 'ORT Session Init',
        description: 'Initializing a C++ inference session.',
        versions: [{
          label: 'v1.19',
          python: `import onnxruntime as ort\nsession = ort.InferenceSession("model.onnx")`,
          cpp: `#include <onnxruntime_cxx_api.h>\n\nint main() {\n  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");\n  Ort::Session session(env, L"model.onnx", Ort::SessionOptions{nullptr});\n  return 0;\n}`
        }]
      }
    ]
  },
  {
    id: 'tensorflow',
    name: 'TensorFlow',
    icon: '🔶',
    color: 'from-yellow-500 to-orange-600',
    description: 'Production-grade ecosystem for ML deployment.',
    pythonInstall: 'pip install tensorflow',
    cppInstall: 'Use Bazel to build LibTensorFlow or use TF-Lite headers.',
    optimizationTips: ["XLA compilation", "Mixed precision"],
    trainingGuide: { description: "Keras workflow", code: "model.fit(data)" },
    examples: []
  },
  {
    id: 'tensorrt',
    name: 'TensorRT',
    icon: '⚡',
    color: 'from-green-500 to-emerald-600',
    description: 'NVIDIA specialized inference SDK.',
    pythonInstall: 'pip install tensorrt',
    cppInstall: 'Install via CUDA toolkit/Local repo.',
    optimizationTips: ["INT8 Quantization", "Layer fusion"],
    trainingGuide: { description: "N/A", code: "" },
    examples: []
  },
  {
    id: 'ultralytics',
    name: 'Ultralytics YOLO',
    icon: '🎯',
    color: 'from-purple-500 to-pink-600',
    description: 'Leading vision AI framework.',
    pythonInstall: 'pip install ultralytics',
    cppInstall: 'Export to ONNX/TensorRT for C++.',
    optimizationTips: ["FP16 export", "Dynamic batching"],
    trainingGuide: { description: "YOLO training", code: "model.train(data='coco.yaml')" },
    examples: []
  },
  {
    id: 'roboflow',
    name: 'Roboflow',
    icon: '🏷️',
    color: 'from-cyan-500 to-blue-600',
    description: 'Dataset management and auto-labeling.',
    pythonInstall: 'pip install roboflow',
    cppInstall: 'REST API via libcurl.',
    optimizationTips: ["Active learning", "Preprocessing"],
    trainingGuide: { description: "AutoML", code: "project.train()" },
    examples: []
  },
  {
    id: 'sam3',
    name: 'SAM 3',
    icon: '✂️',
    color: 'from-rose-500 to-red-600',
    description: 'Segment Anything Model v3.',
    pythonInstall: 'pip install segment-anything-3',
    cppInstall: 'ONNX Runtime integration.',
    optimizationTips: ["Quantized prompt encoder"],
    trainingGuide: { description: "Foundation model tuning", code: "" },
    examples: []
  },
  {
    id: 'golang',
    name: 'Golang / ONNX',
    icon: '🐹',
    color: 'from-cyan-400 to-blue-500',
    description: 'High-concurrency inference servers with Go.',
    goInstall: 'go get github.com/yalue/onnxruntime_go',
    pythonInstall: 'pip install onnx',
    cppInstall: 'CGO linking to onnxruntime.so',
    optimizationTips: [
      "Use shared thread-safe sessions across goroutines",
      "Minimize CGO boundaries to reduce context-switching overhead",
      "Leverage Go channels for non-blocking pre/post processing pipelines"
    ],
    trainingGuide: {
      description: "Fine-tuning via Gotorch (LibTorch wrappers).",
      code: "import \"github.com/wangkuiyi/gotorch\"\nmodel := MyNetwork()\nmodel.Train()"
    },
    examples: [
      {
        title: 'Native ONNX Inference in Go',
        description: 'Initializing the environment and running basic inference in Go using the yalue/onnxruntime_go wrapper.',
        versions: [
          {
            label: 'v1.18+',
            python: `import onnxruntime as ort\nsession = ort.InferenceSession("model.onnx")\nres = session.run(None, {"input": data})`,
            cpp: `Ort::Env env;\nOrt::Session session(env, L"model.onnx", ...);`,
            go: `package main\n\nimport (\n\t"fmt"\n\tort "github.com/yalue/onnxruntime_go"\n)\n\nfunc main() {\n\t// 1. Point to your shared library path\n\tort.SetSharedLibraryPath("libonnxruntime.so")\n\tort.Initialize()\n\tdefer ort.Destroy()\n\n\t// 2. Prepare tensors\n\tinputShape := ort.NewShape(1, 3, 224, 224)\n\tinputData := make([]float32, 1*3*224*224)\n\tinputTensor, _ := ort.NewTensor(inputShape, inputData)\n\tdefer inputTensor.Destroy()\n\n\t// 3. Load and execute\n\tsession, _ := ort.NewAdvancedSession("model.onnx",\n\t\t[]string{"input"}, []string{"output"},\n\t\t[]ort.ArbitraryTensor{inputTensor}, nil, nil)\n\tdefer session.Destroy()\n\n\tif err := session.Run(); err != nil {\n\t\tfmt.Printf("Error: %v\\n", err)\n\t}\n}`
          }
        ]
      },
      {
        title: 'Concurrent Inference Server',
        description: 'Building a scalable HTTP server that shares a single model across multiple goroutines using Go native networking.',
        versions: [
          {
            label: 'Production-Ready',
            python: `# Multi-processing required for Python throughput`,
            cpp: `#include <thread>\n// Manual thread pooling required`,
            go: `package main\n\nimport (\n\t"net/http"\n\t"sync"\n\tort "github.com/yalue/onnxruntime_go"\n)\n\ntype Predictor struct {\n\tsession *ort.AdvancedSession\n}\n\nfunc (p *Predictor) ServeHTTP(w http.ResponseWriter, r *http.Request) {\n\t// ONNX Runtime sessions are thread-safe for the Run() method.\n\t// net/http automatically handles requests in parallel goroutines.\n\terr := p.session.Run()\n\tif err != nil {\n\t\thttp.Error(w, "Inference failed", 500)\n\t\treturn\n\t}\n\tw.Write([]byte("Inference Success"))\n}\n\nfunc main() {\n\tort.Initialize()\n\tdefer ort.Destroy()\n\n\t// Initialize global session once\n\tsharedSession, _ := initSession("resnet50.onnx")\n\n\tp := &Predictor{session: sharedSession}\n\thttp.ListenAndServe(":8080", p)\n}`
          }
        ]
      }
    ]
  }
];
