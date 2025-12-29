
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
      "Utilize 'Channels Last' memory format for up to 20% performance boost on vision models.",
      "Enable CUDA Graphs for static network shapes to reduce CPU launch overhead."
    ],
    trainingGuide: {
      description: "Standard training workflow using DistributedDataParallel (DDP) for multi-GPU scalability.",
      code: "import torch.nn as nn\nimport torch.distributed as dist\nfrom torch.nn.parallel import DistributedDataParallel as DDP\n\n# Wrap model for DDP\nddp_model = DDP(model, device_ids=[rank])\n\n# Training Loop\noptimizer.zero_grad()\noutputs = ddp_model(inputs)\nloss = criterion(outputs, labels)\nloss.backward()\noptimizer.step()"
    },
    examples: [
      {
        title: 'Model Serialization & Loading',
        description: 'Exporting a scripted model from Python and loading it in C++.',
        versions: [
          {
            label: 'v2.x (Latest)',
            python: `import torch\nimport torchvision\n\nmodel = torchvision.models.resnet18(weights="DEFAULT")\nmodel.eval()\nexample = torch.rand(1, 3, 224, 224)\ntraced_script_module = torch.jit.trace(model, example)\n\n# Save for C++\ntraced_script_module.save("model.pt")`,
            cpp: `#include <torch/script.h>\n#include <iostream>\n\nint main() {\n  torch::jit::script::Module module;\n  try {\n    module = torch::jit::load("model.pt");\n  } catch (const c10::Error& e) {\n    std::cerr << "error loading model\\n";\n    return -1;\n  }\n  auto input = torch::ones({1, 3, 224, 224});\n  at::Tensor output = module.forward({input}).toTensor();\n  std::cout << output.slice(1, 0, 5) << std::endl;\n}`
          },
          {
            label: 'v1.13 (Stable)',
            python: `import torch\nimport torchvision\n\nmodel = torchvision.models.resnet18(pretrained=True)\nmodel.eval()\n# ... standard trace`,
            cpp: `// v1.13 loader\n#include <torch/script.h>\n// Same loading logic but uses older pretrained naming conventions`
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
    description: 'Open Neural Network Exchange provides an interoperable format for high-performance inference across different hardwares.',
    pythonInstall: 'pip install onnxruntime # for CPU\npip install onnxruntime-gpu # for GPU support',
    cppInstall: '1. Download ONNX Runtime binaries from GitHub Releases or use vcpkg:\n   vcpkg install onnxruntime\n2. Include onnxruntime_cxx_api.h\n3. Link against onnxruntime.lib/so',
    optimizationTips: [
      "Select the optimal Execution Provider (CUDA, TensorRT, OpenVINO, CoreML) for your hardware.",
      "Set Graph Optimization Level to ORT_ENABLE_ALL for fused kernels and constant folding.",
      "Enable intra-op threading and set thread affinity for CPU-bound workloads."
    ],
    trainingGuide: {
      description: "While primarily for inference, ONNX Runtime Training (ORT Training) speeds up large model fine-tuning.",
      code: "from onnxruntime.training import ORTTrainer\n# Convert PyTorch model to ORT backend\nmodel = ORTModule(pytorch_model)"
    },
    examples: [
      {
        title: 'Inference Session Setup',
        description: 'Comprehensive C++ example for loading a model, preparing input, and processing results.',
        versions: [
          {
            label: 'v1.17+ (Latest)',
            python: `# STEP 1: Export from PyTorch\nimport torch\ntorch.onnx.export(model, dummy_input, "model.onnx", \n                  input_names=['input'], output_names=['output'],\n                  dynamic_axes={'input': {0: 'batch_size'}})`,
            cpp: `#include <onnxruntime_cxx_api.h>\n#include <vector>\n#include <iostream>\n\nint main() {\n    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Inference");\n    Ort::SessionOptions session_options;\n    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);\n    \n    const char* model_path = "model.onnx";\n    Ort::Session session(env, model_path, session_options);\n\n    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);\n    std::vector<float> input_tensor_values(1 * 3 * 224 * 224, 1.0f);\n    std::vector<int64_t> input_node_dims = {1, 3, 224, 224};\n    \n    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(\n        memory_info, input_tensor_values.data(), input_tensor_values.size(), \n        input_node_dims.data(), input_node_dims.size());\n\n    const char* input_names[] = {"input"};\n    const char* output_names[] = {"output"};\n    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);\n\n    float* results = output_tensors.front().GetTensorMutableData<float>();\n    std::cout << "Top prediction value: " << results[0] << std::endl;\n\n    return 0;\n}`
          }
        ]
      }
    ]
  },
  {
    id: 'tensorflow',
    name: 'TensorFlow / TF-Lite',
    icon: '🔶',
    color: 'from-yellow-500 to-orange-600',
    description: 'Google\'s production-focused ecosystem with extensive tools for mobile and embedded deployment.',
    pythonInstall: 'pip install tensorflow',
    cppInstall: '1. For TF-Lite: Download pre-built libraries or use flatbuffers/vcpkg.\n2. Include tensorflow/lite/interpreter.h',
    optimizationTips: [
      "Use XLA (Accelerated Linear Algebra) by setting jit_compile=True in @tf.function.",
      "Optimize data pipelines using tf.data.AUTOTUNE and prefetch()."
    ],
    trainingGuide: {
      description: "Focus on efficient data ingestion and Model checkpointing.",
      code: "model.compile(optimizer='adam', loss='mse')\nmodel.fit(dataset, epochs=10)"
    },
    examples: [
      {
        title: 'TF-Lite Quantization',
        description: 'Reducing model size for mobile devices.',
        versions: [
          {
            label: 'Latest',
            python: `import tensorflow as tf\nconverter = tf.lite.TFLiteConverter.from_saved_model("model")\nconverter.optimizations = [tf.lite.Optimize.DEFAULT]\ntflite_model = converter.convert()`,
            cpp: `#include "tensorflow/lite/interpreter.h"\n// Load and run quantized .tflite`
          }
        ]
      }
    ]
  },
  {
    id: 'tensorrt',
    name: 'NVIDIA TensorRT',
    icon: '⚡',
    color: 'from-green-500 to-emerald-600',
    description: 'The gold standard for high-performance inference on NVIDIA GPUs, offering FP16/INT8 quantization.',
    pythonInstall: 'pip install tensorrt',
    cppInstall: '1. Install via NVIDIA Local Repo (.deb/.rpm) or .tar.gz\n2. Include NvInfer.h and link with -lnvinfer',
    optimizationTips: [
      "Use FP16 or INT8 precision mode for significant speedup.",
      "Profile your engine using the 'trtexec' CLI tool."
    ],
    trainingGuide: {
      description: "Inference-only; perform QAT in PyTorch/TF before export.",
      code: "from pytorch_quantization import nn as qnn"
    },
    examples: [
      {
        title: 'Engine Building',
        description: 'Building an optimized CUDA engine from ONNX.',
        versions: [
          {
            label: 'v8.x',
            python: `import tensorrt as trt\n# TRT builder logic...`,
            cpp: `#include <NvInfer.h>\n// Parse ONNX and build TRT engine`
          }
        ]
      }
    ]
  },
  {
    id: 'ultralytics',
    name: 'Ultralytics (YOLO)',
    icon: '🎯',
    color: 'from-purple-500 to-pink-600',
    description: 'The world\'s leading object detection framework, famous for YOLOv8/v11 and ease of use.',
    pythonInstall: 'pip install ultralytics',
    cppInstall: '1. Export to ONNX or TensorRT using Python first.\n2. Use ONNX Runtime or TensorRT C++ APIs.',
    optimizationTips: [
      "Set imgsz to the smallest power of 32 that preserves your required accuracy.",
      "Export with half=True for FP16 inference on supported hardware.",
      "Use batch=N in prediction to utilize GPU parallelism on video streams."
    ],
    trainingGuide: {
      description: "Simplest training CLI in the industry. Highly customizable via YAML configs.",
      code: "from ultralytics import YOLO\nmodel = YOLO('yolo11n.pt')\nmodel.train(data='coco8.yaml', epochs=100, imgsz=640)"
    },
    examples: [
      {
        title: 'Real-time Object Detection',
        description: 'High performance detection boilerplate for various YOLO generations.',
        versions: [
          {
            label: 'YOLOv11 (New)',
            python: `from ultralytics import YOLO\nmodel = YOLO("yolo11n.pt")\nresults = model.predict("image.jpg", imgsz=640, half=True)`,
            cpp: `// Deployment via exported engine\n// 1. model.export(format='engine')\n// 2. Load with TensorRT C++ API`
          },
          {
            label: 'YOLOv8 (Legacy)',
            python: `from ultralytics import YOLO\nmodel = YOLO("yolov8n.pt")\n# Same API structure as v11`,
            cpp: `// Standard YOLOv8 ONNX Loader`
          }
        ]
      }
    ]
  },
  {
    id: 'roboflow',
    name: 'Roboflow',
    icon: '🏷️',
    color: 'from-cyan-500 to-blue-600',
    description: 'Streamline your computer vision pipeline from labeling to dataset management and deployment.',
    pythonInstall: 'pip install roboflow',
    cppInstall: '1. Install libcurl: sudo apt-get install libcurl4-openssl-dev\n2. Use a JSON library (e.g., nlohmann-json).',
    optimizationTips: [
      "Use 'Active Learning' to automatically upload low-confidence frames.",
      "Leverage 'Roboflow Inference' Docker container."
    ],
    trainingGuide: {
      description: "Upload data and label in the cloud.",
      code: "project.version(1).train()"
    },
    examples: [
      {
        title: 'API Inference',
        description: 'Hosted inference via REST.',
        versions: [
          {
            label: 'v1',
            python: `import requests\n# Post image to Roboflow API`,
            cpp: `#include <curl/curl.h>\n// Post image via libcurl`
          }
        ]
      }
    ]
  },
  {
    id: 'sam3',
    name: 'Segment Anything 3 (SAM 3)',
    icon: '✂️',
    color: 'from-pink-500 to-rose-600',
    description: 'The next generation of zero-shot segmentation. SAM 3 provides ultra-high resolution masks with minimal prompt latency.',
    pythonInstall: 'pip install segment-anything-3',
    cppInstall: '1. Export SAM 3 to ONNX with dynamic shapes.\n2. Use ONNX Runtime C++ API.',
    optimizationTips: [
      "Use TensorRT for the image encoder to achieve sub-20ms encoding on RTX 4090.",
      "Quantize the prompt encoder to INT8 as it is highly sensitive to latency."
    ],
    trainingGuide: {
      description: "Foundation model approach; usually involves fine-tuning the decoder.",
      code: "from sam3 import Sam3Predictor\npredictor = Sam3Predictor(checkpoint='sam3_h.pth')"
    },
    examples: [
      {
        title: 'ONNX Segmentation Inference',
        description: 'C++ boilerplate for loading SAM 3 components via ONNX Runtime and performing masked segmentation.',
        versions: [
          {
            label: 'v3.x (Latest)',
            python: `# STEP 1: Export SAM 3 Decoder to ONNX\nfrom sam3.utils.onnx import export_onnx_model\nexport_onnx_model(model, "sam3_decoder.onnx", opset=17)`,
            cpp: `#include <onnxruntime_cxx_api.h>\n#include <vector>\n#include <iostream>\n\nint main() {\n    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SAM3_App");\n    Ort::SessionOptions session_options;\n    \n    // Load SAM 3 Decoder ONNX (Assuming embeddings are pre-computed by encoder)\n    Ort::Session session(env, L"sam3_decoder.onnx", session_options);\n    Ort::AllocatorWithDefaultOptions allocator;\n\n    // SAM 3 inputs: image_embeddings (1, 256, 64, 64), point_coords (1, N, 2), point_labels (1, N)\n    std::vector<int64_t> embed_dims = {1, 256, 64, 64};\n    std::vector<float> embed_vals(1 * 256 * 64 * 64, 0.5f);\n    \n    std::vector<int64_t> point_dims = {1, 1, 2};\n    std::vector<float> point_coords = {500.0f, 375.0f}; // Target point\n\n    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);\n    \n    Ort::Value embed_tensor = Ort::Value::CreateTensor<float>(\n        memory_info, embed_vals.data(), embed_vals.size(), embed_dims.data(), embed_dims.size());\n    \n    Ort::Value point_tensor = Ort::Value::CreateTensor<float>(\n        memory_info, point_coords.data(), point_coords.size(), point_dims.data(), point_dims.size());\n\n    const char* input_names[] = {"image_embeddings", "point_coords"};\n    const char* output_names[] = {"masks", "iou_predictions"};\n    Ort::Value inputs[] = {std::move(embed_tensor), std::move(point_tensor)};\n\n    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, inputs, 2, output_names, 2);\n    \n    // Extract mask data (typically 1, 3, H, W for multi-mask outputs)\n    float* mask_data = outputs[0].GetTensorMutableData<float>();\n    std::cout << "Mask generated. Confidence score: " << outputs[1].GetTensorData<float>()[0] << std::endl;\n\n    return 0;\n}`
          }
        ]
      }
    ]
  },
  {
    id: 'golang',
    name: 'Go (Golang) for AI',
    icon: '🐹',
    color: 'from-cyan-400 to-blue-500',
    description: 'Superior concurrency and deployment simplicity for ML inference servers.',
    pythonInstall: 'n/a',
    cppInstall: 'CGO used for native linking.',
    goInstall: 'go get github.com/yalue/onnxruntime_go',
    optimizationTips: [
      "Use CGO sparingly to minimize transition overhead.",
      "Leverage Go routines for parallel pre-processing."
    ],
    trainingGuide: {
      description: "Training via wrappers like Gotorch.",
      code: "import \"github.com/wangkuiyi/gotorch\""
    },
    examples: [
      {
        title: 'Native ONNX Execution',
        description: 'Using Go to host highly concurrent inference servers.',
        versions: [
          {
            label: 'v1.18+',
            python: `# Prepare model\nmodel.export(format='onnx')`,
            cpp: `// Standard C++ ONNX logic`,
            go: `import "github.com/yalue/onnxruntime_go"\n// ... init & run`
          }
        ]
      }
    ]
  }
];
