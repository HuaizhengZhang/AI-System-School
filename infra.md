# Machine Learning Infrastructure

Frameworks, infra and useful toolkits (e.g., visulization) for training, inference or both. You can check [[AI infrastructures list]](https://github.com/1duo/awesome-ai-infrastructures) for more.

- [Paper](#paper)
- [Tool](#userful-tools)
- [Project with code](#project)

## Paper

- The Case for Learning-and-System Co-design [[Paper]](https://dl.acm.org/citation.cfm?id=3352031)
  - Mike Liang, C.J., Xue, H., Yang, M. and Zhou, L., 2019. 
  - ACM SIGOPS Operating Systems Review, 53(1), pp.68-74.
  - Summary: Make the system learnable. Propose a framework named AutoSys which contains both training plane and inference plane


## Project

- Jittor is a high-performance deep learning framework based on JIT compiling and meta-operators.[[GitHub]](https://github.com/Jittor/jittor)
- MindSpore is a new open source deep learning training/inference framework that could be used for mobile, edge and cloud scenarios.[[GitHub]](https://github.com/mindspore-ai/mindspore)
- MegEngine is a fast, scalable and easy-to-use numerical evaluation framework, with auto-differentiation.[[GitHub]](https://github.com/MegEngine/MegEngine)
- cortexlabs/cortex: Deploy machine learning applications without worrying about setting up infrastructure, managing dependencies, or orchestrating data pipelines. [[GitHub]](https://github.com/cortexlabs/cortex)
- Osquery is a SQL powered operating system instrumentation, monitoring, and analytics framework. [[Facebook Project]](https://osquery.io/)
- Kubeflow: Kubeflow is a machine learning (ML) toolkit that is dedicated to making deployments of ML workflows on Kubernetes simple, portable, and scalable. [[GitHub]](https://github.com/kubeflow/pipelines)
- Polytaxon: A platform for reproducible and scalable machine learning and deep learning on kubernetes. [[GitHub]](https://github.com/polyaxon/polyaxon)
- MLOps on Azure [[GitHub]](https://github.com/microsoft/MLOps)
- Flame: An ML framework to accelerate research and its path to production. [[GitHub]](https://github.com/Open-ASAPP/flambe)
- Ludwig is a toolbox built on top of TensorFlow that allows to train and test deep learning models without the need to write code. [[GitHub]](https://github.com/uber/ludwig)
- intel-analytics/analytics-zoo Distributed Tensorflow, Keras and BigDL on Apache Spark [[GitHub]](https://github.com/intel-analytics/analytics-zoo)
- Machine Learning for .NET [[GitHub]](https://github.com/dotnet/machinelearning)
  - ML.NET is a cross-platform open-source machine learning framework which makes machine learning accessible to .NET developers.
  - ML.NET allows .NET developers to develop their own models and infuse custom machine learning into their applications, using .NET, even without prior expertise in developing or tuning machine learning models.
- ONNX: Open Neural Network Exchange [[GitHub]](https://github.com/onnx/onnx)
- ONNXRuntime: has an open architecture that is continually evolving to address the newest developments and challenges in AI and Deep Learning. ONNX Runtime stays up to date with the ONNX standard, supporting all ONNX releases with future compatibility and maintaining backwards compatibility with prior releases. [[GitHub]](https://github.com/microsoft/onnxruntime)
- BentoML: Machine Learning Toolkit for packaging and deploying models [[GitHub]](https://github.com/bentoml/BentoML)
- EuclidesDB: A multi-model machine learning feature embedding database [[GitHub]](https://github.com/perone/euclidesdb)
- Prefect: Perfect is a new workflow management system, designed for modern infrastructure and powered by the open-source Prefect Core workflow engine. [[GitHub]](https://github.com/PrefectHQ/prefect)
- MindsDB: MindsDB's goal is to make it very simple for developers to use the power of artificial neural networks in their projects [[GitHub]](https://github.com/mindsdb/mindsdb)
- PAI: OpenPAI is an open source platform that provides complete AI model training and resource management capabilities. [[Microsoft Project]](https://github.com/Microsoft/pai#resources)
- Bistro: Scheduling Data-Parallel Jobs Against Live Production Systems [[Facebook Project]](https://github.com/facebook/bistro)
- GNES is Generic Neural Elastic Search, a cloud-native semantic search system based on deep neural network. [[GitHub]](https://github.com/gnes-ai/gnes)


## GPU Sharing

- Yu, P. and Chowdhury, M., 2019. Salus: Fine-Grained GPU Sharing Primitives for Deep Learning Applications. arXiv preprint arXiv:1902.04610. [[Paper]](https://arxiv.org/pdf/1902.04610.pdf) [[GitHub]](https://github.com/SymbioticLab/Salus)
- gpushare-scheduler-extender [[GitHub]](https://github.com/HuaizhengZhang/gpushare-scheduler-extender)
  - More and more data scientists run their Nvidia GPU based inference tasks on Kubernetes. Some of these tasks can be run on the same Nvidia GPU device to increase GPU utilization. So one important challenge is how to share GPUs between the pods
  
  
## Userful Tools

#### Profile
- Collective Knowledge repository to automate MLPerf - a broad ML benchmark suite for measuring performance of ML software frameworks, ML hardware accelerators, and ML cloud platforms [[GitHub]](https://github.com/ctuning/ck-mlperf)
- NetworKit is a growing open-source toolkit for large-scale network analysis. [[GitHub]](https://github.com/kit-parco/networkit)
- gpu-sentry: Flask-based package for monitoring utilisation of nVidia GPUs. [[GitHub]](https://github.com/jacenkow/gpu-sentry)
- anderskm/gputil: A Python module for getting the GPU status from NVIDA GPUs using nvidia-smi programmically in Python [[GitHub]](https://github.com/anderskm/gputil)
- Pytorch-Memory-Utils: detect your GPU memory during training with Pytorch. [[GitHub]](https://github.com/Oldpan/Pytorch-Memory-Utils)
- torchstat: a lightweight neural network analyzer based on PyTorch. [[GitHub]](https://github.com/Swall0w/torchstat)
- NVIDIA GPU Monitoring Tools [[GitHub]](https://github.com/NVIDIA/gpu-monitoring-tools)
- PyTorch/cpuinfo: cpuinfo is a library to detect essential for performance optimization information about host CPU. [[GitHub]](https://github.com/pytorch/cpuinfo)
- Popular Network memory consumption and FLOP counts [[GitHub]](https://github.com/albanie/convnet-burden)
- Intel® VTune™ Amplifier [[Website]](https://software.intel.com/en-us/vtune)
  - Stop guessing why software is slow. Advanced sampling and profiling techniques quickly analyze your code, isolate issues, and deliver insights for optimizing performance on modern processors
- Pyflame: A Ptracing Profiler For Python [[GitHub]](https://github.com/uber/pyflame)

#### Others
- Facebook AI Performance Evaluation Platform [[GitHub]](https://github.com/facebook/FAI-PEP)
- Netron: Visualizer for deep learning and machine learning models [[GitHub]](https://github.com/lutzroeder/netron)
- Facebook/FBGEMM: FBGEMM (Facebook GEneral Matrix Multiplication) is a low-precision, high-performance matrix-matrix multiplications and convolution library for server-side inference. [[GitHub]](https://github.com/pytorch/FBGEMM)
- Dslabs: Distributed Systems Labs and Framework for UW system course [[GitHub]](https://github.com/emichael/dslabs)
- Machine Learning Model Zoo [[Website]](https://modelzoo.co/)
- Faiss: A library for efficient similarity search and clustering of dense vectors [[GitHub]](https://github.com/facebookresearch/faiss)
- Microsoft/MMdnn: A comprehensive, cross-framework solution to convert, visualize and diagnose deep neural network models.[[GitHub]](https://github.com/Microsoft/MMdnn)
- Example recipes for Kubernetes Network Policies that you can just copy paste [[GitHub]](https://github.com/ahmetb/kubernetes-network-policy-recipes)
