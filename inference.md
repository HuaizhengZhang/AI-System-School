# Inference System

System for machine learning inference.

## Benchmark
- Reddi, Vijay Janapa, et al. "Mlperf inference benchmark." arXiv preprint arXiv:1911.02549 (2019). [[Paper]](https://arxiv.org/pdf/1911.02549.pdf) [[GitHub]](https://github.com/mlperf/inference)
- Bianco, Simone, et al. "Benchmark analysis of representative deep neural network architectures." IEEE Access 6 (2018): 64270-64277. [[Paper]](https://arxiv.org/abs/1810.00736)
- Almeida, Mario, et al. "EmBench: Quantifying Performance Variations of Deep Neural Networks across Modern Commodity Devices." The 3rd International Workshop on Deep Learning for Mobile Systems and Applications. 2019. [[Paper]](https://arxiv.org/pdf/1905.07346.pdf)

## Model Zoo (Experiment Version Control)

- TRAINS - Auto-Magical Experiment Manager & Version Control for AI [[GitHub]](https://github.com/allegroai/trains)
- ModelDB: A system to manage ML models [[GitHub]](https://github.com/mitdbg/modeldb) [[MIT short paper]](https://mitdbg.github.io/modeldb/papers/hilda_modeldb.pdf)
- iterative/dvc: Data & models versioning for ML projects, make them shareable and reproducible [[GitHub]](https://github.com/iterative/dvc)

## Model Serving

- Seldon Core: Blazing Fast, Industry-Ready ML. An open source platform to deploy your machine learning models on Kubernetes at massive scale. [[GitHub]](https://github.com/SeldonIO/seldon-core)
- MArk: Exploiting Cloud Services for Cost-Effective, SLO-Aware Machine Learning Inference Serving [[Paper]](https://www.usenix.org/system/files/atc19-zhang-chengliang.pdf) [[GitHub]](https://github.com/marcoszh/MArk-Project)
  - Zhang, C., Yu, M., Wang, W. and Yan, F., 2019. 
  - In 2019 {USENIX} Annual Technical Conference ({USENIX}{ATC} 19) (pp. 1049-1062).
  - Summary: address the scalability and cost minimization issues for model serving on the public cloud.
  
- Willump: A Statistically-Aware End-to-end Optimizer for Machine Learning Inference. [[arxiv]](https://arxiv.org/pdf/1906.01974.pdf)[[GitHub]](https://github.com/stanford-futuredata/Willump)
  - Peter Kraft, Daniel Kang, Deepak Narayanan, Shoumik Palkar, Peter Bailis, Matei Zaharia.
  - arXiv Preprint. 2019.
- Parity Models: Erasure-Coded Resilience for Prediction Serving Systems(SOSP2019) [[Paper]](http://www.cs.cmu.edu/~rvinayak/papers/sosp2019parity-models.pdf) [[GitHub]](https://github.com/Thesys-lab/parity-models)
- INFaaS: A Model-less Inference Serving System Romero [[Paper]](https://arxiv.org/abs/1905.13348) [[GitHub]](https://github.com/stanford-mast/INFaaS)
  - F., Li, Q., Yadwadkar, N.J. and Kozyrakis, C., 2019.
  - arXiv preprint arXiv:1905.13348.
- Nexus: Nexus is a scalable and efficient serving system for DNN applications on GPU cluster (SOSP2019) [[Paper]](https://pdfs.semanticscholar.org/0c0f/353dbac84311ea4f1485d4a8ac0b0459be8c.pdf) [[GitHub]](https://github.com/uwsampl/nexus)
- Deep Learning Inference Service at Microsoft [[Paper]](https://www.usenix.org/system/files/opml19papers-soifer.pdf)
  - J Soifer, et al. (*OptML2019*)
- {PRETZEL}: Opening the Black Box of Machine Learning Prediction Serving Systems. [[Paper]](https://www.usenix.org/system/files/osdi18-lee.pdf)
  - Lee, Y., Scolari, A., Chun, B.G., Santambrogio, M.D., Weimer, M. and Interlandi, M., 2018. (*OSDI 2018*)
- Brusta: PyTorch model serving project [[GitHub]](https://github.com/hyoungseok/brusta)
- Model Server for Apache MXNet: Model Server for Apache MXNet is a tool for serving neural net models for inference [[GitHub]](https://github.com/awslabs/mxnet-model-server)
- TFX: A TensorFlow-Based Production-Scale Machine Learning Platform [[Paper]](http://stevenwhang.com/tfx_paper.pdf) [[Website]](https://www.tensorflow.org/tfx) [[GitHub]](https://github.com/tensorflow/tfx)
  - Baylor, Denis, et al. (*KDD 2017*)
- Tensorflow-serving: Flexible, high-performance ml serving [[Paper]](https://arxiv.org/pdf/1712.06139) [[GitHub]](https://github.com/tensorflow/serving)
  - Olston, Christopher, et al.
- IntelAI/OpenVINO-model-server: Inference model server implementation with gRPC interface, compatible with TensorFlow serving API and OpenVINO™ as the execution backend. [[GitHub]](https://github.com/IntelAI/OpenVINO-model-server)
- Clipper: A Low-Latency Online Prediction Serving System [[Paper]](https://www.usenix.org/system/files/conference/nsdi17/nsdi17-crankshaw.pdf)
[[GitHub]](https://github.com/ucbrise/clipper)
  - Crankshaw, Daniel, et al. (*NSDI 2017*)
  - Summary: Adaptive batch 
- InferLine: ML Inference Pipeline Composition Framework [[Paper]](https://arxiv.org/pdf/1812.01776.pdf) [[GitHub]](https://github.com/simon-mo/inferline-models)
  - Crankshaw, Daniel, et al. (*Preprint*)
  - Summary: update version of Clipper
- TrIMS: Transparent and Isolated Model Sharing for Low Latency Deep LearningInference in Function as a Service Environments [[Paper]](https://arxiv.org/pdf/1811.09732.pdf)
  - Dakkak, Abdul, et al (*Preprint*)
  - Summary: model cold start problem
- Rafiki: machine learning as an analytics service system [[Paper]](http://www.vldb.org/pvldb/vol12/p128-wang.pdf) [[GitHub]](https://github.com/nginyc/rafiki)
  - Wang, Wei, Jinyang Gao, Meihui Zhang, Sheng Wang, Gang Chen, Teck Khim Ng, Beng Chin Ooi, Jie Shao, and Moaz Reyad.
  - Summary: Contain both training and inference. Auto-Hype-Parameter search for training. Ensemble models for inference. Using DRL to balance trade-off between accuracy and latency.
- GraphPipe: Machine Learning Model Deployment Made Simple [[GitHub]](https://github.com/oracle/graphpipe)
- Orkhon: ML Inference Framework and Server Runtime [[GitHub]](https://github.com/vertexclique/orkhon)
- NVIDIA/tensorrt-inference-server: The TensorRT Inference Server provides a cloud inferencing solution optimized for NVIDIA GPUs. [[GitHub]](https://github.com/NVIDIA/tensorrt-inference-server)
- Apache PredictionIO® is an open source Machine Learning Server built on top of a state-of-the-art open source stack for developers and data scientists to create predictive engines for any machine learning task [[Website]](http://predictionio.apache.org/)

## Cache for Inference

- Kumar, Adarsh, et al. "Accelerating deep learning inference via freezing." 11th {USENIX} Workshop on Hot Topics in Cloud Computing (HotCloud 19). 2019. [[Paper]](http://shivaram.org/publications/freeze-hotcloud19.pdf)
- Xu, Mengwei, et al. "DeepCache: Principled cache for mobile deep vision." Proceedings of the 24th Annual International Conference on Mobile Computing and Networking. 2018. [[Paper]](https://arxiv.org/pdf/1712.01670.pdf)
- Park, Keunyoung, and Doo-Hyun Kim. "Accelerating image classification using feature map similarity in convolutional neural networks." Applied Sciences 9.1 (2019): 108. [[Paper]](https://www.mdpi.com/2076-3417/9/1/108/htm)
- Cavigelli, Lukas, and Luca Benini. "CBinfer: Exploiting frame-to-frame locality for faster convolutional network inference on video streams." IEEE Transactions on Circuits and Systems for Video Technology (2019). [[Paper]](https://arxiv.org/pdf/1808.05488)

## Inference Optimization

- TensorRT is a C++ library that facilitates high performance inference on NVIDIA GPUs and deep learning accelerators. [[GitHub]](https://github.com/NVIDIA/TensorRT)
- Dynamic Space-Time Scheduling for GPU Inference [[Paper]](http://learningsys.org/nips18/assets/papers/102CameraReadySubmissionGPU_Virtualization%20(8).pdf) [[GitHub]](https://github.com/ucbrise/caravel)
  - Jain, Paras, et al. (*NIPS 18, System for ML*)
  - Summary: optimization for GPU Multi-tenancy
- Dynamic Scheduling For Dynamic Control Flow in Deep Learning Systems [[Paper]](http://www.cs.cmu.edu/~jinlianw/papers/dynamic_scheduling_nips18_sysml.pdf)
  - Wei, Jinliang, Garth Gibson, Vijay Vasudevan, and Eric Xing. (*On going*)
- Accelerating Deep Learning Workloads through Efficient Multi-Model Execution. [[Paper]](https://cs.stanford.edu/~matei/papers/2018/mlsys_hivemind.pdf)
  - D. Narayanan, K. Santhanam, A. Phanishayee and M. Zaharia. (*NeurIPS Systems for ML Workshop 2018*)
  - Summary: They assume that their system, HiveMind, is given as input models grouped into model batches that are amenable to co-optimization and co-execution. a compiler, and a runtime.
- DeepCPU: Serving RNN-based Deep Learning Models 10x Faster [[Paper]](https://www.usenix.org/system/files/conference/atc18/atc18-zhang-minjia.pdf)
  - Minjia Zhang, Samyam Rajbhandari, Wenhan Wang, and Yuxiong He, Microsoft AI and Research (*ATC 2018*)

## Cluster Management for Inference (now only contain multi-tenant)

- Ease. ml: Towards multi-tenant resource sharing for machine learning workloads [[Paper]](http://www.vldb.org/pvldb/vol11/p607-li.pdf) [[GitHub]](https://github.com/DS3Lab/easeml) [[Demo]](http://www.vldb.org/pvldb/vol11/p2054-karlas.pdf)
  - Li, Tian, et al
  - Proceedings of the VLDB Endowment 11.5 (2018): 607-620.
- Perseus: Characterizing Performance and Cost of Multi-Tenant Serving for CNN Models [[Paper]](https://arxiv.org/pdf/1912.02322.pdf)
  - LeMay, Matthew, Shijian Li, and Tian Guo. 
  - arXiv preprint arXiv:1912.02322 (2019).
  
## Machine Learning Compiler
- {TVM}: An Automated End-to-End Optimizing Compiler for Deep Learning [[Paper]](https://www.usenix.org/system/files/osdi18-chen.pdf) [[YouTube]](https://www.youtube.com/watch?v=I1APhlSjVjs) [[Project Website]](https://tvm.ai/)
  - Chen, Tianqi, et al. (*OSDI 2018*)
  - Summary: Automated optimization is very impressive: cost model (rank objective function) + schedule explorer (parallel simulated annealing)
- Facebook TC: Tensor Comprehensions (TC) is a fully-functional C++ library to automatically synthesize high-performance machine learning kernels using Halide, ISL and NVRTC or LLVM. [[GitHub]](https://github.com/facebookresearch/TensorComprehensions)
- Tensorflow/mlir: "Multi-Level Intermediate Representation" Compiler Infrastructure [[GitHub]](https://github.com/tensorflow/mlir) [[Video]](https://www.youtube.com/watch?v=qzljG6DKgic)
- PyTorch/glow: Compiler for Neural Network hardware accelerators [[GitHub]](https://github.com/pytorch/glow)
- TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions [[Paper]](https://cs.stanford.edu/~matei/papers/2019/sosp_taso.pdf) [[GitHub]](https://github.com/jiazhihao/TASO)
  - Jia, Zhihao, Oded Padon, James Thomas, Todd Warszawski, Matei Zaharia, and Alex Aiken. (*SOSP 2019*)
  - Experiments tested on TVM and XLA
