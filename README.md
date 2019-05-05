[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![GitHub](https://img.shields.io/badge/License-MIT-lightgrey.svg)


# Awesome System for Machine Learning 

### *Path to system for AI* [[Whitepaper You Must Read]](http://www.sysml.cc/doc/sysml-whitepaper.pdf)

A curated list of research in machine learning system. Link to the code if available is also present. I also summarize some papers if I think they are really interesting.

![AI system](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/blob/master/imgs/AI_system.png)

## Table of Contents
### Resources
- [Book](#book)
- [Video](#video)
- [Course](#course)
- [Blog](#blog)
- [Survey](#survey)
- [Tool](#userful-tools)
- [Project with code](#project)
### Papers

#### System for AI
- [Data Processing](#data-processing)
- [Distributed Training](#machine-learning-system-papers-training)
- [Model Serving](#model-serving)
- [Inference Optimization](#machine-learning-system-papers-inference)
- [Machine Learning Compiler](#machine-learning-compiler)
- [AutoML System](#automl-system)
- [Deep Reinforcement Learning System](#deep-reinforcement-learning-system)
- [Edge AI](#edge-or-mobile-papers)
- [Video System](#video-system-papers)

#### AI for System
- [Resource Management](#resource-management)
- [Advanced Theory](#advanced-theory)
- [Traditional System Optimization](#traditional-system-optimization-papers)

### PR template
```
- Title [[Paper]](link) [[GitHub]](link)
  - Author (*conference(journal) year*)
  - Summary: 
```

## Book

- Computer Architecture: A Quantitative Approach [[Must read]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.115.1881&rep=rep1&type=pdf)
- Streaming Systems [[Book]](https://www.oreilly.com/library/view/streaming-systems/9781491983867/)
- Kubernetes in Action (start to read) [[Book]](https://www.oreilly.com/library/view/kubernetes-in-action/9781617293726/)

## Video

- SysML 2019: [[YouTube]](https://www.youtube.com/channel/UChutDKIa-AYyAmbT45s991g/videos)
- ScaledML 2019: David Patterson, Ion Stoica, Dawn Song and so on [[YouTube]](https://www.youtube.com/playlist?list=PLRM2gQVaW_wWXoUnSfZTxpgDmNaAS1RtG)
- ScaledML 2018: Jeff Dean, Ion Stoica, Yangqing Jia and so on [[YouTube]](https://www.youtube.com/playlist?list=PLRM2gQVaW_wW9KAxcibxdqY_TDyvmEjzm) [[Slides]](https://www.matroid.com/blog/post/slides-and-videos-from-scaledml-2018)
- A New Golden Age for Computer Architecture History, Challenges, and Opportunities. David Patterson [[YouTube]](https://www.youtube.com/watch?v=uyc_pDBJotI&t=767s)
- How to Have a Bad Career. David Patterson (I am a big fan) [[YouTube]](https://www.youtube.com/watch?v=Rn1w4MRHIhc)
- SysML 18: Perspectives and Challenges. Michael Jordan [[YouTube]](https://www.youtube.com/watch?v=4inIBmY8dQI&t=26s)
- SysML 18: Systems and Machine Learning Symbiosis. Jeff Dean [[YouTube]](https://www.youtube.com/watch?v=Nj6uxDki6-0)

## Course

- CS294: AI For Systems and Systems For AI. [[UC Berkeley]](https://github.com/ucbrise/cs294-ai-sys-sp19) (*Strong Recommendation*)
- CSE 599W: System for ML.  [[Chen Tianqi]](https://github.com/tqchen) [[University of Washington]](http://dlsys.cs.washington.edu/)
- CSE 291F: Advanced Data Analytics and ML Systems. [[UCSD]](http://cseweb.ucsd.edu/classes/wi19/cse291-f/)
- CSci 8980: Machine Learning in Computer Systems [[University of Minnesota, Twin Cities]](http://www-users.cselabs.umn.edu/classes/Spring-2019/csci8980/)

## Blog

- Machine learning in Netflix [[Medium]](https://medium.com/@NetflixTechBlog)
- 了解/从事机器学习/深度学习系统相关的研究需要什么样的知识结构？ [[Zhihu]](https://www.zhihu.com/question/315611053/answer/623529977)
- SciPy Conference Materials (slides, repo) [[GitHub]](https://github.com/deniederhut/Slides-SciPyConf-2018)

## Survey

- Scalable Deep Learning on Distributed Infrastructures: Challenges, Techniques and Tools [[Paper]](https://arxiv.org/pdf/1903.11314.pdf)
  - RUBEN MAYER, HANS-ARNO JACOBSEN
  - Summary:
- How (and How Not) to Write a Good Systems Paper [[Advice]](https://www.usenix.org/legacy/events/samples/submit/advice_old.html)
- Applied machine learning at Facebook: a datacenter infrastructure perspective [[Paper]](https://research.fb.com/wp-content/uploads/2017/12/hpca-2018-facebook.pdf)
  - Hazelwood, Kim, et al. (*HPCA 2018*)
- Infrastructure for Usable Machine Learning: The Stanford DAWN Project
  - Bailis, Peter, Kunle Olukotun, Christopher Ré, and Matei Zaharia. (*preprint 2017*)
- Hidden technical debt in machine learning systems [[Paper]](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf)
  - Sculley, David, et al. (*NIPS 2015*)
  - Summary: 
- End-to-end arguments in system design [[Paper]](http://web.mit.edu/Saltzer/www/publications/endtoend/endtoend.pdf)
  - Saltzer, Jerome H., David P. Reed, and David D. Clark. 
- System Design for Large Scale Machine Learning [[Thesis]](http://shivaram.org/publications/shivaram-dissertation.pdf)
- Deep Learning Inference in Facebook Data Centers: Characterization, Performance Optimizations and Hardware Implications [[Paper]](https://arxiv.org/pdf/1811.09886.pdf)
  - Park, Jongsoo, Maxim Naumov, Protonu Basu et al. *arXiv 2018*
  - Summary: This paper presents a characterizations of DL models and then shows the new design principle of DL hardware.

## Userful Tools
- Pyflame: A Ptracing Profiler For Python [[GitHub]](https://github.com/uber/pyflame)
- Netron: Visualizer for deep learning and machine learning models [[GitHub]](https://github.com/lutzroeder/netron)
- Facebook/FBGEMM: FBGEMM (Facebook GEneral Matrix Multiplication) is a low-precision, high-performance matrix-matrix multiplications and convolution library for server-side inference. [[GitHub]](https://github.com/pytorch/FBGEMM)
- XiaoMi/mobile-ai-bench: Benchmarking Neural Network Inference on Mobile Devices [[GitHub]](https://github.com/XiaoMi/mobile-ai-bench)
- Dslabs: Distributed Systems Labs and Framework for UW system course [[GitHub]](https://github.com/emichael/dslabs)
- Machine Learning Model Zoo [[Website]](https://modelzoo.co/)
- MLPerf Benchmark Suite/Inference: Reference implementations of inference benchmarks [[GitHub]](https://github.com/mlperf/inference)
- Pytorch-Memory-Utils: detect your GPU memory during training with Pytorch. [[GitHub]](https://github.com/Oldpan/Pytorch-Memory-Utils)
- Faiss: A library for efficient similarity search and clustering of dense vectors [[GitHub]](https://github.com/facebookresearch/faiss)
- torchstat: a lightweight neural network analyzer based on PyTorch. [[GitHub]](https://github.com/Swall0w/torchstat)
- Microsoft/MMdnn: A comprehensive, cross-framework solution to convert, visualize and diagnose deep neural network models.[[GitHub]](https://github.com/Microsoft/MMdnn)
- Popular Network memory consumption and FLOP counts [[GitHub]](https://github.com/albanie/convnet-burden)
- Intel® VTune™ Amplifier [[Website]](https://software.intel.com/en-us/vtune)
  - Stop guessing why software is slow. Advanced sampling and profiling techniques quickly analyze your code, isolate issues, and deliver insights for optimizing performance on modern processors
- gpushare-scheduler-extender [[GitHub]](https://github.com/HuaizhengZhang/gpushare-scheduler-extender)
  - More and more data scientists run their Nvidia GPU based inference tasks on Kubernetes. Some of these tasks can be run on the same Nvidia GPU device to increase GPU utilization. So one important challenge is how to share GPUs between the pods
- TensorRT [[NVIDIA]](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)
  - It is designed to work in a complementary fashion with training frameworks such as TensorFlow, Caffe, PyTorch, MXNet, etc. It focuses specifically on running an already trained network quickly and efficiently on a GPU for the purpose of generating a result


## Project
- Machine Learning for .NET [[GitHub]](https://github.com/dotnet/machinelearning)
  - ML.NET is a cross-platform open-source machine learning framework which makes machine learning accessible to .NET developers.
  - ML.NET allows .NET developers to develop their own models and infuse custom machine learning into their applications, using .NET, even without prior expertise in developing or tuning machine learning models.
- ONNX: Open Neural Network Exchange [[GitHub]](https://github.com/onnx/onnx)
- BentoML: Machine Learning Toolkit for packaging and deploying models [[GitHub]](https://github.com/bentoml/BentoML)
- ModelDB: A system to manage ML models [[GitHub]](https://github.com/mitdbg/modeldb) [[MIT short paper]](https://mitdbg.github.io/modeldb/papers/hilda_modeldb.pdf)
- EuclidesDB: A multi-model machine learning feature embedding database [[GitHub]](https://github.com/perone/euclidesdb)
- Prefect: Perfect is a new workflow management system, designed for modern infrastructure and powered by the open-source Prefect Core workflow engine. [[GitHub]](https://github.com/PrefectHQ/prefect)
- MindsDB: MindsDB's goal is to make it very simple for developers to use the power of artificial neural networks in their projects [[GitHub]](https://github.com/mindsdb/mindsdb)
- PAI: OpenPAI is an open source platform that provides complete AI model training and resource management capabilities. [[Microsoft Project]](https://github.com/Microsoft/pai#resources)
- Bistro: Scheduling Data-Parallel Jobs Against Live Production Systems [[Facebook Project]](https://github.com/facebook/bistro)
- Osquery is a SQL powered operating system instrumentation, monitoring, and analytics framework. [[Facebook Project]](https://osquery.io/)
- Seldon: Sheldon Core is an open source platform for deploying machine learning models on a Kubernetes cluster.[[GitHub]](https://github.com/SeldonIO/seldon-core)
- Kubeflow: Kubeflow is a machine learning (ML) toolkit that is dedicated to making deployments of ML workflows on Kubernetes simple, portable, and scalable. [[GitHub]](https://github.com/kubeflow/pipelines)

## Data Prcocessing
- Google/jax: Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more [[GitHub]](https://github.com/google/jax)
- CuPy: NumPy-like API accelerated with CUDA [[GitHub]](https://github.com/cupy/cupy)
- Modin: Speed up your Pandas workflows by changing a single line of code [[GitHub]](https://github.com/modin-project/modin)
- Weld: Weld is a runtime for improving the performance of data-intensive applications. [[Project Website]](https://www.weld.rs/)
- Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines [[Project Website]](http://halide-lang.org/)
  - Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo Durand, Saman Amarasinghe. (*PLDI 2013*)
  - Summary: Halide is a programming language designed to make it easier to write high-performance image and array processing code on modern machines.
  
## Machine Learning System Papers (Training)
- Exploring Hidden Dimensions in Parallelizing Convolutional Neural Networks. [[Paper]](http://proceedings.mlr.press/v80/jia18a/jia18a.pdf) [[GitHub]](https://github.com/flexflow/FlexFlow)
  - Zhihao Jia, Sina Lin, Charles R. Qi, and Alex Aiken. (*ICML 2018*)
- Mesh-TensorFlow: Deep Learning for Supercomputers [[Paper]](https://arxiv.org/pdf/1811.02084.pdf) [[GitHub]](https://github.com/tensorflow/mesh)
  - Shazeer, Noam, Youlong Cheng, Niki Parmar, Dustin Tran, et al. (*NIPS 2018*)
  - Summary: Data parallelism for language model
- PyTorch-BigGraph: A Large-scale Graph Embedding System [[Paper]](https://arxiv.org/pdf/1903.12287.pdf) [[GitHub]](https://github.com/facebookresearch/PyTorch-BigGraph)
  - Lerer, Adam and Wu, Ledell and Shen, Jiajun and Lacroix, Timothee and Wehrstedt, Luca and Bose, Abhijit and Peysakhovich, Alex (*SysML 2019*)
- Beyond data and model parallelism for deep neural networks [[Paper]](https://arxiv.org/pdf/1807.05358.pdf) [[GitHub]](https://github.com/jiazhihao/metaflow_sysml19)
  - Jia, Zhihao, Matei Zaharia, and Alex Aiken. (*SysML 2019*)
  - Summary: SOAP (sample, operation, attribution and parameter) parallelism. Operator graph, device topology and extution optimizer. MCMC search algorithm and excution simulator.
- Device placement optimization with reinforcement learning [[Paper]](https://arxiv.org/pdf/1706.04972.pdf)
  - Mirhoseini, Azalia, Hieu Pham, Quoc V. Le, Benoit Steiner, Rasmus Larsen, Yuefeng Zhou, Naveen Kumar, Mohammad Norouzi, Samy Bengio, and Jeff Dean. (*ICML 17*)
  - Summary: Using REINFORCE learn a device placement policy. Group operations to excute. Need a lot of GPUs.
- Spotlight: Optimizing device placement for training deep neural networks  [[Paper]](http://proceedings.mlr.press/v80/gao18a/gao18a.pdf)
  - Gao, Yuanxiang, Li Chen, and Baochun Li (*ICML 18*)
- GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism [[Paper]](https://arxiv.org/pdf/1811.06965.pdf)[[GitHub]](https://github.com/tensorflow/lingvo/blob/master/lingvo/core/gpipe.py) [[News]](https://www.cnbeta.com/articles/tech/824495.htm)
  - Huang, Yanping, et al. (*arXiv preprint arXiv:1811.06965 (2018)*)
  - Summary: 
- Gandiva: Introspective cluster scheduling for deep learning. [[Paper]](https://www.usenix.org/system/files/osdi18-xiao.pdf)
  - Xiao, Wencong, et al. (*OSDI 2018*)
  - Summary: Improvet the efficency of hyper-parameter in cluster. Aware of hardware utilization.
- Optimus: an efficient dynamic resource scheduler for deep learning clusters [[Paper]](https://i.cs.hku.hk/~cwu/papers/yhpeng-eurosys18.pdf)
  - Peng, Yanghua, et al. (*EuroSys 2018*)
  - Summary: Job scheduling on clusters. Total complete time as the metric.
- Horovod: Distributed training framework for TensorFlow, Keras, and PyTorch. 
[[GitHub]](https://github.com/uber/horovod)


## Model Serving
- {PRETZEL}: Opening the Black Box of Machine Learning Prediction Serving Systems. [[Paper]](https://www.usenix.org/system/files/osdi18-lee.pdf)
  - Lee, Y., Scolari, A., Chun, B.G., Santambrogio, M.D., Weimer, M. and Interlandi, M., 2018. (*OSDI 2018*)
  - Summary:
- Brusta: PyTorch model serving project [[GitHub]](https://github.com/hyoungseok/brusta)
- Model Server for Apache MXNet: Model Server for Apache MXNet is a tool for serving neural net models for inference [[GitHub]](https://github.com/awslabs/mxnet-model-server)
- TFX: A TensorFlow-Based Production-Scale Machine Learning Platform [[Paper]](http://stevenwhang.com/tfx_paper.pdf) [[Website]](https://www.tensorflow.org/tfx)
  - Baylor, Denis, et al. (*KDD 2017*)
  - Summary: 
- Tensorflow-serving: Flexible, high-performance ml serving [[Paper]](https://arxiv.org/pdf/1712.06139) [[GitHub]](https://github.com/tensorflow/serving)
  - Olston, Christopher, et al.
- IntelAI/OpenVINO-model-server: Inference model server implementation with gRPC interface, compatible with TensorFlow serving API and OpenVINO™ as the execution backend. [[GitHub]](https://github.com/IntelAI/OpenVINO-model-server)
- Clipper: A Low-Latency Online Prediction Serving System [[Paper]](https://www.usenix.org/system/files/conference/nsdi17/nsdi17-crankshaw.pdf)
[[GitHub]](https://github.com/ucbrise/clipper)
  - Crankshaw, Daniel, et al. (*NSDI 2017*)
  - Summary: Adaptive batch 
- InferLine: ML Inference Pipeline Composition Framework [[Paper]](https://arxiv.org/pdf/1812.01776.pdf)
  - Crankshaw, Daniel, et al. (*Preprint*)
  - Summary: update version of Clipper
- TrIMS: Transparent and Isolated Model Sharing for Low Latency Deep LearningInference in Function as a Service Environments [[Paper]](https://arxiv.org/pdf/1811.09732.pdf)
  - Dakkak, Abdul, et al (*Preprint*)
  - Summary: model cold start problem
- Rafiki: machine learning as an analytics service system [[Paper]](http://www.vldb.org/pvldb/vol12/p128-wang.pdf) [[GitHub]](https://github.com/nginyc/rafiki)
  - Wang, Wei, Jinyang Gao, Meihui Zhang, Sheng Wang, Gang Chen, Teck Khim Ng, Beng Chin Ooi, Jie Shao, and Moaz Reyad.
  - Summary: Contain both training and inference. Auto-Hype-Parameter search for training. Ensemble models for inference. Using DRL to balance trade-off between accuracy and latency.
- GraphPipe: Machine Learning Model Deployment Made Simple [[GitHub]](https://github.com/oracle/graphpipe)
- Nexus: Nexus is a scalable and efficient serving system for DNN applications on GPU cluster. [[GitHub]](https://github.com/uwsampl/nexus)


## Machine Learning System Papers (Inference)

- Dynamic Space-Time Scheduling for GPU Inference [[Paper]](http://learningsys.org/nips18/assets/papers/102CameraReadySubmissionGPU_Virtualization%20(8).pdf)
  - Jain, Paras, et al. (*NIPS 18, System for ML*)
  - Summary: 
- Dynamic Scheduling For Dynamic Control Flow in Deep Learning Systems [[Paper]](http://www.cs.cmu.edu/~jinlianw/papers/dynamic_scheduling_nips18_sysml.pdf)
  - Wei, Jinliang, Garth Gibson, Vijay Vasudevan, and Eric Xing. (*On going*)
- Accelerating Deep Learning Workloads through Efficient Multi-Model Execution. [[Paper]](https://cs.stanford.edu/~matei/papers/2018/mlsys_hivemind.pdf)
  - D. Narayanan, K. Santhanam, A. Phanishayee and M. Zaharia. (*NeurIPS Systems for ML Workshop 2018*)
  - Summary: They assume that their system, HiveMind, is given as input models grouped into model batches that are amenable to co-optimization and co-execution. a compiler, and a runtime.

 
## Machine Learning Compiler

- TVM: An Automated End-to-End Optimizing Compiler for Deep Learning
[[Project Website]](https://tvm.ai/)
  - {TVM}: An Automated End-to-End Optimizing Compiler for Deep Learning [[Paper]](https://www.usenix.org/system/files/osdi18-chen.pdf)
    - Chen, Tianqi, et al. (*OSDI 2018*)
- Facebook TC: Tensor Comprehensions (TC) is a fully-functional C++ library to automatically synthesize high-performance machine learning kernels using Halide, ISL and NVRTC or LLVM. [[GitHub]](https://github.com/facebookresearch/TensorComprehensions)
- Tensorflow/mlir: "Multi-Level Intermediate Representation" Compiler Infrastructure [[GitHub]](https://github.com/tensorflow/mlir)
- PyTorch/glow： Compiler for Neural Network hardware accelerators [[GitHub]](https://github.com/pytorch/glow)

## AutoML System
- Taking human out of learning applications: A survey on automated machine learning. [[Must Read Survey]](https://arxiv.org/pdf/1810.13306.pdf)
  - Quanming, Y., Mengshuo, W., Hugo, J.E., Isabelle, G., Yi-Qi, H., Yu-Feng, L., Wei-Wei, T., Qiang, Y. and Yang, Y.
- Aut-sklearn: Automated Machine Learning with scikit-learn [[GitHub]](https://github.com/automl/auto-sklearn) [[Paper]](https://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf)
- NNI: An open source AutoML toolkit for neural architecture search and hyper-parameter tuning [[GitHub]](https://github.com/Microsoft/nni)
- AutoKeras: Accessible AutoML for deep learning. [[GitHub]](https://github.com/keras-team/autokeras)
- Facebook/Ax: Adaptive experimentation is the machine-learning guided process of iteratively exploring a (possibly infinite) parameter space in order to identify optimal configurations in a resource-efficient manner. [[GitHub]](https://github.com/facebook/Ax)
- DeepSwarm: DeepSwarm is an open-source library which uses Ant Colony Optimization to tackle the neural architecture search problem. [[GitHub]](https://github.com/Pattio/DeepSwarm)
    
## Deep Reinforcement Learning System

- Ray: A Distributed Framework for Emerging {AI} Applications [[GitHub]](https://www.usenix.org/conference/osdi18/presentation/moritz)
  - Moritz, Philipp, et al. (*OSDI 2018*)
  - Summary: Distributed DRL training, simulation and inference system. Can be used as a high-performance python framework.
- Elf: An extensive, lightweight and flexible research platform for real-time strategy games [[Paper]](https://papers.nips.cc/paper/6859-elf-an-extensive-lightweight-and-flexible-research-platform-for-real-time-strategy-games.pdf) [[GitHub]](https://github.com/facebookresearch/ELF)
  - Tian, Yuandong, Qucheng Gong, Wenling Shang, Yuxin Wu, and C. Lawrence Zitnick. (*NIPS 2017*)
  - Summary:
- Horizon: Facebook's Open Source Applied Reinforcement Learning Platform [[Paper]](https://arxiv.org/pdf/1811.00260) [[GitHub]](https://github.com/facebookresearch/Horizon)
  - Gauci, Jason, et al. (*preprint 2019*)
- RLgraph: Modular Computation Graphs for Deep Reinforcement Learning [[Paper]](http://www.sysml.cc/doc/2019/43.pdf)[[GitHub]](https://github.com/rlgraph/rlgraph)
  - Schaarschmidt, Michael, Sven Mika, Kai Fricke, and Eiko Yoneki. (*SysML 2019*)
  - Summary:

## Video System

### Tools
- VidGear: Powerful Multi-Threaded OpenCV and FFmpeg based Turbo Video Processing Python Library with unique State-of-the-Art Features. [[GitHub]](https://github.com/abhiTronix/vidgear)
- NVIDIA DALI: A library containing both highly optimized building blocks and an execution engine for data pre-processing in deep learning applications [[GitHub]](https://github.com/NVIDIA/DALI)
- TensorStream: A library for real-time video stream decoding to CUDA memory [[GitHub]](https://github.com/Fonbet/argus-tensor-stream)

### Papers
- CaTDet: Cascaded Tracked Detector for Efficient Object Detection from Video [[Paper]](http://www.sysml.cc/doc/2019/111.pdf)
  - Mao, Huizi, Taeyoung Kong, and William J. Dally. (*SysML2019*)
- Live Video Analytics at Scale with Approximation and Delay-Tolerance [[Paper]](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/videostorm_nsdi17.pdf)
  - Zhang, Haoyu, Ganesh Ananthanarayanan, Peter Bodik, Matthai Philipose, Paramvir Bahl, and Michael J. Freedman. (*NSDI 2017*)
- Chameleon: scalable adaptation of video analytics [[Paper]](http://people.cs.uchicago.edu/~junchenj/docs/Chameleon_SIGCOMM_CameraReady.pdf)
  - Jiang, Junchen, et al. (*SIGCOMM 2018*)
  - Summary: Configuration controller for balancing accuracy and resource. Golden configuration is a good design. Periodic profiling often exceeded any resource savings gained by adapting the configurations.
- Noscope: optimizing neural network queries over video at scale [[Paper]](https://arxiv.org/pdf/1703.02529) [[GitHub]](https://github.com/stanford-futuredata/noscope)
  - Kang, Daniel, John Emmons, Firas Abuzaid, Peter Bailis, and Matei Zaharia. (*VLDB2017*)
  - Summary: 
- SVE: Distributed video processing at Facebook scale [[Paper]](http://www.cs.princeton.edu/~wlloyd/papers/sve-sosp17.pdf)
  - Huang, Qi, et al. (*SOSP2017*)
  - Summary: 
- Scanner: Efficient Video Analysis at Scale [[Paper]](http://graphics.stanford.edu/papers/scanner/poms18_scanner.pdf)[[GitHub]](https://github.com/scanner-research/scanner)
  - Poms, Alex, Will Crichton, Pat Hanrahan, and Kayvon Fatahalian (*SIGGRAPH 2018*)
  - Summary:
- A cloud-based large-scale distributed video analysis system [[Paper]](https://ai.google/research/pubs/pub45631)
  - Wang, Yongzhe, et al. (*ICIP 2016*)
- Rosetta: Large scale system for text detection and recognition in images [[Paper]](https://research.fb.com/wp-content/uploads/2018/10/Rosetta-Large-scale-system-for-text-detection-and-recognition-in-images.pdf)
  - Borisyuk, Fedor, Albert Gordo, and Viswanath Sivakumar. (*KDD 2018*)
  - Summary: 
- Neural adaptive content-aware internet video delivery. [[Paper]](https://www.usenix.org/system/files/osdi18-yeo.pdf) [[GitHub]](https://github.com/kaist-ina/NAS_public)
  - Yeo, H., Jung, Y., Kim, J., Shin, J. and Han, D., 2018.  (*OSDI 2018*)
  - Summary: Combine video super-resolution and ABR
  
## Edge or Mobile Papers 

- NestDNN: Resource-Aware Multi-Tenant On-Device Deep Learning for Continuous Mobile Vision [[Paper]]()
  - Fang, Biyi, Xiao Zeng, and Mi Zhang. (*MobiCom 2018*)
  - Summary: Borrow some ideas from network prune. The pruned model then recovers to trade-off computation resource and accuracy at runtime
- Lavea: Latency-aware video analytics on edge computing platform [[Paper]](http://www.cs.wayne.edu/~weisong/papers/yi17-LAVEA.pdf)
  - Yi, Shanhe, et al. (*Second ACM/IEEE Symposium on Edge Computing. ACM, 2017.*)
- Scaling Video Analytics on Constrained Edge Nodes [[Paper]](http://www.sysml.cc/doc/2019/197.pdf) [[GitHub]](https://github.com/viscloud/filterforward)
  - Canel, C., Kim, T., Zhou, G., Li, C., Lim, H., Andersen, D. G., Kaminsky, M., and Dulloo (*SysML 2019*)
  
## Resource Management

- Resource management with deep reinforcement learning [[Paper]](https://people.csail.mit.edu/alizadeh/papers/deeprm-hotnets16.pdf) [[GitHub]](https://github.com/hongzimao/deeprm)
  - Mao, Hongzi, Mohammad Alizadeh, Ishai Menache, and Srikanth Kandula (*ACM HotNets 2016*)
  - Summary:  Highly cited paper. Nice definaton. An example solution that translates the problem of packing tasks with multiple resource demands into a learning problem and then used DRL to solve it.

## Advanced Theory

- Differentiable MPC for End-to-end Planning and Control [[Paper]](https://www.cc.gatech.edu/~bboots3/files/DMPC.pdf)  [[GitHub]](https://locuslab.github.io/mpc.pytorch/)
  - Amos, Brandon, Ivan Jimenez, Jacob Sacks, Byron Boots, and J. Zico Kolter (*NIPS 2018*)
  
## Traditional System Optimization Papers

- AutoScale: Dynamic, Robust Capacity Management for Multi-Tier Data Centers
[[Paper]](https://www3.cs.stonybrook.edu/~anshul/tocs12.pdf)
  - Gandhi, Anshul, et al. (*TOCS 2012*)
- Large-scale cluster management at Google with Borg [[Paper]](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/43438.pdf)
  - Verma, Abhishek, et al. (*ECCS2015*)
