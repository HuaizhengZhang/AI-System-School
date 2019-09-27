[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/graphs/commit-activity)
[![Commit Activity](https://img.shields.io/github/commit-activity/m/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg?color=red)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/graphs/commit-activity)
[![Last Commit](https://img.shields.io/github/last-commit/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/commits/master)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![GitHub license](https://img.shields.io/github/license/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg?color=blue)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg?style=social)](https://GitHub.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/stargazers/)


# Awesome System for Machine Learning 

### *Path to system for AI* [[Whitepaper You Must Read]](http://www.sysml.cc/doc/sysml-whitepaper.pdf)

A curated list of research in machine learning system. Link to the code if available is also present. I also summarize some papers if I think they are really interesting.

![AI system](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/blob/master/imgs/AI_system.png)

## Table of Contents
### Resources
- [Survey](#survey)
- [Book](#book)
- [Video](#video)
- [Course](#course)
- [Blog](#blog)
- [Tool](#userful-tools)
- [Project with code](#project)
### Papers

#### System for AI
- [Data Processing](#data-processing)
- [Distributed Training](#machine-learning-system-papers-training)
- [Model Database](#model-database-experiment-version-control)
- [Model Serving](#model-serving)
- [Inference Optimization](#machine-learning-system-papers-inference)
- [Machine Learning Infrastructure](#machine-learning-infrastructure)
- [Machine Learning Compiler](#machine-learning-compiler)
- [AutoML System](#automl-system)
- [Deep Reinforcement Learning System](#deep-reinforcement-learning-system)
- [Edge AI](#edge-or-mobile-papers)
- [Video System](#video-system)

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

## Survey

- awesome-production-machine-learning: A curated list of awesome open source libraries to deploy, monitor, version and scale your machine learning [[GitHub]](https://github.com/EthicalML/awesome-production-machine-learning)
- Survey on End-To-End Machine Learning Automation [[Paper]](https://arxiv.org/pdf/1906.02287.pdf) [[GitHub]](https://github.com/DataSystemsGroupUT/AutoML_Survey)
- Opportunities and Challenges Of Machine Learning Accelerators In Production [[Paper]](https://www.usenix.org/system/files/opml19papers-ananthanarayanan.pdf)
  - Ananthanarayanan, Rajagopal, et al. "
  - 2019 {USENIX} Conference on Operational Machine Learning (OpML 19). 2019.
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
- A Berkeley View of Systems Challenges for AI [[Paper]](https://arxiv.org/pdf/1712.05855.pdf)


## Book

- Computer Architecture: A Quantitative Approach [[Must read]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.115.1881&rep=rep1&type=pdf)
- Streaming Systems [[Book]](https://www.oreilly.com/library/view/streaming-systems/9781491983867/)
- Kubernetes in Action (start to read) [[Book]](https://www.oreilly.com/library/view/kubernetes-in-action/9781617293726/)
- Machine Learning Systems: Designs that scale [[Website]](https://www.manning.com/books/machine-learning-systems)

## Video
- Introduction to Microservices, Docker, and Kubernetes [[YouTube]](https://www.youtube.com/watch?v=1xo-0gCVhTU)
- ICML Keynote: Lessons Learned from Helping 200,000 non-ML experts use ML [[Video]](https://slideslive.com/38916584/keynote-lessons-learned-from-helping-200000-nonml-experts-use-ml)
- Adaptive & Multitask Learning Systems [[Website]](https://www.amtl-workshop.org/schedule)
- System thinking. A TED talk. [[YouTube]](https://www.youtube.com/watch?v=_vS_b7cJn2A)
- Flexible systems are the next frontier of machine learning. Jeff Dean [[YouTube]](https://www.youtube.com/watch?v=Jnunp-EymJQ&list=WL&index=12)
- Is It Time to Rewrite the Operating System in Rust? [[YouTube]](https://www.youtube.com/watch?v=HgtRAbE1nBM&list=WL&index=17&t=0s)
- InfoQ: AI, ML and Data Engineering [[YouTube]](https://www.youtube.com/playlist?list=PLndbWGuLoHeYsZk6VpCEj_SSd9IFgjJ-2)
  - Start to watch.
- Netflix: Human-centric Machine Learning Infrastructure [[InfoQ]](https://www.infoq.com/presentations/netflix-ml-infrastructure?utm_source=youtube&utm_medium=link&utm_campaign=qcontalks)
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
- Tutorial code on how to build your own Deep Learning System in 2k Lines [[GitHub]](https://github.com/tqchen/tinyflow)
- CSE 291F: Advanced Data Analytics and ML Systems. [[UCSD]](http://cseweb.ucsd.edu/classes/wi19/cse291-f/)
- CSci 8980: Machine Learning in Computer Systems [[University of Minnesota, Twin Cities]](http://www-users.cselabs.umn.edu/classes/Spring-2019/csci8980/)
- Mu Li (MxNet, Parameter Server): Introduction to Deep Learning [[Best DL Course I think]](https://courses.d2l.ai/berkeley-stat-157/index.html)  [[Book]](https://www.d2l.ai/)

## Blog

- Building Robust Production-Ready Deep Learning Vision Models in Minutes [[Blog]](https://medium.com/google-developer-experts/building-robust-production-ready-deep-learning-vision-models-in-minutes-acd716f6450a)
- Deploy Machine Learning Models with Keras, FastAPI, Redis and Docker [[Blog]](https://medium.com/@shane.soh/deploy-machine-learning-models-with-keras-fastapi-redis-and-docker-4940df614ece)
- How to Deploy a Machine Learning Model -- Creating a production-ready API using FastAPI + Uvicorn [[Blog]](https://towardsdatascience.com/how-to-deploy-a-machine-learning-model-dc51200fe8cf) [[GitHub]](https://github.com/MaartenGr/ML-API)
- Deploying a Machine Learning Model as a REST API [[Blog]](https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166)
- Continuous Delivery for Machine Learning [[Blog]](https://martinfowler.com/articles/cd4ml.html)
- Kubernetes CheatSheets In A4 [[GitHub]](https://github.com/HuaizhengZhang/cheatsheet-kubernetes-A4)
- A Gentle Introduction to Kubernetes [[Blog]](https://medium.com/faun/a-gentle-introduction-to-kubernetes-4961e443ba26)
- Train and Deploy Machine Learning Model With Web Interface - Docker, PyTorch & Flask [[GitHub]](https://github.com/imadelh/ML-web-app)
- Learning Kubernetes, The Chinese Taoist Way [[GitHub]](https://github.com/caicloud/kube-ladder)
- Data pipelines, Luigi, Airflow: everything you need to know [[Blog]](https://towardsdatascience.com/data-pipelines-luigi-airflow-everything-you-need-to-know-18dc741449b7)
- The Deep Learning Toolset — An Overview [[Blog]](https://medium.com/luminovo/the-deep-learning-toolset-an-overview-b71756016c06)
- Summary of CSE 599W: Systems for ML [[Chinese Blog]](http://jcf94.com/2018/10/04/2018-10-04-cse559w/)
- Polyaxon, Argo and Seldon for Model Training, Package and Deployment in Kubernetes [[Blog]](https://medium.com/analytics-vidhya/polyaxon-argo-and-seldon-for-model-training-package-and-deployment-in-kubernetes-fa089ba7d60b)
- Overview of the different approaches to putting Machine Learning (ML) models in production [[Blog]](https://medium.com/analytics-and-data/overview-of-the-different-approaches-to-putting-machinelearning-ml-models-in-production-c699b34abf86)
- Being a Data Scientist does not make you a Software Engineer [[Part1]](https://towardsdatascience.com/being-a-data-scientist-does-not-make-you-a-software-engineer-c64081526372)
  Architecting a Machine Learning Pipeline [[Part2]](https://towardsdatascience.com/architecting-a-machine-learning-pipeline-a847f094d1c7)
- Model Serving in PyTorch [[Blog]](https://pytorch.org/blog/model-serving-in-pyorch/)
- Machine learning in Netflix [[Medium]](https://medium.com/@NetflixTechBlog)
- SciPy Conference Materials (slides, repo) [[GitHub]](https://github.com/deniederhut/Slides-SciPyConf-2018)
- 继Spark之后，UC Berkeley 推出新一代AI计算引擎——Ray [[Blog]](http://www.qtmuniao.com/2019/04/06/ray/)
- 了解/从事机器学习/深度学习系统相关的研究需要什么样的知识结构？ [[Zhihu]](https://www.zhihu.com/question/315611053/answer/623529977)
- Learn Kubernetes in Under 3 Hours: A Detailed Guide to Orchestrating Containers [[Blog]](https://www.freecodecamp.org/news/learn-kubernetes-in-under-3-hours-a-detailed-guide-to-orchestrating-containers-114ff420e882/) [[GitHub]](https://github.com/rinormaloku/k8s-mastery)
- data-engineer-roadmap: Learning from multiple companies in Silicon Valley. Netflix, Facebook, Google, Startups [[GitHub]](https://github.com/hasbrain/data-engineer-roadmap)
- TensorFlow Serving + Docker + Tornado机器学习模型生产级快速部署 [[Blog]](https://zhuanlan.zhihu.com/p/52096200?utm_source=wechat_session&utm_medium=social&utm_oi=38612796178432)
- Deploying a Machine Learning Model as a REST API [[Blog]](https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166)

## Userful Tools

#### Profile
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
- MLPerf Benchmark Suite/Inference: Reference implementations of inference benchmarks [[GitHub]](https://github.com/mlperf/inference)
- Faiss: A library for efficient similarity search and clustering of dense vectors [[GitHub]](https://github.com/facebookresearch/faiss)
- Microsoft/MMdnn: A comprehensive, cross-framework solution to convert, visualize and diagnose deep neural network models.[[GitHub]](https://github.com/Microsoft/MMdnn)
- gpushare-scheduler-extender [[GitHub]](https://github.com/HuaizhengZhang/gpushare-scheduler-extender)
  - More and more data scientists run their Nvidia GPU based inference tasks on Kubernetes. Some of these tasks can be run on the same Nvidia GPU device to increase GPU utilization. So one important challenge is how to share GPUs between the pods
- Example recipes for Kubernetes Network Policies that you can just copy paste [[GitHub]](https://github.com/ahmetb/kubernetes-network-policy-recipes)


## Project

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


## Data Processing
- Kedro is a workflow development tool that helps you build data pipelines that are robust, scalable, deployable, reproducible and versioned. [[GitHub]](https://github.com/quantumblacklabs/kedro)
- Google/jax: Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more [[GitHub]](https://github.com/google/jax)
- CuPy: NumPy-like API accelerated with CUDA [[GitHub]](https://github.com/cupy/cupy)
- Modin: Speed up your Pandas workflows by changing a single line of code [[GitHub]](https://github.com/modin-project/modin)
- Weld: Weld is a runtime for improving the performance of data-intensive applications. [[Project Website]](https://www.weld.rs/)
- Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines [[Project Website]](http://halide-lang.org/)
  - Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo Durand, Saman Amarasinghe. (*PLDI 2013*)
  - Summary: Halide is a programming language designed to make it easier to write high-performance image and array processing code on modern machines.
- a-mma/AquilaDB: Resilient, Replicated, Decentralized, Host neutral vector database to store Feature Vectors along with JSON Metadata. Do similarity search from anywhere, even from the darkest rifts of Aquila. Production ready solution for Machine Learning engineers and Data scientists. [[GitHub]](https://github.com/a-mma/AquilaDB)
- ShannonAI/service-streamer: Boosting your Web Services of Deep Learning Applications. [[GitHub]](https://github.com/ShannonAI/service-streamer)

## Machine Learning System Papers (Training)

- Class materials for a distributed systems lecture series [[GitHub]](https://github.com/aphyr/distsys-class)

- bytedance/byteps: A high performance and general PS framework for distributed training [[GitHub]](https://github.com/bytedance/byteps)

#### Training(Parallelism)
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
- Horovod: Distributed training framework for TensorFlow, Keras, and PyTorch. 
[[GitHub]](https://github.com/uber/horovod)
- Distributed machine learning infrastructure for large-scale robotics research [[GitHub]](https://github.com/google-research/tensor2robot) [[Blog]](https://ai.google/research/teams/brain/robotics/)

#### Training(Multi-jobs on cluster)
- Gandiva: Introspective cluster scheduling for deep learning. [[Paper]](https://www.usenix.org/system/files/osdi18-xiao.pdf)
  - Xiao, Wencong, et al. (*OSDI 2018*)
  - Summary: Improvet the efficency of hyper-parameter in cluster. Aware of hardware utilization.
- Optimus: an efficient dynamic resource scheduler for deep learning clusters [[Paper]](https://i.cs.hku.hk/~cwu/papers/yhpeng-eurosys18.pdf)
  - Peng, Yanghua, et al. (*EuroSys 2018*)
  - Summary: Job scheduling on clusters. Total complete time as the metric.
- Multi-tenant GPU clusters for deep learning workloads: Analysis and implications. [[Paper]](https://www.microsoft.com/en-us/research/uploads/prod/2018/05/gpu_sched_tr.pdf) [[dataset]](https://github.com/msr-fiddle/philly-traces)
  - Jeon, Myeongjae, Shivaram Venkataraman, Junjie Qian, Amar Phanishayee, Wencong Xiao, and Fan Yang
- Slurm: A Highly Scalable Workload Manager [[GitHub]](https://github.com/SchedMD/slurm)

## Model Database (Experiment Version Control)
- TRAINS - Auto-Magical Experiment Manager & Version Control for AI [[GitHub]](https://github.com/allegroai/trains)
- ModelDB: A system to manage ML models [[GitHub]](https://github.com/mitdbg/modeldb) [[MIT short paper]](https://mitdbg.github.io/modeldb/papers/hilda_modeldb.pdf)
- iterative/dvc: Data & models versioning for ML projects, make them shareable and reproducible [[GitHub]](https://github.com/iterative/dvc)

## Model Serving
- Deep Learning Inference Service at Microsoft [[Paper]](https://www.usenix.org/system/files/opml19papers-soifer.pdf)
  - J Soifer, et al. (*OptML2019*)
- {PRETZEL}: Opening the Black Box of Machine Learning Prediction Serving Systems. [[Paper]](https://www.usenix.org/system/files/osdi18-lee.pdf)
  - Lee, Y., Scolari, A., Chun, B.G., Santambrogio, M.D., Weimer, M. and Interlandi, M., 2018. (*OSDI 2018*)
  - Summary:
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
- Nexus: Nexus is a scalable and efficient serving system for DNN applications on GPU cluster. [[Paper]](https://pdfs.semanticscholar.org/0c0f/353dbac84311ea4f1485d4a8ac0b0459be8c.pdf) [[GitHub]](https://github.com/uwsampl/nexus)
- Deepcpu: Serving rnn-based deep learning models 10x faster. [[Paper]](https://www.usenix.org/system/files/conference/atc18/atc18-zhang-minjia.pdf)
   - Zhang, M., Rajbhandari, S., Wang, W. and He, Y., 2018. (*ATC2018*)
- Orkhon: ML Inference Framework and Server Runtime [[GitHub]](https://github.com/vertexclique/orkhon)
- TensorRT [[NVIDIA]](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)
  - It is designed to work in a complementary fashion with training frameworks such as TensorFlow, Caffe, PyTorch, MXNet, etc. It focuses specifically on running an already trained network quickly and efficiently on a GPU for the purpose of generating a result
- NVIDIA/tensorrt-inference-server: The TensorRT Inference Server provides a cloud inferencing solution optimized for NVIDIA GPUs. [[GitHub]](https://github.com/NVIDIA/tensorrt-inference-server)
- Apache PredictionIO® is an open source Machine Learning Server built on top of a state-of-the-art open source stack for developers and data scientists to create predictive engines for any machine learning task [[Website]](http://predictionio.apache.org/)


## Machine Learning System Papers (Inference)

- TensorRT is a C++ library that facilitates high performance inference on NVIDIA GPUs and deep learning accelerators. [[GitHub]](https://github.com/NVIDIA/TensorRT)
- Dynamic Space-Time Scheduling for GPU Inference [[Paper]](http://learningsys.org/nips18/assets/papers/102CameraReadySubmissionGPU_Virtualization%20(8).pdf)
  - Jain, Paras, et al. (*NIPS 18, System for ML*)
  - Summary: 
- Dynamic Scheduling For Dynamic Control Flow in Deep Learning Systems [[Paper]](http://www.cs.cmu.edu/~jinlianw/papers/dynamic_scheduling_nips18_sysml.pdf)
  - Wei, Jinliang, Garth Gibson, Vijay Vasudevan, and Eric Xing. (*On going*)
- Accelerating Deep Learning Workloads through Efficient Multi-Model Execution. [[Paper]](https://cs.stanford.edu/~matei/papers/2018/mlsys_hivemind.pdf)
  - D. Narayanan, K. Santhanam, A. Phanishayee and M. Zaharia. (*NeurIPS Systems for ML Workshop 2018*)
  - Summary: They assume that their system, HiveMind, is given as input models grouped into model batches that are amenable to co-optimization and co-execution. a compiler, and a runtime.
- DeepCPU: Serving RNN-based Deep Learning Models 10x Faster [[Paper]](https://www.usenix.org/system/files/conference/atc18/atc18-zhang-minjia.pdf)
  - Minjia Zhang, Samyam Rajbhandari, Wenhan Wang, and Yuxiong He, Microsoft AI and Research (*ATC 2018*)
  
  
## Machine Learning Compiler

- TVM: An Automated End-to-End Optimizing Compiler for Deep Learning
[[Project Website]](https://tvm.ai/)
  - {TVM}: An Automated End-to-End Optimizing Compiler for Deep Learning [[Paper]](https://www.usenix.org/system/files/osdi18-chen.pdf) [[YouTube]](https://www.youtube.com/watch?v=I1APhlSjVjs)
    - Chen, Tianqi, et al. (*OSDI 2018*)
- Facebook TC: Tensor Comprehensions (TC) is a fully-functional C++ library to automatically synthesize high-performance machine learning kernels using Halide, ISL and NVRTC or LLVM. [[GitHub]](https://github.com/facebookresearch/TensorComprehensions)
- Tensorflow/mlir: "Multi-Level Intermediate Representation" Compiler Infrastructure [[GitHub]](https://github.com/tensorflow/mlir) [[Video]](https://www.youtube.com/watch?v=qzljG6DKgic)
- PyTorch/glow： Compiler for Neural Network hardware accelerators [[GitHub]](https://github.com/pytorch/glow)

## Machine Learning Infrastructure
- AI infrastructures list [[GitHub]](https://github.com/1duo/awesome-ai-infrastructures)
- cortexlabs/cortex: Deploy machine learning applications without worrying about setting up infrastructure, managing dependencies, or orchestrating data pipelines. [[GitHub]](https://github.com/cortexlabs/cortex)
- Osquery is a SQL powered operating system instrumentation, monitoring, and analytics framework. [[Facebook Project]](https://osquery.io/)
- Seldon: Sheldon Core is an open source platform for deploying machine learning models on a Kubernetes cluster.[[GitHub]](https://github.com/SeldonIO/seldon-core)
- Kubeflow: Kubeflow is a machine learning (ML) toolkit that is dedicated to making deployments of ML workflows on Kubernetes simple, portable, and scalable. [[GitHub]](https://github.com/kubeflow/pipelines)
- Polytaxon: A platform for reproducible and scalable machine learning and deep learning on kubernetes. [[GitHub]](https://github.com/polyaxon/polyaxon)
- MLOps on Azure [[GitHub]](https://github.com/microsoft/MLOps)
- Flame: An ML framework to accelerate research and its path to production. [[GitHub]](https://github.com/Open-ASAPP/flambe)
- Ludwig is a toolbox built on top of TensorFlow that allows to train and test deep learning models without the need to write code. [[GitHub]](https://github.com/uber/ludwig)
- intel-analytics/analytics-zoo Distributed Tensorflow, Keras and BigDL on Apache Spark [[GitHub]](https://github.com/intel-analytics/analytics-zoo)

## AutoML System

#### Survey
- A curated list of automated machine learning papers, articles, tutorials, slides and projects [[GitHub]](https://github.com/hibayesian/awesome-automl-papers)
- Taking human out of learning applications: A survey on automated machine learning. [[Must Read Survey]](https://arxiv.org/pdf/1810.13306.pdf)
  - Quanming, Y., Mengshuo, W., Hugo, J.E., Isabelle, G., Yi-Qi, H., Yu-Feng, L., Wei-Wei, T., Qiang, Y. and Yang, Y.
  
#### General Auto Learning toolkit 
- Google vizier: A service for black-box optimization. [[Paper]](https://ai.google/research/pubs/pub46180.pdf) [[GitHub]](https://github.com/tobegit3hub/advisor)
  - Golovin, Daniel, et al. (*SIGMOD 2017*)
- Aut-sklearn: Automated Machine Learning with scikit-learn [[GitHub]](https://github.com/automl/auto-sklearn) [[Paper]](https://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf)
- Katib: A Distributed General AutoML Platform on Kubernetes [[GitHub]](https://github.com/kubeflow/katib/) [[Paper]](https://www.usenix.org/system/files/opml19papers-zhou.pdf)
- NNI: An open source AutoML toolkit for neural architecture search and hyper-parameter tuning [[GitHub]](https://github.com/Microsoft/nni)
- AutoKeras: Accessible AutoML for deep learning. [[GitHub]](https://github.com/keras-team/autokeras)
- Facebook/Ax: Adaptive experimentation is the machine-learning guided process of iteratively exploring a (possibly infinite) parameter space in order to identify optimal configurations in a resource-efficient manner. [[GitHub]](https://github.com/facebook/Ax)
- DeepSwarm: DeepSwarm is an open-source library which uses Ant Colony Optimization to tackle the neural architecture search problem. [[GitHub]](https://github.com/Pattio/DeepSwarm)
- Google/AdaNet: AdaNet is a lightweight TensorFlow-based framework for automatically learning high-quality models with minimal expert. Importantly, AdaNet provides a general framework for not only learning a neural network architecture, but also for learning to ensemble to obtain even better models. [[GitHub]](https://github.com/tensorflow/adanet)

#### Auto Model Selection

- Automating model search for large scale machine learning. 
  - Sparks, E.R., Talwalkar, A., Haas, D., Franklin, M.J., Jordan, M.I. and Kraska, T., 2015, August.
  - In Proceedings of the Sixth ACM Symposium on Cloud Computing (pp. 368-380). ACM.
    
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
- VideoFlow: Python framework that facilitates the quick development of complex video analysis applications and other series-processing based applications in a multiprocessing environment. [[GitHub]](https://github.com/videoflow/videoflow)
- VidGear: Powerful Multi-Threaded OpenCV and FFmpeg based Turbo Video Processing Python Library with unique State-of-the-Art Features. [[GitHub]](https://github.com/abhiTronix/vidgear)
- NVIDIA DALI: A library containing both highly optimized building blocks and an execution engine for data pre-processing in deep learning applications [[GitHub]](https://github.com/NVIDIA/DALI)
- TensorStream: A library for real-time video stream decoding to CUDA memory [[GitHub]](https://github.com/Fonbet/argus-tensor-stream)
- C++ image processing library with using of SIMD: SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX-512, VMX(Altivec) [[GitHub]](https://github.com/ermig1979/Simd)
- Pretrained image and video models for Pytorch. [[GitHub]](https://github.com/alexandonian/pretorched-x)
- LiveDetect - Live video client to DeepDetect. [[GitHub]](https://github.com/jolibrain/livedetect)

### Papers

- Puffer: Puffer is a Stanford University research study about using machine learning to improve video-streaming algorithms. Please visit [[GitHub]](https://github.com/StanfordSNR/puffer)
- Visual Road: A Video Data Management Benchmark [[Project Website]](http://db.cs.washington.edu/projects/visualroad/)
  - Brandon Haynes, Amrita Mazumdar, Magdalena Balazinska, Luis Ceze, Alvin Cheung (*SIGMOD 2019*)
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
- Mobile Computer Vision @ Facebook [[GitHub]](https://github.com/facebookresearch/mobile-vision)
- Neurosurgeon: Collaborative intelligence between the cloud and mobile edge. [[Paper]](http://web.eecs.umich.edu/~jahausw/publications/kang2017neurosurgeon.pdf)
  - Kang, Y., Hauswald, J., Gao, C., Rovinski, A., Mudge, T., Mars, J. and Tang, L., 2017, April. 
  - In ACM SIGARCH Computer Architecture News (Vol. 45, No. 1, pp. 615-629). ACM.
- 26ms Inference Time for ResNet-50: Towards Real-Time Execution of all DNNs on Smartphone [[Paper]](https://arxiv.org/pdf/1905.00571.pdf)
  - Wei Niu, Xiaolong Ma, Yanzhi Wang, Bin Ren (*ICML2019*)
- NestDNN: Resource-Aware Multi-Tenant On-Device Deep Learning for Continuous Mobile Vision [[Paper]]()
  - Fang, Biyi, Xiao Zeng, and Mi Zhang. (*MobiCom 2018*)
  - Summary: Borrow some ideas from network prune. The pruned model then recovers to trade-off computation resource and accuracy at runtime
- Lavea: Latency-aware video analytics on edge computing platform [[Paper]](http://www.cs.wayne.edu/~weisong/papers/yi17-LAVEA.pdf)
  - Yi, Shanhe, et al. (*Second ACM/IEEE Symposium on Edge Computing. ACM, 2017.*)
- Scaling Video Analytics on Constrained Edge Nodes [[Paper]](http://www.sysml.cc/doc/2019/197.pdf) [[GitHub]](https://github.com/viscloud/filterforward)
  - Canel, C., Kim, T., Zhou, G., Li, C., Lim, H., Andersen, D. G., Kaminsky, M., and Dulloo (*SysML 2019*)
- alibaba/MNN: MNN is a lightweight deep neural network inference engine. It loads models and do inference on devices. [[GitHub]](https://github.com/alibaba/MNN)
- XiaoMi/mobile-ai-bench: Benchmarking Neural Network Inference on Mobile Devices [[GitHub]](https://github.com/XiaoMi/mobile-ai-bench)
- XiaoMi/mace-models: Mobile AI Compute Engine Model Zoo [[GitHub]](https://github.com/XiaoMi/mace-models)
  
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
