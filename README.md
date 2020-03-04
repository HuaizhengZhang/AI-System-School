[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/graphs/commit-activity)
[![Commit Activity](https://img.shields.io/github/commit-activity/m/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg?color=red)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/graphs/commit-activity)
[![Last Commit](https://img.shields.io/github/last-commit/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/commits/master)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![GitHub license](https://img.shields.io/github/license/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg?color=blue)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg?style=social)](https://GitHub.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/stargazers/)


# Awesome System for Machine Learning 

### *Path to system for AI* [[Whitepaper You Must Read]](./paper/sysml-whitepaper.pdf)

A curated list of research in machine learning system. Link to the code if available is also present. I also summarize some papers if I think they are really interesting.

![AI system](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/blob/master/imgs/AI_system.png)


## General Resources
- [Survey](#survey)
- [Book](#book)
- [Video](#video)
- [Course](#course)
- [Blog](#blog)
- [Tool](#userful-tools)
- [Project with code](#project)

## System for AI Papers
- [Data Processing](data_processing.md#data-processing)
- [Training System](training.md#training-system)
- [Inference System](inference.md#inference-system)
- [Machine Learning Infrastructure](infra.md#machine-learning-infrastructure)
- [AutoML System](AutoML_system.md#automl-system)
- [Deep Reinforcement Learning System](drl_system.md#deep-reinforcement-learning-system)
- [Edge AI](edge_system.md#edge-or-mobile-papers)
- [Video System](video_system.md#video-system)


### PR template
```
- Title [[Paper]](link) [[GitHub]](link)
  - Author (*conference(journal) year*)
  - Summary: 
```

## Survey
- Toward Highly Available, Intelligent Cloud and ML Systems [[Slide]](http://sysnetome.com/Talks/cguo_netai_2018.pdf)
- awesome-production-machine-learning: A curated list of awesome open source libraries to deploy, monitor, version and scale your machine learning [[GitHub]](https://github.com/EthicalML/awesome-production-machine-learning)
- Opportunities and Challenges Of Machine Learning Accelerators In Production [[Paper]](https://www.usenix.org/system/files/opml19papers-ananthanarayanan.pdf)
  - Ananthanarayanan, Rajagopal, et al. "
  - 2019 {USENIX} Conference on Operational Machine Learning (OpML 19). 2019.
- How (and How Not) to Write a Good Systems Paper [[Advice]](https://www.usenix.org/legacy/events/samples/submit/advice_old.html)
- Applied machine learning at Facebook: a datacenter infrastructure perspective [[Paper]](https://research.fb.com/wp-content/uploads/2017/12/hpca-2018-facebook.pdf)
  - Hazelwood, Kim, et al. (*HPCA 2018*)
- Infrastructure for Usable Machine Learning: The Stanford DAWN Project
  - Bailis, Peter, Kunle Olukotun, Christopher Ré, and Matei Zaharia. (*preprint 2017*)
- Hidden technical debt in machine learning systems [[Paper]](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf)
  - Sculley, David, et al. (*NIPS 2015*)
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

- From Research to Production with PyTorch [[Video]](https://www.infoq.com/presentations/pytorch-torchscript-botorch/#downloadPdf/)
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

- CS6465: Emerging Cloud Technologies and Systems Challenges [[Cornell]](http://www.cs.cornell.edu/courses/cs6465/2019fa/)
- CS294: AI For Systems and Systems For AI. [[UC Berkeley Spring]](https://github.com/ucbrise/cs294-ai-sys-sp19) (*Strong Recommendation*) [[Machine Learning Systems (Fall 2019)]](https://ucbrise.github.io/cs294-ai-sys-fa19/)
- CSE 599W: System for ML.  [[Chen Tianqi]](https://github.com/tqchen) [[University of Washington]](http://dlsys.cs.washington.edu/)
- Tutorial code on how to build your own Deep Learning System in 2k Lines [[GitHub]](https://github.com/tqchen/tinyflow)
- CSE 291F: Advanced Data Analytics and ML Systems. [[UCSD]](http://cseweb.ucsd.edu/classes/wi19/cse291-f/)
- CSci 8980: Machine Learning in Computer Systems [[University of Minnesota, Twin Cities]](http://www-users.cselabs.umn.edu/classes/Spring-2019/csci8980/)
- Mu Li (MxNet, Parameter Server): Introduction to Deep Learning [[Best DL Course I think]](https://courses.d2l.ai/berkeley-stat-157/index.html)  [[Book]](https://www.d2l.ai/)

## Blog

- Parallelizing across multiple CPU/GPUs to speed up deep learning inference at the edge [[Amazon Blog]](https://aws.amazon.com/blogs/machine-learning/parallelizing-across-multiple-cpu-gpus-to-speed-up-deep-learning-inference-at-the-edge/)
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



