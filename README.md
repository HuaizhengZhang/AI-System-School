[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/graphs/commit-activity)
[![Commit Activity](https://img.shields.io/github/commit-activity/m/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg?color=red)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/graphs/commit-activity)
[![Last Commit](https://img.shields.io/github/last-commit/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/commits/master)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![GitHub license](https://img.shields.io/github/license/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg?color=blue)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg?style=social)](https://GitHub.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/stargazers/)

# Awesome System for Machine Learning

### *Path to system for AI* [[Whitepaper You Must Read]](./paper/sysml-whitepaper.pdf)

A curated list of research in machine learning system. Link to the code if available is also present. Now we have a [team](#maintainer) to maintain this project. *You are very welcome to pull request by using our template*.

![AI system](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/blob/master/imgs/AI_system.png)

## General Resources

- [Survey](#survey)
- [Book](#book)
- [Video](#video)
- [Course](#course)
- [Blog](#blog)

## System for AI Papers (Ordered by Category)

- [Data Processing](data_processing.md#data-processing)
- [Training System](training.md#training-system)
- [Inference System](inference.md#inference-system)
- [Machine Learning Infrastructure](infra.md#machine-learning-infrastructure)
- [AutoML System](AutoML_system.md#automl-system)
- [Deep Reinforcement Learning System](drl_system.md#deep-reinforcement-learning-system)
- [Edge AI](edge_system.md#edge-or-mobile-papers)
- [Video System](video_system.md#video-system)
- [GNN System](GNN_system.md#system-for-gnn-traininginference)
- [Federated Learning System](federated_learning_system.md#federated-learning-system)

## System for ML or ML for System on Top System Conference (with notes)

### Conferene

- MLsys [[2021 notes]](./note/MLSys2021.md)
- NSDI [[2021 notes]](./note/NSDI2021.md)
- OSDI [[2020 notes]](./note/OSDI2020.md)
- SOSP
- ATC
- Eurosys [[2021 notes]](./note/Eurosys2021.md)
- Middleware
- SoCC [[2019]](https://acmsocc.github.io/2019/schedule.html) [[2020]](https://acmsocc.github.io/2020/accepted-papers.html)
- TinyML [[2021]](https://openreview.net/group?id=tinyml.org/tinyML/2021/Research_Symposium)

### Workshop

- NIPS learning system workshop
- ICML learning system workshop
- OptML (Rising Star)
- HotCloud
- HotEdge
- HotEdge
- HotOS
- NetAI (ACM SIGCOMM Workshop on Network Meets AI & ML)
- EdgeSys (EuroSys workshop about edge computing) [[website]](https://edge-sys.github.io/2021/)
- EuroMLSys [[website]](https://www.euromlsys.eu/#schedule)
- Large-Scale Distributed Systems and Middleware (LADIS) (EuroSys workshop) [[website]](https://haoc2021.cs.jhu.edu/)


## Survey

- Toward Highly Available, Intelligent Cloud and ML Systems [[Slide]](http://sysnetome.com/Talks/cguo_netai_2018.pdf)
- A curated list of awesome System Designing articles, videos and resources for distributed computing, AKA Big Data. [[GitHub]](https://github.com/madd86/awesome-system-design)
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
- Trust in Machine Learning [[Website]](https://www.manning.com/books/trust-in-machine-learning)

## Video

- ScalaDML2020: Learn from the best minds in the machine learning community. [[Video]](https://info.matroid.com/scaledml-media-archive-preview)
- Jeff Dean: "Achieving Rapid Response Times in Large Online Services" Keynote - Velocity 2014 [[YouTube]](https://www.youtube.com/watch?v=1-3Ahy7Fxsc)
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
- Topics in Networks: Machine Learning for Networking and Systems, Autumn 2019 [[Course Website]](https://people.cs.uchicago.edu/~junchenj/34702-fall19/syllabus.html)
- CS6465: Emerging Cloud Technologies and Systems Challenges [[Cornell]](http://www.cs.cornell.edu/courses/cs6465/2019fa/)
- CS294: AI For Systems and Systems For AI. [[UC Berkeley Spring]](https://github.com/ucbrise/cs294-ai-sys-sp19) (*Strong Recommendation*) [[Machine Learning Systems (Fall 2019)]](https://ucbrise.github.io/cs294-ai-sys-fa19/)
- CSE 599W: System for ML.  [[Chen Tianqi]](https://github.com/tqchen) [[University of Washington]](http://dlsys.cs.washington.edu/)
- EECS 598: Systems for AI (W'20). [[Mosharaf Chowdhury]](https://www.mosharaf.com/) [[Systems for AI (W'20)]](https://github.com/mosharaf/eecs598/tree/w20-ai)
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


## Maintainer

- Huaizheng Zhang, Nanyang Technological University. [:octocat:](https://github.com/HuaizhengZhang)
- Yizheng Huang, Nanyang Technological University. *Focus on Inference System Section* [:octocat:](https://github.com/huangyz0918)
- Meng Shen, Nanyang Technological University. *Focus on Edge AI system Section* [:octocat:](https://github.com/GuluDeemo)
- Weiming Zhuang,Nanyang Technological University & SenseTime. *Focus on Federated Learning System* [:octocat:](https://github.com/weimingwill)
