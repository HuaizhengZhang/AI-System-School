# Awesome-System-for-Machine-Learning[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<p align="center">
  <img width="250" src="https://camo.githubusercontent.com/1131548cf666e1150ebd2a52f44776d539f06324/68747470733a2f2f63646e2e7261776769742e636f6d2f73696e647265736f726875732f617765736f6d652f6d61737465722f6d656469612f6c6f676f2e737667" "Awesome!">
</p>



## Book

- Computer Architecture: A Quantitative Approach [[Must read]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.115.1881&rep=rep1&type=pdf)

- Streaming Systems [[Book]](https://www.oreilly.com/library/view/streaming-systems/9781491983867/)

## Course

- CS294: AI For Systems and Systems For AI  [[Course Website]](https://github.com/ucbrise/cs294-ai-sys-sp19) ### *Strong Recommendation*

## Survey

- Hidden technical debt in machine learning systems [[Paper]](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf)
  - Sculley, David, et al. (*NIPS 2015*)
  - Summary: 

- End-to-end arguments in system design [[Paper]](http://web.mit.edu/Saltzer/www/publications/endtoend/endtoend.pdf)
  - Saltzer, Jerome H., David P. Reed, and David D. Clark. 
  
- System Design for Large Scale Machine Learning [[Thesis]](http://shivaram.org/publications/shivaram-dissertation.pdf)

- Deep Learning Inference in Facebook Data Centers: Characterization, Performance Optimizations and Hardware Implications [[Paper]]https://arxiv.org/pdf/1811.09886.pdf)
  - Park, Jongsoo, Maxim Naumov, Protonu Basu et al. *arXiv 2018*
  - Summary: This paper presents a characterizations of DL models and then shows the new design principle of DL hardware.

## Userful Tools

- Intel® VTune™ Amplifier [[Website]](https://software.intel.com/en-us/vtune)
  - Stop guessing why software is slow. Advanced sampling and profiling techniques quickly analyze your code, isolate issues, and deliver insights for optimizing performance on modern processors
  
- NVIDIA DALI [[GitHub]](https://github.com/NVIDIA/DALI)
  - A library containing both highly optimized building blocks and an execution engine for data pre-processing in deep learning applications
  
- gpushare-scheduler-extender [[GitHub]](https://github.com/HuaizhengZhang/gpushare-scheduler-extender)
  - Some of these tasks can be run on the same Nvidia GPU device to increase GPU utilization
  
- Machine Learning Pipelines for Kubeflow [[GitHub]](https://github.com/kubeflow/pipelines)
  - Kubeflow is a machine learning (ML) toolkit that is dedicated to making deployments of ML workflows on Kubernetes simple, portable, and scalable.

#### 

## Project

- MindsDB: MindsDB's goal is to make it very simple for developers to use the power of artificial neural networks in their projects [[GitHub]](https://github.com/mindsdb/mindsdb)

- PAI: OpenPAI is an open source platform that provides complete AI model training and resource management capabilities. [[Microsoft Project]](https://github.com/Microsoft/pai#resources)

- Bistro: Scheduling Data-Parallel Jobs Against Live Production Systems [[Facebook Project]](https://github.com/facebook/bistro)

- Osquery is a SQL powered operating system instrumentation, monitoring, and analytics framework. [[Facebook Project]](https://osquery.io/)

- TVM: An Automated End-to-End Optimizing Compiler for Deep Learning
[[Project Website]](https://tvm.ai/)

- Horovod: Distributed training framework for TensorFlow, Keras, and PyTorch. 
[[GitHub]](https://github.com/uber/horovod)

## Machine Learning System Papers (Inference)

- Clipper: A Low-Latency Online Prediction Serving System
[[GitHub]](https://github.com/ucbrise/clipper)
  - Crankshaw, Daniel, et al. (*NSDI 2017*)
  - Summary: Adaptive batch
  
- InferLine: ML Inference Pipeline Composition Framework [[Paper]](https://arxiv.org/pdf/1812.01776.pdf)
  - Crankshaw, Daniel, et al. (*Preprint*)
  - Summary:

- TrIMS: Transparent and Isolated Model Sharing for Low Latency Deep LearningInference in Function as a Service Environments [[Paper]](https://arxiv.org/pdf/1811.09732.pdf)
  - Dakkak, Abdul, et al (*Preprint*)
  - Summary:

- Dynamic Space-Time Scheduling for GPU Inference [[Paper]](http://learningsys.org/nips18/assets/papers/102CameraReadySubmissionGPU_Virtualization%20(8).pdf)
  - Jain, Paras, et al. (*NIPS 18, System for ML*)
  - Summary: 

- Dynamic Scheduling For Dynamic Control Flow in Deep Learning Systems
  - Wei, Jinliang, Garth Gibson, Vijay Vasudevan, and Eric Xing. (*On going*)
  - Summary:
  
- Accelerating Deep Learning Workloads through Efficient Multi-Model Execution. 
  - D. Narayanan, K. Santhanam, A. Phanishayee and M. Zaharia. (*NeurIPS Systems for ML Workshop 2018*)
  - Summary: They assume that their system, HiveMind, is given as input models grouped into model batches that are amenable to co-optimization and co-execution. a compiler, and a runtime.

## Machine Learning System Papers (Training)

- Beyond data and model parallelism for deep neural networks
  - Jia, Zhihao, Matei Zaharia, and Alex Aiken. (*SysML 2019*)
  - Summary: SOAP (sample, operation, attribution and parameter) parallelism. Operator graph, device topology and extution optimizer. MCMC search algorithm and excution simulator.

- Device placement optimization with reinforcement learning
  - Mirhoseini, Azalia, Hieu Pham, Quoc V. Le, Benoit Steiner, Rasmus Larsen, Yuefeng Zhou, Naveen Kumar, Mohammad Norouzi, Samy Bengio, and Jeff Dean. (*ICML 17*)
  - Summary: Using REINFORCE learn a device placement policy. Group operations to excute. Need a lot of GPUs.

- Spotlight: Optimizing device placement for training deep neural networks
  - Gao, Yuanxiang, Li Chen, and Baochun Li (*ICML 18*)
  - Summary:

## System Optimization

- Resource management with deep reinforcement learning 
  - Mao, Hongzi, Mohammad Alizadeh, Ishai Menache, and Srikanth Kandula (*ACM HotNets 2016*)
  - Summary:
  
## Deep Reinforcement Learning System

- Ray: A Distributed Framework for Emerging {AI} Applications [[GitHub]](https://www.usenix.org/conference/osdi18/presentation/moritz)
  - Moritz, Philipp, et al. (*OSDI 2018*)
  - Summary: Distributed DRL system

## Video System papers

- Live Video Analytics at Scale with Approximation and Delay-Tolerance [[Paper]](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/videostorm_nsdi17.pdf)
  - Zhang, Haoyu, Ganesh Ananthanarayanan, Peter Bodik, Matthai Philipose, Paramvir Bahl, and Michael J. Freedman. (*NSDI 2017*)
  - Summary: 

## Traditional System Optimization Papers

- AutoScale: Dynamic, Robust Capacity Management for Multi-Tier Data Centers
[[Paper]](https://dl.acm.org/citation.cfm?id=2382556)

## Edge or Mobile Papers 

- Lavea: Latency-aware video analytics on edge computing platform [[Paper]](http://www.cs.wayne.edu/~weisong/papers/yi17-LAVEA.pdf)
  - Yi, Shanhe, et al. (*Second ACM/IEEE Symposium on Edge Computing. ACM, 2017.*)
  - Summary:

## Advanced Theory
- Differentiable MPC for End-to-end Planning and Control [[Paper]](https://www.cc.gatech.edu/~bboots3/files/DMPC.pdf)  [[GitHub]](https://locuslab.github.io/mpc.pytorch/)
  - Amos, Brandon, Ivan Jimenez, Jacob Sacks, Byron Boots, and J. Zico Kolter (*NIPS 2018*)
  - Summary:
