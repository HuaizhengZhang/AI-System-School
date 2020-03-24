# Training System

System for deep learning training.

## Survey

- Mayer, Ruben, and Hans-Arno Jacobsen. "Scalable Deep Learning on Distributed Infrastructures: Challenges, Techniques, and Tools." ACM Computing Surveys (CSUR) 53.1 (2020): 1-37. [[Paper]](https://arxiv.org/pdf/1903.11314.pdf)


## Training(Parallelism)

- ZeRO: Memory Optimization Towards Training A Trillion Parameter Models. *Microsoft Work* [[Paper]](https://arxiv.org/pdf/1910.02054.pdf) [[GitHub]](https://github.com/microsoft/DeepSpeed)
- Class materials for a distributed systems lecture series [[GitHub]](https://github.com/aphyr/distsys-class)
- bytedance/byteps: A high performance and general PS framework for distributed training [[GitHub]](https://github.com/bytedance/byteps)
- PipeDream: Generalized Pipeline Parallelism for DNN Training (SOSP2019) [[Paper]](https://cs.stanford.edu/~matei/papers/2019/sosp_pipedream.pdf) [[Github]](https://github.com/msr-fiddle/pipedream)
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
- Horovod: Distributed training framework for TensorFlow, Keras, and PyTorch. 
[[GitHub]](https://github.com/uber/horovod)
- Distributed machine learning infrastructure for large-scale robotics research [[GitHub]](https://github.com/google-research/tensor2robot) [[Blog]](https://ai.google/research/teams/brain/robotics/)
- A Generic Communication Scheduler for Distributed DNN Training Acceleration [[Paper]](https://i.cs.hku.hk/~cwu/papers/yhpeng-sosp19.pdf) [[BytePS]](https://github.com/bytedance/byteps)
  - PENG, Y., Zhu, Y., CHEN, Y., BAO, Y., Yi, B., Lan, C., Wu, C. and Guo, (*SOSP 2019*)
  - Summary: communication schedular


## Training(Multi-jobs on cluster)

- Microsoft OpenPAI HiveDScheduler: As one standalone component of Microsoft OpenPAI, HiveD is designed to be a Kubernetes Scheduler Extender for Multi-Tenant GPU clusters. [[Project]](https://github.com/microsoft/hivedscheduler)
- Gandiva: Introspective cluster scheduling for deep learning. [[Paper]](https://www.usenix.org/system/files/osdi18-xiao.pdf)
  - Xiao, Wencong, et al. (*OSDI 2018*)
  - Summary: Improvet the efficency of hyper-parameter in cluster. Aware of hardware utilization.
- Optimus: an efficient dynamic resource scheduler for deep learning clusters [[Paper]](https://i.cs.hku.hk/~cwu/papers/yhpeng-eurosys18.pdf)
  - Peng, Yanghua, et al. (*EuroSys 2018*)
  - Summary: Job scheduling on clusters. Total complete time as the metric.
- Multi-tenant GPU clusters for deep learning workloads: Analysis and implications. [[Paper]](https://www.microsoft.com/en-us/research/uploads/prod/2018/05/gpu_sched_tr.pdf) [[dataset]](https://github.com/msr-fiddle/philly-traces)
  - Jeon, Myeongjae, Shivaram Venkataraman, Junjie Qian, Amar Phanishayee, Wencong Xiao, and Fan Yang
- Slurm: A Highly Scalable Workload Manager [[GitHub]](https://github.com/SchedMD/slurm)
