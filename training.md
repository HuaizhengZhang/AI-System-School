# Training System

System for deep learning training. Currently, I only summarize some arxiv papers here and put accepted papers into [conference](note) section

## Survey

- Mayer, Ruben, and Hans-Arno Jacobsen. "Scalable Deep Learning on Distributed Infrastructures: Challenges, Techniques, and Tools." ACM Computing Surveys (CSUR) 53.1 (2020): 1-37. [[Paper]](https://arxiv.org/pdf/1903.11314.pdf)

## Training(Multi-jobs on cluster)

- Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning [[arxiv]](https://arxiv.org/pdf/2008.12260.pdf) [[GitHub]](https://github.com/petuum/adaptdl)
  - arXiv preprint arXiv:2008.12260 (2020).
  - Qiao, Aurick, Willie Neiswanger, Qirong Ho, Hao Zhang, Gregory R. Ganger, and Eric P. Xing
- Themis: Fair and Efficient {GPU} Cluster Scheduling. [[Paper]](http://wisr.cs.wisc.edu/papers/nsdi20-themis.pdf)
  - Mahajan, K., Balasubramanian, A., Singhvi, A., Venkataraman, S., Akella, A., Phanishayee, A. and Chawla, S., 2020. Themis: Fair and Efficient {GPU} Cluster Scheduling.
  - In 17th {USENIX} Symposium on Networked Systems Design and Implementation ({NSDI} 20) (pp. 289-304).
- Tiresias: A {GPU} cluster manager for distributed deep learning. [[Paper]](https://www.usenix.org/system/files/nsdi19-gu.pdf) [[GitHub]](https://github.com/SymbioticLab/Tiresias)
  - Gu, J., Chowdhury, M., Shin, K.G., Zhu, Y., Jeon, M., Qian, J., Liu, H. and Guo, C., 2019. 
  - In 16th {USENIX} Symposium on Networked Systems Design and Implementation ({NSDI} 19) (pp. 485-500).
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


## Training(Parallelism)

- ZeRO: Memory Optimization Towards Training A Trillion Parameter Models. *Microsoft Work* [[Paper]](https://arxiv.org/pdf/1910.02054.pdf) [[GitHub]](https://github.com/microsoft/DeepSpeed)
- Class materials for a distributed systems lecture series [[GitHub]](https://github.com/aphyr/distsys-class)
- A Unified Architecture for Accelerating Distributed
DNN Training in Heterogeneous GPU/CPU Clusters [[Paper]](https://www.usenix.org/system/files/osdi20-jiang.pdf) [[GitHub]](https://github.com/bytedance/byteps)
  - Yimin Jiang, Yibo Zhu, Chang Lan, Bairen Yi, Yong Cui, Chuanxiong Guo. (OSDI 2020)
  - Summary: SoTA Parameter Server
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
- GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism [[Paper]](https://arxiv.org/pdf/1811.06965.pdf) [[GitHub]](https://github.com/tensorflow/lingvo/blob/master/lingvo/core/gpipe.py) [[News]](https://www.cnbeta.com/articles/tech/824495.htm)
  - Huang, Yanping, et al. (*arXiv preprint arXiv:1811.06965 (2018)*)
- Horovod: Distributed training framework for TensorFlow, Keras, and PyTorch. 
[[GitHub]](https://github.com/uber/horovod)
- Distributed machine learning infrastructure for large-scale robotics research [[GitHub]](https://github.com/google-research/tensor2robot) [[Blog]](https://ai.google/research/teams/brain/robotics/)
- A Generic Communication Scheduler for Distributed DNN Training Acceleration [[Paper]](https://i.cs.hku.hk/~cwu/papers/yhpeng-sosp19.pdf) [[BytePS]](https://github.com/bytedance/byteps)
  - PENG, Y., Zhu, Y., CHEN, Y., BAO, Y., Yi, B., Lan, C., Wu, C. and Guo, (*SOSP 2019*)
  - Summary: communication schedular


## Training(Fault-tolerant)

- Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates [[Paper]](https://dl.acm.org/doi/abs/10.1145/3600006.3613152) [[GitHub]](https://github.com/SymbioticLab/Oobleck)
  - Jang, Insu and Yang, Zhenning and Zhang, Zhen and Jin, Xin and Chowdhury, Mosharaf, (*SOSP 2023*)
- Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs [[Paper]](https://www.usenix.org/conference/nsdi23/presentation/thorpe) [[GitHub]](https://github.com/uclasystem/bamboo)
  - John Thorpe, Pengzhan Zhao, Jonathan Eyolfson, Yifan Qiao, Zhihao Jia, Minjia Zhang, Ravi Netravali and Guoqing Harry Xu, (*NSDI 2023*)
- Varuna: scalable, low-cost training of massive deep learning models [[Paper]](https://dl.acm.org/doi/abs/10.1145/3492321.3519584) [[GitHub]](https://github.com/microsoft/varuna)
  - Athlur, Sanjith and Saran, Nitika and Sivathanu, Muthian and Ramjee, Ramachandran and Kwatra, Nipun, (*EuroSys 2022*)


## Training(Energy-efficient)

- Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training [[Paper]](https://www.usenix.org/system/files/nsdi23-you.pdf) [[GitHub]](https://github.com/ml-energy/zeus) [[ml.energy]](https://ml.energy/zeus/) [[The ML.ENERGY Initiative]](https://ml.energy/)
  - Jie You, Jae-Won Chung, and Mosharaf Chowdhury (*NSDI 2023*)

