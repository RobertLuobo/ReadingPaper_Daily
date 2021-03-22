# Model_Compression_Paper

  ### Type of Pruning

| Type        |      `F`       |      `W`       |   `Other`   |
| :---------- | :------------: | :------------: | :---------: |
| Explanation | Filter pruning | Weight pruning | other types |

  

| `Conf`    |  `2015`   |  `2016`   |  `2017`   |  `2018`   |   `2019`    |                            `2020`                            |    `2021`    |
| --------- | :-------: | :-------: | :-------: | :-------: | :---------: | :----------------------------------------------------------: | :----------: |
| `AAAI`    |   `539`   |   `548`   |   `649`   |   `938`   |   `1147`    |                            `1591`                            |    `1692`    |
| `ICLR`    |           | `oral-15` |   `198`   | `336(23)` |  `502(24)`  |                            `687`                             |  `860(53)`   |
| `CVPR`    | `602(71)` | `643(83)` | `783(71)` | `979(70)` | `1300(288)` |                         `1470(335)`                          | Feb.28(7015) |
| `NeurIPS` |   `479`   |   `645`   |   `954`   |  `1011`   |   `1428`    | [`1900 (105)`](https://mp.weixin.qq.com/s/dDCaBWSwh88dNs7yeKI_ag) |              |
| `ICML`    |           |           |   `433`   |   `621`   |    `774`    |                            `1088`                            |   May.8th    |
| `IJCAI`   |   `572`   |   `551`   |   `660`   |   `710`   |    `850`    |                            `592`                             |              |
| `ICCV`    |           |    `-`    |   `621`   |    `-`    |    1077     |                             `-`                              |              |
| `ECCV`    |           |   `415`   |    `-`    |   `778`   |     `-`     |                            `1360`                            |              |
| `MLsys`   |           |           |           |           |             |                                                              |              |
| `ISCA`    |    57     |    54     |    54     |    63     |     62      |                              77                              |              |
| `ECAI`    |    `-`    |    562    |    `-`    |    656    |     `-`     |                             365                              |              |

  

  `MLsys`:https://proceedings.mlsys.org/paper/2019

  `ICCV`https://dblp.org/db/conf/iccv/iccv2019.html

  `ICCV` https://dblp.org/db/conf/iccv/iccv2017.html

  `ECCV` https://link.springer.com/conference/eccv

  `ECCV` https://zhuanlan.zhihu.com/p/157569669

  `CVPR` https://dblp.org/db/conf/cvpr/cvpr2020.html


  `ICDE`  `ACCV` `WACV` `BMVC`

  `WACV`:(Applications of Computer Vision)

`nsdi`   `sigcomm`   `osdi`   `sosp`   `sigmod`   `mobicom`   `sosp`   `ATC`   `MLsys`  

  
 ####  2021:

1. Diversifying Sample Generation for Accurate Data-Free Quantization  
2. BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction
3. Learnable Companding Quantization for Accurate Low-bit Neural Networks   
4. UMEC: Unified Model and Embedding Compression for Efficient Recommendation Systems
5. Distribution Adaptive INT8 Quantization for Training CNNs
6. Refine Myself by Teaching Myself: Feature Refinement via Self-Knowledge Distillation  
7. You Only Look One-level Feature  
8. Probabilistic two-stage detection
9. General Instance Distillation for Object Detection   
10. Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection   
11. Sparse R-CNN: End-to-End Object Detection with Learnable Proposals  
12. Relation Networks for Object Detection  
13. RelationNet++: Bridging Visual Representations for Object Detection via Transformer Decoder   
14. Weighted boxes fusion: Ensembling boxes from different object detection models
15. Dert：End-to-End Object Detection with Transformers    
16. Fine-grained Angular Contrastive Learning with Coarse Labels(😮oral)     使用自监督进行 Coarse Labels（粗标签）的细粒度分类方面的工作。粗标签与细粒度标签相比，更容易和更便宜，因为细粒度标签通常需要域专家。   
17. UP-DETR: Unsupervised Pre-training for Object Detection with Transformers(oral)   
18. ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network  
19. RepVGG: Making VGG-style ConvNets Great Again   
20. Revisiting Dynamic Convolution via Matrix Decomposition
21. An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection
22. CenterMask : Real-Time Anchor-Free Instance Segmentation
23. VarGNet: Variable Group Convolutional Neural Network for Efficient Embedded Computing
24. Manifold Regularized Dynamic Network Pruning   
25. Fast and Accurate Model Scaling   
26. Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions
27. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
28. On Implicit Filter Level Sparsity in Convolutional Neural Networks



#####  稀疏化

 2019  SeerNet: Predicting Convolutional Neural Network Feature-Map Sparsity Through Low-Bit Quantization  `CVPR`      



##### 量化2021-(3)

1.  Any-Precision Deep Neural Networks                             `AAAI` 
2.  Post-training Quantization with Multiple Points  Mixed Precision without Mixed Precision | `AAAI`   `Mixed Precision`
3.  CPT:Efficient deep neural network training via cyclic precision   `ICLR`                  



#### 量化 2020-(14)

1.  Precision Gating Improving Neural Network Efficiency with Dynamic Dual-Precision Activations       `ICLR`     
2.  **Post-training Quantization** with Multiple Points  Mixed Precision without Mixed Precision      `ICML`     
3.  Towards Unified INT8 Training for Convolutional Neural Network |      `CVPR`        `商汤bp+qat`   
4.  APoT-addive powers-of-two quantization an efficient non-uniform discretization for neural networks       `ICLR`             `非线性量化scheme`
5.  **Post-Training** **Piecewise Linear Quantization for Deep Neural Networks**  `ECCV`**(oral)**   
6.  Training Quantized Neural Networks With a Full-Precision Auxiliary Module  `CVPR`**(oral)**  
7.  MCUNet: Tiny Deep Learning on IoT Devices                        `NeurIPS`      
8.  HAWQ-V2 Hessian Aware trace-Weighted Quantization of Neural Networks     `NeurIPS`    
9.  HAWQ-V3: Dyadic Neural Network Quantization                    
10.  Subtensor Quantization for Mobilenets                               `-`                 `Mobilenets`  
11.  Generative Low-bitwidth Data Free Quantization                     `ECCV`                   `GAN `          





##### 剪枝 2020

1.  EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning    `ECCV`**(Oral)**       `F`    [PyTorch(Author)](https://github.com/anonymous47823493/EagleEye) 
    [DSA: More Efficient Budgeted Pruning via Differentiable Sparsity Allocation](https://arxiv.org/abs/2004.02164)         `ECCV`            `F`                             
2.  AutoCompress: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates         `AAAI`            `F`                                -                 
3.  Pruning from Scratch                                                 `AAAI`          `Other`                               -   
4.  [DHP: Differentiable Meta Pruning via HyperNetworks](https://arxiv.org/abs/2003.13683)         `ECCV`            `F`        [PyTorch(Author)](https://github.com/ofsoundof/dhp)      
5.  [Towards Efficient Model Compression via Learned Global Ranking](https://arxiv.org/abs/1904.12368)    `CVPR`**(Oral)**       `F`        [Pytorch(Author)](https://github.com/cmu-enyac/LeGR)     
6.  HRank: Filter Pruning using High-Rank Feature Map               `CVPR`**(Oral)**       `F`                                 可                              
7.  Soft Threshold Weight Reparameterization for Learnable Sparsity         `ICML`           `WF`         [Pytorch(Author)](https://github.com/RAIVNLab/STR)   
8.  Network Pruning by Greedy Subnetwork Selection                       `ICML`            `F`                                 -                               
9.  Operation-Aware Soft Channel Pruning using Differentiable Masks         `ICML`            `F`                                Mask                             





##### 量化 2019-(22)

1.  **ACIQ**-Analytical Clipping for Integer Quantization of Neural Networks       `ICLR`      

2.  **Differentiable** Quantization of Deep Neural Networks         `NeurIPS`        `没代码+NAS`    

3.  Post training 4-bit quantization of convolutional networks for rapid-deployment      `NeurIPS`            **ACIQ**   

4.  **Data-Free Quantization Through Weight Equalization and Bias Correction**   `ICCV`**(Oral)**                  

5.  Data-Free Quantization Through Weight Equalization and Bias Correction        `ICCV`                   

6.  **HAWQ**: Hessian AWare Quantization of Neural Networks with Mixed-Precision  `ICCV`(**Poster**)     `可微分`       

7.  **(DSQ)**Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks        `ICCV`                   `可微分`      

8.  Low-bit Quantization of Neural Networks for Efficient Inference  `ICCV Workshops`         `没代码`       

9.  **Quantization Networks**                                          `CVPR`                  可微分       

10.  Fully Quantized Network for **Object Detection**                  `CVPR`                 没代码       

11.  HAQ Hardware-Aware Automated Quantization With Mixed Precision        `CVPR`                     `RL`         

12.  Accelerating Convolutional Neural Networks via Activation Map Compression        `CVPR`             `没代码`     

13.  Learning to quantize deep networks by optimizing quantization intervals with **task loss**        `CVPR`              `可微分`     

14.  Accelerating Convolutional Neural Networks via Activation Map Compression        `CVPR`         `没看懂pipeline`   

15.  Fighting Quantization Bias With Bias                               `CVPR W`       给量化误差补偿bias 

16.  Learning low-precision neural networks without Straight-Through Estimator(STE)       `IJCAI`         `可微分`       

17.  **OCS**-Improving Neural Network  Quantization without Retraining using **Outlier Channel Splitting.** |       `ICML`                      

18.  Same, Same But Different Recovering Neural Network Quantization Error Through Weight Factorization        `ICML`       ` 与高通的DFQ很像`  

19.  Learning low-precision neural networks without Straight-Through Estimator (STE)       `IJCAI`          `没代码+可微分`    

20.  SeerNet Predicting Convolutional Neural Network Feature-Map Sparsity Through Low-Bit Quantization        `ECCV`                   `稀疏化`     

21.  DAC  Data-free Automatic Acceleration of Convolutional Networks        `WACV`           `DW Conv`      

22.  A Quantization-Friendly Separable Convolution for MobileNets         `-`             `MobileNets`    




##### 剪枝 2019

1.  The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks  `ICLR`**(Best)**   `W`   `winning ticket`

2.  Rethinking the Value of Network Pruning                            `ICLR`        `F`     slim prune   

3.  Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration  `CVPR` **(Oral)**   `F`  `基于几何平均数`  

4.  Importance Estimation for Neural Network Pruning                   `CVPR`        `F`      `Nvidia`    

5.  Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure       `CVPR`        `F`        `聚类`      

   



##### 量化 2018-(11)

1.  **PACT: Parameterized Clipping Activation for Quantized Neural Networks** |  `ICLR`    
2.  Scalable methods for 8-bit training of neural networks       | `NeurIPS`    
3.  Two-step quantization for low-bit neural networks            |  `CVPR`    
4.  **Quantization and Training of Neural Networks for Efﬁcient Integer-Arithmetic-Only Inference** |  `CVPR`     `**QAT和fold Bn**` 
5.  Joint training of low-precision neural network with quantization interval Parameters | `NeurIPS`   
6.  **Lq-nets** Learned quantization for highly accurate and compact deep neural networks |  `ECCV`     
7.  Apprentice Using KD Techniques to Improve Low-Precision Network Accuracy |  `ICLR`     
8.  calable Methods for 8-bit Training of Neura Network          | `NeurIPS` |      |                  
9.  Quantization mimic  Towards very tiny cnn for object detection |  `ECCV`   |      |     KD+量化    
10.  Mimicking very efficient network for object detection        |  `CVPR`   |      |      跟上面      
11.  Training and inference with integers in deep neural networks |  `ICLR`   |      |      `WAGE`      



##### 剪枝 2018

1.  Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers |  `ICLR`   | `F`  | ISAT+质疑了norm-based  
2.  A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers |  `ECCV`   | `w`  |         ADMM          
3.  Amc: Automl for model compression and acceleration on mobile devices |  `ECCV`   | `F`  |      **还没看**       
4.  Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks |  `IJCAI`  | `F`  |   剪枝后还可以恢复    
5.  Data-Driven Sparse Structure Selection for Deep Neural Networks |  `ECCV`   | `F`  |        APG +Bn      




  ##### 剪枝 2017

1.  Pruning Filters for Efficient ConvNets                       |  `ICLR`   | `F`  |      abs(filter)     
2.  Pruning Convolutional Neural Networks for Resource Efficient Inference |  `ICLR`   | `F`  | 基于一阶泰勒展开近似  
3.  ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression |  `ICCV`   | `F`  | 找一组channel近似全集  
4.  Channel pruning for accelerating very deep neural networks   |  `ICCV`   | `F`  |    LASSO回归、孙剑     
5.  Learning Efficient Convolutional Networks Through Network Slimming |  `ICCV`   | `F`  |       基于BN层        
6.  Runtime Neural Pruning                                       | `NeurIPS` |      |       Markov+RL   
7.  Network trimming  A data-driven neuron pruning approach towards efficient deep architectures | `NeurIPS` |      |         APoZ         






##### 量化2015 & 2016 & 2017-(8)

1.  **HWGQ**-Deep Learning With Low Precision by Half-wave Gaussian Quantization | ``CVPR``  |      |           孙剑            
2.  **Weighted-Entropy-based** Quantization for Deep Neural Networks |  `CVPR`   |      |        `not code`         |
3.  **WRPN** Wide Reduced-Precision Networks                     |  `ICLR`   |      | `intel`+distiller框架集成 |
4.  **DoReFa-Net:** training low bitwidth convolutional neural networks with low bitwidth gradients |  `ICLR`   |      |          超低bit          
5.  **XNOR-Net:** ImageNet Classification Using Binary Convolutional Neural Networks |  `ECCV`   |      |          超低bit          
6.  **Binaryconnect** Training deep neural networks with binary weights during propagations | `NeurIPS` |      |          超低bit     
7.  **INQ**-Incremental network quantization Towards lossless cnns with low-precision weight |  `ICLR`   |      |          `intel`  
8.  Convolutional Neural Networks using Logarithmic Data Representation |  `ICML`   |      |          scheme           

