# Model_Compression_Paper

  ### Type of Pruning

| Type        |      `F`       |      `W`       |   `Other`   |
| :---------- | :------------: | :------------: | :---------: |
| Explanation | Filter pruning | Weight pruning | other types |

  

| `Conf`    |  `2015`   |  `2016`   |  `2017`   |  `2018`   |   `2019`    |                            `2020`                            |    `2021`    |
| --------- | :-------: | :-------: | :-------: | :-------: | :---------: | :----------------------------------------------------------: | :----------: |
| `AAAI`    |   `539`   |   `548`   |   `649`   |   `938`   |   `1147`    |                            `1591`                            |    `1692`    |
| `CVPR`    | `602(71)` | `643(83)` | `783(71)` | `979(70)` | `1300(288)` |                         `1470(335)`                          | Feb.28(7015) |
| `NeurIPS` |   `479`   |   `645`   |   `954`   |  `1011`   |   `1428`    | [`1900 (105)`](https://mp.weixin.qq.com/s/dDCaBWSwh88dNs7yeKI_ag) |              |
| `ICLR`    |           | `oral-15` |   `198`   | `336(23)` |  `502(24)`  |                            `687`                             |  `860(53)`   |
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

  

###  稀疏化



 2019  SeerNet: Predicting Convolutional Neural Network Feature-Map Sparsity Through Low-Bit Quantization  `CVPR` 



### 量化2021-(3)

| Title                                                        | Venue  | Type |     Notion      |
| :----------------------------------------------------------- | :----: | :--: | :-------------: |
| Any-Precision Deep Neural Networks                           | `AAAI` |      |                 |
| Post-training Quantization with Multiple Points  Mixed Precision without Mixed Precision | `AAAI` |      | Mixed Precision |
| CPT:Efficient deep neural network training via cyclic precision | `ICLR` |      |                 |


### 量化 2020-(14)

| Title                                                        |      Venue       | Type |      Notion      |
| :----------------------------------------------------------- | :--------------: | :--: | :--------------: |
| Precision Gating Improving Neural Network Efficiency with Dynamic Dual-Precision Activations |      `ICLR`      |      |                  |
| **Post-training Quantization** with Multiple Points  Mixed Precision without Mixed Precision |      `ICML`      |      |                  |
| Towards Unified INT8 Training for Convolutional Neural Network |      `CVPR`      |      |    商汤bp+qat    |
| APoT-addive powers-of-two quantization an efficient non-uniform discretization for neural networks |      `ICLR`      |      | 非线性量化scheme |
| **Post-Training** **Piecewise Linear Quantization for Deep Neural Networks** | `ECCV`**(oral)** |      |                  |
| Training Quantized Neural Networks With a Full-Precision Auxiliary Module. | `CVPR`**(oral)** |      |                  |
| MCUNet: Tiny Deep Learning on IoT Devices                    |    `NeurIPS`     |      |                  |
| HAWQ-V2 Hessian Aware trace-Weighted Quantization of Neural Networks |    `NeurIPS`     |      |                  |
| HAWQ-V3: Dyadic Neural Network Quantization                  |       `-`        |      |                  |
| Subtensor Quantization for Mobilenets                        |       `-`        |      |    Mobilenets    |
| Generative Low-bitwidth Data Free Quantization               |      `ECCV`      |      |       GAN        |


---

### 剪枝 2020

| Title                                                        |         Venue         |  Type   |                             Code                             |
| :----------------------------------------------------------- | :-------------------: | :-----: | :----------------------------------------------------------: |
| EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning |   `ECCV`**(Oral)**    |   `F`   | [PyTorch(Author)](https://github.com/anonymous47823493/EagleEye) |
| [DSA: More Efficient Budgeted Pruning via Differentiable Sparsity Allocation](https://arxiv.org/abs/2004.02164) |        `ECCV`         |   `F`   |                                                              |
| AutoCompress: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates |        `AAAI`         |   `F`   |                              -                               |
| Pruning from Scratch                                         |        `AAAI`         | `Other` |                              -                               |
| [DHP: Differentiable Meta Pruning via HyperNetworks](https://arxiv.org/abs/2003.13683) |        `ECCV`         |   `F`   |     [PyTorch(Author)](https://github.com/ofsoundof/dhp)      |
| [Towards Efficient Model Compression via Learned Global Ranking](https://arxiv.org/abs/1904.12368) |   `CVPR`**(Oral)**    |   `F`   |     [Pytorch(Author)](https://github.com/cmu-enyac/LeGR)     |
| HRank: Filter Pruning using High-Rank Feature Map            |   `CVPR`**(Oral)**    |   `F`   |                              可                              |
| Soft Threshold Weight Reparameterization for Learnable Sparsity |        `ICML`         |  `WF`   |      [Pytorch(Author)](https://github.com/RAIVNLab/STR)      |
| Network Pruning by Greedy Subnetwork Selection               |        `ICML`         |   `F`   |                              -                               |
| Operation-Aware Soft Channel Pruning using Differentiable Masks |        `ICML`         |   `F`   |                             Mask                             |


---

### 量化 2019-(22)

| Title                                                        |       Venue        | Type |       Notion       |
| :----------------------------------------------------------- | :----------------: | :--: | :----------------: |
| **ACIQ**-Analytical Clipping for Integer Quantization of Neural Networks |       `ICLR`       |      |                    |
| **Differentiable** Quantization of Deep Neural Networks      |     `NeurIPS`      |      |     没代码+NAS     |
| Post training 4-bit quantization of convolutional networks for rapid-deployment |     `NeurIPS`      |      |      **ACIQ**      |
| **Data-Free Quantization Through Weight Equalization and Bias Correction** |  `ICCV`**(Oral)**  |      |                    |
| Data-Free Quantization Through Weight Equalization and Bias Correction |       `ICCV`       |      |                    |
| **HAWQ**: Hessian AWare Quantization of Neural Networks with Mixed-Precision | `ICCV`(**Poster**) |      |       可微分       |
| **(DSQ)**Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks |       `ICCV`       |      |       可微分       |
| Low-bit Quantization of Neural Networks for Efficient Inference |  `ICCV Workshops`  |      |       没代码       |
| **Quantization Networks**                                    |       `CVPR`       |      |       可微分       |
| Fully Quantized Network for **Object Detection**             |       `CVPR`       |      |       没代码       |
| HAQ Hardware-Aware Automated Quantization With Mixed Precision |       `CVPR`       |      |         RL         |
| Accelerating Convolutional Neural Networks via Activation Map Compression |       `CVPR`       |      |       没代码       |
| Learning to quantize deep networks by optimizing quantization intervals with **task loss** |       `CVPR`       |      |       可微分       |
| Accelerating Convolutional Neural Networks via Activation Map Compression |       `CVPR`       |      |   没看懂pipeline   |
| Fighting Quantization Bias With Bias                         |      `CVPR W`      |      | 给量化误差补偿bias |
| Learning low-precision neural networks without Straight-Through Estimator(STE) |      `IJCAI`       |      |       可微分       |
| **OCS**-Improving Neural Network  Quantization without Retraining using **Outlier Channel Splitting.** |       `ICML`       |      |                    |
| Same, Same But Different Recovering Neural Network Quantization Error Through Weight Factorization |       `ICML`       |      |  与高通的DFQ很像   |
| Learning low-precision neural networks without Straight-Through Estimator (STE) |      `IJCAI`       |      |   没代码+可微分    |
| SeerNet Predicting Convolutional Neural Network Feature-Map Sparsity Through Low-Bit Quantization |       `ECCV`       |      |      `稀疏化`      |
| DAC  Data-free Automatic Acceleration of Convolutional Networks |       `WACV`       |      |     `DW Conv`      |
| A Quantization-Friendly Separable Convolution for MobileNets |        `-`         |      |     MobileNets     |


---

  ### 剪枝 2019

| Title                                                        |       Venue       | Type |     Notion     |
| :----------------------------------------------------------- | :---------------: | :--: | :------------: |
| The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks | `ICLR`**(Best)**  | `W`  | winning ticket |
| Rethinking the Value of Network Pruning                      |      `ICLR`       | `F`  |   slim prune   |
| Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration | `CVPR` **(Oral)** | `F`  | 基于几何平均数 |
| Importance Estimation for Neural Network Pruning             |      `CVPR`       | `F`  |    `Nvidia`    |
| Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure |      `CVPR`       | `F`  |      聚类      |
  

---

### 量化 2018-(11)

| Title                                                        |   Venue   | Type |      Notion      |
| :----------------------------------------------------------- | :-------: | :--: | :--------------: |
| **PACT: Parameterized Clipping Activation for Quantized Neural Networks** |  `ICLR`   |      |                  |
| Scalable methods for 8-bit training of neural networks       | `NeurIPS` |      |                  |
| Two-step quantization for low-bit neural networks            |  `CVPR`   |      |                  |
| **Quantization and Training of Neural Networks for Efﬁcient Integer-Arithmetic-Only Inference** |  `CVPR`   |      | **QAT和fold Bn** |
| Joint training of low-precision neural network with quantization interval Parameters | `NeurIPS` |      |     samsung      |
| **Lq-nets** Learned quantization for highly accurate and compact deep neural networks |  `ECCV`   |      |                  |
| Apprentice Using KD Techniques to Improve Low-Precision Network Accuracy |  `ICLR`   |      |                  |
| calable Methods for 8-bit Training of Neura Network          | `NeurIPS` |      |                  |
| Quantization mimic  Towards very tiny cnn for object detection |  `ECCV`   |      |     KD+量化      |
| Mimicking very efficient network for object detection        |  `CVPR`   |      |      跟上面      |
| Training and inference with integers in deep neural networks |  `ICLR`   |      |      `WAGE`      |


---

### 剪枝 2018

| Title                                                        |   Venue   | Type |        Notion         |
| :----------------------------------------------------------- | :-------: | :--: | :-------------------: |
| Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers |  `ICLR`   | `F`  | ISAT+质疑了norm-based |
| A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers |  `ECCV`   | `w`  |         ADMM          |
| Amc: Automl for model compression and acceleration on mobile devices |  `ECCV`   | `F`  |      **还没看**       |
| Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks |  `IJCAI`  | `F`  |   剪枝后还可以恢复    |
| Data-Driven Sparse Structure Selection for Deep Neural Networks |  `ECCV`   | `F`  |        APG +Bn        |

---

  ### 剪枝 2017

| Title                                                        |   Venue   | Type |        Notion         |
| ------------------------------------------------------------ | :-------: | :--: | :-------------------: |
| Pruning Filters for Efficient ConvNets                       |  `ICLR`   | `F`  |      abs(filter)      |
| Pruning Convolutional Neural Networks for Resource Efficient Inference |  `ICLR`   | `F`  | 基于一阶泰勒展开近似  |
| ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression |  `ICCV`   | `F`  | 找一组channel近似全集 |
| Channel pruning for accelerating very deep neural networks   |  `ICCV`   | `F`  |    LASSO回归、孙剑    |
| Learning Efficient Convolutional Networks Through Network Slimming |  `ICCV`   | `F`  |       基于BN层        |
| Runtime Neural Pruning                                       | `NeurIPS` |      |       Markov+RL       |
| Network trimming  A data-driven neuron pruning approach towards efficient deep architectures | `NeurIPS` |      |         APoZ          |


---

### 量化2015 & 2016 & 2017-(8)

| Title                                                        |   Venue   | Type |          Notion           |
| :----------------------------------------------------------- | :-------: | :--: | :-----------------------: |
| **HWGQ**-Deep Learning With Low Precision by Half-wave Gaussian Quantization | ``CVPR``  |      |           孙剑            |
| **Weighted-Entropy-based** Quantization for Deep Neural Networks |  `CVPR`   |      |        `not code`         |
| **WRPN** Wide Reduced-Precision Networks                     |  `ICLR`   |      | `intel`+distiller框架集成 |
| **DoReFa-Net:** training low bitwidth convolutional neural networks with low bitwidth gradients |  `ICLR`   |      |          超低bit          |
| **XNOR-Net:** ImageNet Classification Using Binary Convolutional Neural Networks |  `ECCV`   |      |          超低bit          |
| **Binaryconnect** Training deep neural networks with binary weights during propagations | `NeurIPS` |      |          超低bit          |
| **INQ**-Incremental network quantization Towards lossless cnns with low-precision weight |  `ICLR`   |      |          `intel`          |
| Convolutional Neural Networks using Logarithmic Data Representation |  `ICML`   |      |          scheme           |
