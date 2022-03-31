# Memory-Augmented Deep Unfolding Network for Compressive Sensing (ACM MM 2021)
This repository is for MADUN introduced in the following paper：

Jiechong Song, Bin Chen and [Jian Zhang](http://jianzhang.tech/), "Memory-Augmented Deep Unfolding Network for Compressive Sensing ", in the 29th ACM International Conference on Multimedia (ACM MM), 2021. [PDF](https://arxiv.org/abs/2110.09766)

## Requirements

- Python == 3.8
- Pytorch == 1.8.0
- numpy
- opencv
- skimage
- scipy

## Introduction

Mapping a truncated optimization method into a deep neural network, deep unfolding network (DUN) has attracted growing attention in compressive sensing (CS) due to its good interpretability and high performance. Each stage in DUNs corresponds to one iteration in optimization. By understanding DUNs from the perspective of the human brain’s memory processing, we find there exists two issues in existing DUNs. One is the information between every two adjacent stages, which can be regarded as short-term memory, is usually lost seriously. The other is no explicit mechanism to ensure that the previous stages affect the current stage, which means memory is easily forgotten. To solve these issues, in this paper, a novel DUN with persistent memory for CS is proposed, dubbed Memory-Augmented Deep Unfolding Network (MADUN). We design a memory-augmented proximal mapping module (MAPMM) by combining two types of memory augmentation mechanisms, namely High-throughput Short-term Memory (HSM) and Cross-stage Long-term Memory (CLM). HSM is exploited to allow DUNs to transmit multi-channel short-term memory, which greatly reduces information loss between adjacent stages. CLM is utilized to develop the dependency of deep information across cascading stages, which greatly enhances network representation capability. Extensive CS experiments on natural and MR images show that with the strong ability to maintain and balance information our MADUN outperforms existing state-of-the-art methods by a large margin. 

![PMM_MAPMM](https://github.com/songjiechong/MADUN-ACMMM2021/blob/master/Fig/PMM_MAPMM.png)

## Dataset

### Train data

[train400](https://drive.google.com/file/d/15FatS3wYupcoJq44jxwkm6Kdr0rATPd0/view?usp=sharing)

### Test data

Set11

CBSD68 

Urban100

## Citation

If you find our work helpful in your resarch or work, please cite the following paper.

```
@inproceedings{song2021memory,
  title={Memory-Augmented Deep Unfolding Network for Compressive Sensing},
  author={Song, Jiechong and Chen, Bin and Zhang, Jian},
  booktitle={Proceedings of the ACM International Conference on Multimedia (ACM MM)},
  year={2021}
}
```
