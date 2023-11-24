# Memory-Augmented Deep Unfolding Network for Compressive Sensing (ACM MM 2021)
This repository is for MADUN introduced in the following paper：

[Jiechong Song](https://scholar.google.com/citations?hl=en&user=EBOtupAAAAAJ), Bin Chen and [Jian Zhang](http://jianzhang.tech/), "Memory-Augmented Deep Unfolding Network for Compressive Sensing ", in the 29th ACM International Conference on Multimedia (ACM MM), 2021. [PDF](https://arxiv.org/abs/2110.09766)

## 🚩 News(2023-11-24)
### ✅ 2023-11-24
  
  We release [MAPUN code](https://github.com/songjiechong/MADUN-ACMMM2021/tree/main/MAPUN). And the test command is `python TEST_CS_MAPUN.py --cs_ratio 10/25/30/40/50 --test_name Set11/CBSD68/Urban100`

  
### ✅ 2023-3-2
  
  Our extended version has been accepted by IJCV (International Journal of Computer Vision)!
  
  [Jiechong Song](https://scholar.google.com/citations?hl=en&user=EBOtupAAAAAJ), Bin Chen and [Jian Zhang](http://jianzhang.tech/), "Deep Memory-Augmented Proximal Unrolling Network for Compressive Sensing", in the International Journal of Computer Vision (IJCV), 2023. [PDF](https://link.springer.com/article/10.1007/s11263-023-01765-2)

## 🔧 Requirements
- Python == 3.8
- Pytorch == 1.8.0

## :art: Abstract
Mapping a truncated optimization method into a deep neural network, deep unfolding network (DUN) has attracted growing attention in compressive sensing (CS) due to its good interpretability and high performance. Each stage in DUNs corresponds to one iteration in optimization. By understanding DUNs from the perspective of the human brain’s memory processing, we find there exists two issues in existing DUNs. One is the information between every two adjacent stages, which can be regarded as short-term memory, is usually lost seriously. The other is no explicit mechanism to ensure that the previous stages affect the current stage, which means memory is easily forgotten. To solve these issues, in this paper, a novel DUN with persistent memory for CS is proposed, dubbed Memory-Augmented Deep Unfolding Network (MADUN). We design a memory-augmented proximal mapping module (MAPMM) by combining two types of memory augmentation mechanisms, namely High-throughput Short-term Memory (HSM) and Cross-stage Long-term Memory (CLM). HSM is exploited to allow DUNs to transmit multi-channel short-term memory, which greatly reduces information loss between adjacent stages. CLM is utilized to develop the dependency of deep information across cascading stages, which greatly enhances network representation capability. Extensive CS experiments on natural and MR images show that with the strong ability to maintain and balance information our MADUN outperforms existing state-of-the-art methods by a large margin. 

<img width="1001" alt="PMM_MAPMM" src="https://user-images.githubusercontent.com/62560218/161186801-95d503f6-f2fa-4dcc-8c60-fc80aab65079.png">


## 👀 Datasets
- Train data: [train400](https://drive.google.com/file/d/15FatS3wYupcoJq44jxwkm6Kdr0rATPd0/view?usp=sharing)
- Test data: Set11, [CBSD68](https://drive.google.com/file/d/1Q_tcV0d8bPU5g0lNhVSZXLFw0whFl8Nt/view?usp=sharing), [Urban100](https://drive.google.com/file/d/1cmYjEJlR2S6cqrPq8oQm3tF9lO2sU0gV/view?usp=sharing), [DIV2K](https://drive.google.com/file/d/1olYhGPuX8QJlewu9riPbiHQ7XiFx98ac/view?usp=sharing)

## :computer: Command

### Train
`python Train_CS_MADUN.py --cs_ratio 10/25/30/40/50                  ` 

### Test
`python TEST_CS_MADUN.py --cs_ratio 10/25/30/40/50 --test_name Set11/CBSD68/Urban100`

## 📑 Citation

If you find our work helpful in your resarch or work, please cite the following paper.

```
@inproceedings{song2021memory,
  title={Memory-Augmented Deep Unfolding Network for Compressive Sensing},
  author={Song, Jiechong and Chen, Bin and Zhang, Jian},
  booktitle={Proceedings of the ACM International Conference on Multimedia (ACM MM)},
  year={2021}
}
@article{song2023deep,
  title={Deep Memory-Augmented Proximal Unrolling Network for Compressive Sensing},
  author={Song, Jiechong and Chen, Bin and Zhang, Jian},
  journal={International Journal of Computer Vision},
  pages={1--20},
  year={2023},
  publisher={Springer}
}
```

## :e-mail: Contact
If you have any question, please email `songjiechong@pku.edu.cn`.

## :hugs: Acknowledgements
This code is built on [ISTA-Net-PyTorch](https://github.com/jianzhangcs/ISTA-Net-PyTorch). We thank the authors for sharing their codes.
