# </center> VULNERABILITY AND DEFENSE: MITIGATING BACKDOORATTACKS IN DEEP LEARNING-BASED CROWD COUNTING MODELS </center>

## </cneter> Abstract </center>

Crowd counting aims to infer the number of people or objects in an image through different methods. It is widely used in surveillance, sensitive events, etc., and plays a vital role in a series of security- critical applications. Most of the state-of-the-art crowd counting models are based on deep learning, which are very efficient and accurate in handling dense scenes. Although such models are effective, they are still vulnerable to backdoor attacks. Attackers can compromise model accuracy by poisoning
surveillance data or using global triggers, leading to inaccurate crowd counts. In this paper, we verify
the vulnerability of deep learning-based crowd counting models to backdoor attacks and prove the
effectiveness of density manipulation attacks on two different types of crowd counting models. At
the same time, a defense method similar to fine-tuning is proposed based on this backdoor attack. Through in-depth analysis, we observe that our defense method not only reduces the effectiveness
of backdoor attacks – the attack success rate ρAsr by 72.5%, but also improves the accuracy of the
original model’s prediction – the accuracy ρAcc by 66.5%. Our work can help eliminate potential
backdoor attacks on crowd counting models.

<div align="center">
  <img src="fig/all.png" alt="风景图片">
  <p>Figure 1：The general framework of our work. Step 1, we create poisoned data and train clean models and models with backdoors. Step 2, we verify the effect of the attack under models with different backdoor strategies. Step 3, we select new clean data and the originalclean data to train the attacked model together, and then obtain a fine-tuned model. After that We input poisoned data to verify the effectiveness of our defense method, and also test the effect of the model on the clean data set.</p>
</div>

## Requirement
Follow the requirements of the targeted models

## Dataset
ShanghaiTech Dataset: https://www.kaggle.com/datasets/tthien/shanghaitech

## The Targeted Models
CSRNet: https://github.com/CommissarMa/CSRNet-pytorch

CLIP-EBC: https://github.com/Yiming-M/CLIP-EBC

## Injection Trigger & Density Map altering
For ground-truth, using the original methods to processing images.

Using the dmap_for_SHHB_poisoned.py in CSRnet/data/part_B_final file to generate posioned densitymap

## Visualization
Using the gradcam.py in CSRnet file to visualize.

## References

CSRnet
```
@inproceedings{li2018csrnet,
  title={CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes},
  author={Li, Yuhong and Zhang, Xiaofan and Chen, Deming},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1091--1100},
  year={2018}
}
```


CLIP-EBC
```
@article{ma2024clip,
title={CLIP-EBC: CLIP Can Count Accurately through Enhanced Blockwise Classification},
author={Ma, Yiming and Sanchez, Victor and Guha, Tanaya},
journal={arXiv preprint arXiv:2403.09281},
year={2024}
}
```


ShanghaiTech Datasets
```
@inproceedings{zhang2016single,
  title={Single-image crowd counting via multi-column convolutional neural network},
  author={Zhang, Yingying and Zhou, Desen and Chen, Siqin and Gao, Shenghua and Ma, Yi},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={589--597},
  year={2016}
}
```
