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
