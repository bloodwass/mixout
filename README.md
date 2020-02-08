# Implementation of Mixout with PyTorch
This repository contains a PyTorch code of mixout. This technique regularizes learning to minimize the deviation from the target parameters. For more detailed description of mixout, see "Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models" [[link]](https://arxiv.org/abs/1909.11299).       

# How to use
There is an example code (**example.py**) about applying mixout to a model. In **mixout.py**, you can find the functional version of mixout similar to torch.nn.functional.dropout(p). The module version of mixout is available in **module.py** as well, but it is quite different compared to torch.nn.Dropout(p). I highly recommend users to read **example.py**.   

# Reference
Cheolhyoung Lee, Kyunghyun Cho, and Wanmo Kang, Mixout: Effective regularization to Finetune Large-scale Pretrained Language Models, _International Conference on Learning Representations_ (2020).
