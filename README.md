# A Unified Framework for Consistency Generative Modeling

This repository is the official implementation of *A Unified Framework for Consistency Generative Modeling.*

## Description
Consistency modeling, a novel generative paradigm inspired by diffusion models, has gained traction for its capacity to facilitate real-time generation through
single-step sampling. While its advantages are evident, the understanding of its
underlying principles and effective algorithmic enhancements remain elusive. In
response, we present a unified framework for consistency generative modeling,
without resorting to the predefined diffusion process. Instead, it directly constructs a probability density path that bridges the two distributions. Building upon
this novel perspective, we introduce a more general consistency training objective
that encapsulates previous consistency models and paves the way for innovative,
consistency generation techniques. In particular, we introduce two novel models:
Poisson Consistency Models (PCMs) and Coupling Consistency Models (CCMs),
which extend the prior distribution of latent variables beyond the Gaussian form.
This extension significantly augments the flexibility of generative modeling. Furthermore, we harness the principles of Optimal Transport (OT) to mitigate variance during consistency training, substantially improving convergence and generative quality. Extensive experiments on the generation of synthetic and real-world
datasets, as well as image-to-image translation tasks (I2I), demonstrate the effectiveness of the proposed approaches.

## Requirements

```setup
pip install -r requirements.txt
```

## Training



1) Place the downloaded dataset in ./data.

2) Configure hyperparameters in ./config. 

3) To train the model(s) in the paper, run this command:

#### CIFAR-10 

```train
python main.py --model DCM|DCM-MS|CCM|CCM-OT|PCM --data Cifar10
```

#### CelebA

```train
python main.py --model DCM|DCM-MS|CCM|CCM-OT|PCM --data Celeba
```

#### AFHQ for Im2Im

```train
python main.py --model CCM|CCM-OT --data AFHQ --task Im2Im
```

## Testing

1) Place reference samples in ./assets
 
2) To test the model(s) and calculate the metrics in the paper, run this command:

```test
python main.py --model DCM|DCM-MS|CCM|CCM-OT|PCM --data Cifar10 --train False 
```
