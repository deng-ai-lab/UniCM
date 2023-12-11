# A Unified Framework for Consistency Generative Modeling

This repository is the official implementation of *A Unified Framework for Consistency Generative Modeling.*

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
