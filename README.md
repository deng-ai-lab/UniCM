# A Unified Framework for Consistency Generative Modeling

This repository is the official implementation of *A Unified Framework for Consistency Generative Modeling.*

## Requirements

```setup
pip install -r requirements.txt
```

## Training



1) Place the downloaded dataset in ./data .

2) Configure hyperparameters in ./config. 

3) To train the model(s) in the paper, run this command:

#### CIFAR-10 

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

#### CelebA

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

#### AFHQ for Im2Im

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Testing


3) To test the model(s) and calculate the metrics in the paper, run this command:

```test
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```
