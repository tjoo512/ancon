# Anchored Confidence (AnCon)

This repository is the official implementation of Improving self-training under distribution shifts via anchored confidence with theoretical guarantees (NeurIPS 2024).


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Pretraining a model on the source domain

To pretrain the model(s) on the source domain, run ```erm_source_only.py```. 

For example, to train ResNet-50 on Rw domain of the OfficeHome, run the following command:
```pretrain
python erm_source_only.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 50 -i 500  --log logs/benchmark_erm/OfficeHome_Rw 
```

## Training

To train the model(s) in the paper, run ```train.py```. 

For example, to implement AnCon for adaptation on the target domain Pr from a ResNet-50 pretrained on Rw domain

this command:

```train
python train.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --lr 1e-2 --wd 1e-3 --seed 2024 --epochs 30  --log logs/benchmark_oh/OfficeHome_Pr2Rw --ancon 
```

## TOOD
- Add evaluation code
- Modify README file


## Reference 

Our code is based on the following public repository:

* Tllib: https://github.com/thuml/Transfer-Learning-Library