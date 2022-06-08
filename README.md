# Uncertainty-Aware Pseudo-labeling for Quantum Calculations

This repository hosts the code for 

**Uncertainty-Aware Pseudo-labeling for Quantum Calculations**

*By Kexin Huang, Mykola Bordyuh, Vishnu Sresht, Brajesh Rai.*


## Install

0. Create a conda environment:

```bash
conda create -n Pseudo python=3.8
conda activate Pseudo
```

1. Git clone the repo:

```bash
git clone https://github.com/PfizerRD/pseudo.git 
cd pseudo
```

2. Install [PyTorch](https://pytorch.org/) & [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

3. Install other libraries via

```bash
pip install -r requirements.txt
```

## Dataset
Please download the processed dataset through this [link](https://drive.google.com/file/d/1uF5nNgd3mtm-2uUwLdXfVUFI1hSCi1fw/view?usp=sharing). Unzip it and place the folder "dataset" in the repository.

## Train

```bash
python train.py --label homo \ # molecule target
                --setting low_data \ # low data setting or standard fully supervised setting
                --training_fraction 0.01 \ # fraction of training QM9, rest is used as unlabeled
                --pseudo_label True \ # whether or not to use pseudo-label or standard
                --model dimenet \ # model backbone, select from schnet/dimenet
                --iteration 10 \ # outer loop number of episodes
                --epoch 100 \ # inner loop number of epochs for each episode
                --initial_train_epoch 100 \ # Number of training epochs for the first episode on labeled data
                --batch_size 128 \ # batch size
                --lr 0.001 \ # learning rate
                --lr_decay_factor 0.5 \ # step LR decay factor
                --lr_decay_step_size 25 \ # steps to decay LR
                --uncertainty_type epistemic \ # uncertainty type select from epistemic and aleatoric
                --evi_lambda 0.5 \ # evidental loss regularization coefficient                
```

For more options, checkout the parser in `train.py`. Also checkout a demo test in the notebook `demo.ipynb`.

To reproduce full data setting:

<details>
  <summary>Click here for the code!</summary>

```bash
python train.py --label homo \
                --model dimenet \
                --pseudo_label True \
                --setting standard \
                --pseudo_ensemble True \
                --lr 0.001 \
                --lr_decay_factor 0.5 \
                --lr_decay_step_size 25 \
                --iteration 15 \
                --evi_lambda 0.5 \
                --epoch 75 \
                --batch_size 128 \
                --uncertainty_type epistemic

```
    
</details>

To reproduce low data setting, replace the `training_fraction` value with the low data fraction you consider:

<details>
  <summary>Click here for the code!</summary>
    
```bash
python train.py --label homo \
                --model dimenet \
                --pseudo_label True \
                --setting low_data \
                --training_fraction 0.1 \
                --initial_train_epoch 300 \
                --pseudo_ensemble True \
                --lr 0.001 \
                --lr_decay_factor 0.5 \
                --lr_decay_step_size 15 \
                --iteration 15 \
                --evi_lambda 0.5 \
                --epoch 50 \
                --batch_size 128 \
                --uncertainty_type epistemic

```
    
</details>


## Citation

```
@inproceedings{huang2022uncertainty,
  title={Uncertainty-Aware Pseudo-labeling for Quantum Calculations},
  author={Huang, Kexin and Sresht, Vishnu and Rai, Brajesh and Bordyuh, Mykola},
  booktitle={The 38th Conference on Uncertainty in Artificial Intelligence},
  year={2022}
}
```

## Acknowledgement

We use [DIG](https://github.com/divelab/DIG) library as the backbone for SchNet and DimeNet++ implementation. 