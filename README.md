# Adaptive Pseudo-labeling for Quantum Calculations


This repository hosts the code for 

Pseudo-Sigma: adaptive pseudo-labeling fo quantum calculations by Kexin Huang, Mykola Bordyuh, Vishnu Sresht, Brajesh Rai.


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
python train.py --label mu \ # molecule target
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

## Contact

Feel free to open an issue or send emails to [Kexin](kexinh@stanford.edu).

## Acknowledgement

We use [DIG](https://github.com/divelab/DIG) library as the backbone for SchNet and DimeNet++ implementation. 