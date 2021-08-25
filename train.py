import os
import random
import numpy as np
import math 

import torch
from torch_geometric.data import DataLoader
import sys

from pseudo.dataset import QM9_3D, PC9_QM9_3D
from pseudo.method import SchNet, DimeNetPP, run
from pseudo.evaluation import threedEvaluator
from argparse import ArgumentParser
parser = ArgumentParser(description='pseudo')

## data
parser.add_argument('--label', default='homo', type=str, choices = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'omega1', 'zpve_thermo', 'U0_thermo', 'U_thermo', 'H_thermo', 'G_thermo', 'Cv_thermo'], help = 'molecular target')
parser.add_argument('--training_fraction', default=0.01, type=float, help = 'fraction of training QM9, rest is used as unlabeled')
parser.add_argument('--setting', default='low_data', choices = ['low_data', 'standard'], type = str, help = 'low data setting or standard fully supervised setting')

## wandb
parser.add_argument('--wandb', default='False', type=str, choices = ['True', 'False'], help = 'wandb mode')
parser.add_argument('--proj_name', default='qm_result', type=str, help = 'wandb project name')
parser.add_argument('--entity_name', default='kexin_pfizer', type=str, help = 'wandb entity name')

## model params
parser.add_argument('--pseudo_label', default='True', type=str, choices = ['True', 'False', 'test'], help = 'whether or not to use pseudo-label or standard')
parser.add_argument('--model', default='schnet', type=str, choices = ['schnet', 'dimenet'], help = 'model type')
parser.add_argument('--iteration', default=10, type = int, help = 'outer loop number of episodes')
parser.add_argument('--epoch', default=100, type = int, help = 'inner loop number of epochs for each episode')
parser.add_argument('--initial_train_epoch', default=100, type = int, help = 'Number of training epochs for the first episode on labeled data')
parser.add_argument('--batch_size', default=128, type = int, help = 'batch size')
parser.add_argument('--lr', default=0.001, type = float, help = 'initial learning rate')
parser.add_argument('--lr_decay_factor', default=0.5, type = float, help = 'step LR decay factor')
parser.add_argument('--lr_decay_step_size', default=25, type = int, help = 'steps to decay LR')
parser.add_argument('--reinitialize_lr', default='True', type = str, help = 'whether or not to reinitialize LR in every episode')

##
parser.add_argument('--sep_unlabel', default='True', choices = ['True', 'False'], type = str, help = 'separate labeled vs unlabeled data loss')
parser.add_argument('--pseudo_ensemble', default='True', type = str, choices = ['True', 'False'], help = 'to aggregate pseudo labels across episodes to allow a more stable pseudo-label')
parser.add_argument('--decay_ensemble', default=0.9, type = float, help = 'the decaying coefficient for pseudo ensemble')

## uncertainty
parser.add_argument('--unlabeled_sample_metric', default='uncertainty', choices = ['random', 'consistency', 'uncertainty'], type = str, help = 'how to measure the quality of pseudo-label for each unlabeled data point')
parser.add_argument('--uncertainty', default='evidential', choices = ['False', 'evidential', 'gaussian'], type = str, help = 'uncertainty measure')
parser.add_argument('--uncertainty_type', default='epistemic', choices = ['aleatoric', 'epistemic'], type = str, help = 'evidential uncertainty type')
parser.add_argument('--evi_lambda', default=0.5, type=float, help = 'lambda for the evidential loss')
parser.add_argument('--consistency_metric', default=0.2, type=float, help = 'threshold for consistency')
parser.add_argument('--uncertainty_threshold', default=0.1, type = float, help = 'threshold for uncertainty, not used if using weighted loss')

## adaptice weighting
parser.add_argument('--weighted_loss', default='True', choices = ['True', 'False'], type = str, help = 'use adaptive weighting')
parser.add_argument('--inverse_weighted_loss', default='True', choices = ['True', 'False'], type = str, help = 'use inverse uncertainty or just the uncertainty')

## others
parser.add_argument('--noise_type', default='False', type = str, choices =['False', 'True', 'laplace'], help = 'add positional noise')
parser.add_argument('--noise_var', default=0.001, type = float, help = 'positional noise variance')
parser.add_argument('--student_training', default='False', choices = ['True', 'False'], type = str, help = 'whether or not to do student training')
parser.add_argument('--ood_eval', default='False', type = str, choices = ['True', 'False'], help = 'whether or not to evaluate on OOD dataset, only work for standard setting')

args = parser.parse_args()

if args.setting == 'low_data':
    args.ood_eval = 'False'
else:
    args.ood_eval = 'True'
    
if args.label not in ['gap', 'homo', 'lumo']:
    args.ood_eval = 'False'

if args.pseudo_label == 'False':
    args.exp_name = args.model + '_' +args.label + '_' + args.setting
    if args.uncertainty == 'False':
        args.exp_name += '_vanilla'
else:
    args.exp_name = args.model + '+pl3_' +args.label + '_' + args.setting
    
if args.setting == 'low_data':
    args.exp_name += str(args.training_fraction)

if args.uncertainty in ['evidential', 'gaussian']:    
    args.exp_name = args.exp_name + '_' + args.uncertainty_type     
else:
    args.exp_name += '_nosigma'

if args.student_training == 'True':
    args.exp_name += '_stu'
    
if args.weighted_loss == 'False':
    args.exp_name += '_wlf'
    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

if args.wandb == 'True':
    import wandb
    wandb.init(project=args.proj_name, entity=args.entity_name, name=args.exp_name)
    wandb.config.update(args)

if args.setting == 'low_data':

    dataset = QM9_3D(root='dataset/')
    target = args.label
    pos = dataset.data.pos

    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)

    train_fraction = int(110000 * args.training_fraction)
    mask_unlabelled = dataset.data[target]
    mask_unlabelled[split_idx['train'][train_fraction:110000]] = 0
    dataset.data.y = mask_unlabelled

    mask_unlabelled_status = np.array([0] * len(dataset.data[target]))
    mask_unlabelled_status[split_idx['train'][train_fraction:110000]] = 1
    dataset.data.unlabelled_status = mask_unlabelled_status

    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

    train_dataset = dataset[split_idx['train'][:train_fraction]]
    unlabelled_dataset = dataset[split_idx['train'][train_fraction:110000]]
    split_idx['unlabeled'] = split_idx['train'][train_fraction:110000]
    print('train, validaion, test, unlabeled:', len(train_dataset), len(valid_dataset), len(test_dataset), len(unlabelled_dataset))

else:
    
    dataset = PC9_QM9_3D(root = 'dataset/')
    target = args.label
    
    split_idx = dataset.get_idx_split(train_size = 110000, valid_size = 10000, seed = 42)
    train_fraction = 110000
    
    if args.ood_eval == 'True':
        mask_ood = np.array([0] * len(dataset.data[target]), dtype = float)
        mask_ood[split_idx['unlabeled']] = dataset.data[target][split_idx['unlabeled']]
        dataset.data.y_ood = mask_ood
    
    mask_unlabelled_status = np.array([0] * len(dataset.data[target]))
    mask_unlabelled_status[split_idx['unlabeled']] = 1
    dataset.data.unlabelled_status = mask_unlabelled_status
    
    dataset.data.y = dataset.data[target]
    train_dataset, valid_dataset, test_dataset, unlabelled_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']], dataset[split_idx['unlabeled']]
    print('train, validaion, test, unlabeled:', len(train_dataset), len(valid_dataset), len(test_dataset), len(unlabelled_dataset))
    
if args.model == 'schnet':
    model = SchNet(cutoff = 5.0, 
               num_layers=6, 
               hidden_channels=128, 
               num_filters=128, 
               num_gaussians=25,
              noise_var = args.noise_var, uncertainty = args.uncertainty)
    #model = SchNet(noise_var = args.noise_var, uncertainty = args.uncertainty)
elif args.model == 'dimenet':
    model = DimeNetPP(cutoff=5.0, num_layers=4, 
        hidden_channels=128, out_channels=1, int_emb_size=64, basis_emb_size=8, out_emb_channels=256, 
        num_spherical=7, num_radial=6, envelope_exponent=5, 
        num_before_skip=1, num_after_skip=2, num_output_layers=3,
        noise_var = args.noise_var,  uncertainty = args.uncertainty)

loss_func = torch.nn.L1Loss()
evaluation = threedEvaluator()

run3d = run()

if args.pseudo_label == 'False':

    run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, epochs=args.initial_train_epoch, batch_size=args.batch_size, lr=args.lr, lr_decay_factor=args.lr_decay_factor, lr_decay_step_size=args.lr_decay_step_size, wandb_mode = args.wandb, exp_name = args.exp_name, proj_name = args.proj_name, entity_name = args.entity_name, uncertainty = args.uncertainty, evi_lambda = args.evi_lambda, unlabeled_dataset = unlabelled_dataset, ood_eval = args.ood_eval, weighted_loss = 'False', reinit_lr = args.reinitialize_lr, uncertainty_type = args.uncertainty_type)

elif args.pseudo_label == 'True':        
        
    train_total_dataset = train_dataset
    run3d.run(device, train_total_dataset, 
          valid_dataset, test_dataset, 
          model, loss_func, evaluation, 
          epochs=args.initial_train_epoch, batch_size=args.batch_size, lr=args.lr, 
          lr_decay_factor=args.lr_decay_factor, lr_decay_step_size=args.lr_decay_step_size, wandb_mode = args.wandb, 
          exp_name = args.exp_name, proj_name = args.proj_name, entity_name = args.entity_name, add_noise = 'False', 
          sep_unlabel = 'False', uncertainty = args.uncertainty, evi_lambda = args.evi_lambda,
          unlabeled_dataset = unlabelled_dataset, ood_eval = args.ood_eval, weighted_loss = 'False',  reinit_lr = args.reinitialize_lr,
          uncertainty_type = args.uncertainty_type)

    if args.unlabeled_sample_metric == 'consistency':
        pred_series = []
    
    if args.pseudo_ensemble == 'True':
        pred_ensemble_ckpt = 0
        
    for i in range(args.iteration):
        if args.setting == 'low_data':
            unlabeled_samples_idx = list(range(train_fraction, 110000))            
            unlabelled_dataset = dataset[split_idx['train'][np.array(unlabeled_samples_idx)]]
        elif args.setting == 'standard':
            unlabeled_samples_idx = split_idx['unlabeled'].numpy().tolist()
            unlabelled_dataset = dataset[unlabeled_samples_idx]
            
        if args.model in ['dimenet', 'spherenet']:
            vt_bs = 128
        else:
            vt_bs = 512
            
        unlabelled_dataset_loader = DataLoader(unlabelled_dataset, vt_bs, shuffle=False)
        pred = run3d.evaluate(model, 
                       data_loader = unlabelled_dataset_loader, 
                       energy_and_force = False, p = 100, 
                       evaluation = evaluation, device = device, 
                       uncertainty = args.uncertainty,
                       ood_status = 'False',
                       uncertainty_type = args.uncertainty_type)
        
        if args.uncertainty in ['evidential', 'gaussian']:
            pred, std = pred
        
        if args.pseudo_ensemble == 'True':
            if i == 0:
                pred_ensemble_ckpt = pred
            else:
                pred_temp = pred
                pred = args.decay_ensemble * pred + (1 - args.decay_ensemble) * pred_ensemble_ckpt
                pred_ensemble_ckpt = pred_temp
            
        
        # update the label
        temp = dataset.data.y
        
        if args.setting == 'low_data':
            temp[split_idx['train'][np.array(unlabeled_samples_idx)]] = pred.to('cpu').reshape(-1)
        else:
            temp[np.array(unlabeled_samples_idx)] = pred.to('cpu').reshape(-1)
            
        dataset.data.y = temp
        
        if args.weighted_loss == 'True':
            ## not using any filtering, but generate normalized uncertainty as the weight for loss
            if args.unlabeled_sample_metric == 'uncertainty':
                loss_weight = std.cpu().clone().detach().reshape(-1)
            
            elif args.unlabeled_sample_metric == 'consistency':
                pred_series.append(pred)
                if i > 1:
                    consistency = torch.std(torch.stack(pred_series), axis = 0)
                else:
                    # to avoid nan in the first run
                    consistency = torch.zeros_like(pred_series[0])
                
                loss_weight = torch.tensor(consistency).reshape(-1)
                
            elif args.unlabeled_sample_metric == 'random':
                loss_weight = torch.tensor([1] * len(split_idx['unlabeled']))
            
            if args.inverse_weighted_loss == 'True':
                loss_weight = 1 / (loss_weight + 1e-5)
            loss_weight = loss_weight/sum(loss_weight)
            weights = np.array([0] * len(dataset.data[target]), dtype = float)
            weights[split_idx['unlabeled']] = loss_weight
            dataset.data.loss_weight = weights
            
        else:
            if args.unlabeled_sample_metric == 'consistency':
                pred_series.append(pred)
                if i > 1:
                    consistency = torch.std(torch.stack(pred_series), axis = 0)
                else:
                    # to avoid nan in the first run
                    consistency = torch.zeros_like(pred_series[0])

                consistent_idx = torch.where(consistency < args.consistency_metric)[0]
                print(str(consistent_idx.shape[0] / consistency.shape[0]) + ' of unlabelled data are consistent...')
                print('Consistency mean: ' + str(np.mean(consistency.cpu().numpy())))
                print('Consistency std: ' + str(np.std(consistency.cpu().numpy())))

    #            if i % 5 == 0:
                    #import seaborn as sns
                    #import matplotlib.pyplot as plt
                    #sns.distplot(consistency.cpu())
                    #plt.savefig('distplot_' + str(i) + '.png')
                # filter the ones that are not consistent
                unlabeled_samples_idx = np.array(unlabeled_samples_idx)[consistent_idx.cpu()].tolist()
            if args.unlabeled_sample_metric == 'uncertainty':
                std_idx = torch.where(std < args.uncertainty_threshold)[0]
                print(str(std_idx.shape[0] / std.shape[0]) + ' of unlabelled data are confident...')
                print('Confidence mean: ' + str(np.mean(std.cpu().numpy())))
                print('Confidence std: ' + str(np.std(std.cpu().numpy())))
                unlabeled_samples_idx = np.array(unlabeled_samples_idx)[std_idx.cpu()].tolist()

                if i % 1 == 0:
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    plt.figure()
                    sns.displot(std.cpu().numpy(), kind = "kde")
                    plt.savefig('plots/distplot_' + args.uncertainty + '_' + str(args.iteration) + '.png')

            elif args.unlabeled_sample_metric == 'random':
                unlabeled_samples_idx = unlabeled_samples_idx

        
        train_dataset = dataset[split_idx['train'][:train_fraction]]
        
        if args.setting == 'low_data':
            unlabelled_dataset = dataset[split_idx['train'][np.array(unlabeled_samples_idx)]]
        else:
            unlabelled_dataset = dataset[np.array(unlabeled_samples_idx)]
        
        if len(unlabeled_samples_idx) == 0:
            train_total_dataset = train_dataset
        else:    
            train_total_dataset = train_dataset + unlabelled_dataset
        
        if args.student_training == 'True':
            if args.model == 'schnet':
                model = SchNet(cutoff = 5.0, 
                           num_layers=6, 
                           hidden_channels=128, 
                           num_filters=128, 
                           num_gaussians=25,
                           noise_var = args.noise_var, uncertainty = args.uncertainty)
                #model = SchNet(noise_var = args.noise_var, uncertainty = args.uncertainty)
            elif args.model == 'dimenet':
                model = DimeNetPP(cutoff=5.0, num_layers=4, 
                    hidden_channels=128, out_channels=1, int_emb_size=64, basis_emb_size=8, out_emb_channels=256, 
                    num_spherical=7, num_radial=6, envelope_exponent=5, 
                    num_before_skip=1, num_after_skip=2, num_output_layers=3,
                    noise_var = args.noise_var,  uncertainty = args.uncertainty)
            
        run3d.run(device, train_total_dataset, 
              valid_dataset, test_dataset, 
              model, loss_func, evaluation, 
              epochs=args.epoch, wandb_mode = args.wandb, 
              exp_name = args.exp_name, proj_name = args.proj_name, entity_name = args.entity_name,
              add_noise = args.noise_type, sep_unlabel = args.sep_unlabel, 
              uncertainty = args.uncertainty, evi_lambda = args.evi_lambda, 
              unlabeled_dataset = unlabelled_dataset, ood_eval = args.ood_eval, weighted_loss = args.weighted_loss,
              batch_size=args.batch_size, lr=args.lr, lr_decay_factor=args.lr_decay_factor, 
              lr_decay_step_size=args.lr_decay_step_size,  reinit_lr = args.reinitialize_lr,
              uncertainty_type = args.uncertainty_type)        
    