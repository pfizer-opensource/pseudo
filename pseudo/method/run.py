## based on https://github.com/divelab/DIG

import time
import os
import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader
import numpy as np
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F


def gaussian_loss(pred, targets, weight_loss = None):
    pred_targets, pred_logv = pred
    log_likelihood = 0.5 * ( -torch.exp(-pred_logv) * (pred_targets - targets)**2 - pred_logv - np.log(2*np.pi))    
    loss = -log_likelihood
    if weight_loss is not None:
        return torch.sum(loss * weight_loss)
    else:
        return torch.mean(loss)

def nig_nll(y, gamma, v, alpha, beta):
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
            - alpha * torch.log(two_blambda) \
            + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + two_blambda) \
            + torch.lgamma(alpha) \
            - torch.lgamma(alpha + 0.5)
        
    return nll


def nig_reg(y, gamma, v, alpha, beta):
    error = F.l1_loss(y, gamma, reduction="none")
    evi = 2 * v + alpha
    return error * evi


def evidential_regresssion_loss(y, pred, coeff=1.0, weight_loss = None):
    gamma, v, alpha, beta = pred
    loss_nll = nig_nll(y, gamma, v, alpha, beta)
    loss_reg = nig_reg(y, gamma, v, alpha, beta)
    
    if weight_loss is None:
        loss_ = loss_nll.mean() + coeff * (loss_reg.mean() - 1e-4)
        return loss_
    else:
        loss_ = torch.sum(weight_loss * loss_nll) + coeff * (torch.sum(loss_reg * weight_loss) - 1e-4)
        
        return loss_
    
def get_pred_evidential_aleatoric(out):
    gamma, v, alpha, beta = out
    var = beta / (alpha - 1)
    return gamma, var

def get_pred_evidential_epistemic(out):
    gamma, v, alpha, beta = out
    var = beta / (v * (alpha - 1))
    return gamma, var


class run():

    def __init__(self):
        
        self.best_val_mae = 10000
        self.best_test_mae = 10000
        
    def run(self, device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, epochs=300, batch_size=128, vt_batch_size=128, lr=0.001, lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=0, 
        energy_and_force=False, p=100, save_dir='', log_dir='', wandb_mode = 'False', exp_name = 'dig', proj_name = 'dig_data', entity_name = 'kexin', add_noise = 'False', sep_unlabel = 'False', uncertainty = 'False', evi_lambda = 1e-2, unlabeled_dataset = None, ood_eval = 'False', weighted_loss = 'False', reinit_lr = 'True', sweep_wandb = None, uncertainty_type = 'aleatoric'):
         
        if wandb_mode == 'True':
            import wandb
            wandb.init(project=proj_name, entity=entity_name, name=exp_name)
        model = model.to(device)
        
        if reinit_lr == 'True':
            self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            self.scheduler = StepLR(self.optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)
        elif reinit_lr == 'False':
            if 'optimizer' in self.__dict__:
                pass
            else:
                self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                self.scheduler = StepLR(self.optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)
        
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last = True)
        valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
        
        if unlabeled_dataset is not None:
            unlabeled_loader = DataLoader(unlabeled_dataset, 512, shuffle = False)
        
        best_valid = float('inf')
        best_test = float('inf')
            
        if save_dir != '':
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if log_dir != '':
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir=log_dir)
        
        
        if add_noise == 'True':
            add_pos_noise = True
        else:
            add_pos_noise = False
            
        if sep_unlabel == 'True':
            sep_unlabel = True
        else:
            sep_unlabel = False
            
        for epoch in range(1, epochs + 1):
            print("\n=====Epoch {}".format(epoch), flush=True)
            
            print('\nTraining...', flush=True)
            train_mae = self.train(model, self.optimizer, train_loader, energy_and_force, p, loss_func, device, add_pos_noise, sep_unlabel, uncertainty, evi_lambda, weighted_loss)

            print('\n\nEvaluating...', flush=True)
            valid_mae = self.val(model, valid_loader, energy_and_force, p, evaluation, device, uncertainty, uncertainty_type)

            print('\n\nTesting...', flush=True)
            test_mae = self.val(model, test_loader, energy_and_force, p, evaluation, device, uncertainty, uncertainty_type)

            print()
            print({'Train': train_mae, 'Validation': valid_mae, 'Test': test_mae})

            if log_dir != '':
                writer.add_scalar('train_mae', train_mae, epoch)
                writer.add_scalar('valid_mae', valid_mae, epoch)
                writer.add_scalar('test_mae', test_mae, epoch)
            
            if wandb_mode == 'True':
                wandb.log({"validation mae": valid_mae, 
                          "training mae": train_mae})
            
            if sweep_wandb is not None:
                sweep_wandb.log({"validation_mae": valid_mae, 
                          "training_mae": train_mae})
            
            
            if valid_mae < best_valid:
                best_valid = valid_mae
                best_test = test_mae
                if save_dir != '':
                    print('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(), 'best_valid_mae': best_validid_mae, 'num_params': num_params}
                    torch.save(checkpoint, os.path.join(save_dir, 'valid_checkpoint.pt'))

            self.scheduler.step()
            
            if (ood_eval == 'True') and (epoch % 20 == 0):        
                with torch.no_grad():
                    ood_mae = self.evaluate(model, 
                           data_loader = unlabeled_loader, 
                           energy_and_force = False, p = 100, 
                           evaluation = evaluation, device = device, 
                           uncertainty = uncertainty,
                           ood_status = 'True')
                    print('MAE on PC9: ' + str(ood_mae))

                    if wandb_mode == 'True':
                        wandb.log({"OOD Val MAE": ood_mae})

                    if sweep_wandb is not None:
                        sweep_wandb.log({"ood_val_mae": ood_mae})
            
        print(f'Best validation MAE in current episode: {best_valid}')
        print(f'Test MAE when got best validation result in current episode: {best_test}')
        
        if best_valid < self.best_val_mae:
            self.best_test_mae = best_test
            self.best_val_mae = best_valid
            
        print(f'Best validation MAE in all episode: {self.best_val_mae}')
        print(f'Best testing MAE when got best validation result in all episodes: {self.best_test_mae}')
        
        if wandb_mode == 'True':
            wandb.log({"best validation mae": best_valid})
            wandb.log({"test mae": best_test})
        
        if sweep_wandb is not None:
            sweep_wandb.log({"best_validation_mae": best_valid})
            sweep_wandb.log({"test_mae": best_test})
        
        if log_dir != '':
            writer.close()
        
        if ood_eval == 'True':        
            with torch.no_grad():
                ood_mae = self.evaluate(model, 
                       data_loader = unlabeled_loader, 
                       energy_and_force = False, p = 100, 
                       evaluation = evaluation, device = device, 
                       uncertainty = uncertainty,
                       ood_status = 'True')
                print('MAE on PC9: ' + str(ood_mae))

                if wandb_mode == 'True':
                    wandb.log({"OOD MAE": ood_mae})

                if sweep_wandb is not None:
                    sweep_wandb.log({"ood_mae": ood_mae})
    def train(self, model, optimizer, train_loader, energy_and_force, p, loss_func, device, add_pos_noise, sep_unlabel, uncertainty, evi_lambda, weighted_loss):
        model.train()
        loss_accum = 0
        for step, batch_data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            out = model(batch_data, add_pos_noise)
            
            if sep_unlabel:
                unlabelled_status = torch.Tensor(batch_data.unlabelled_status)
                unlabelled_idx = torch.where(unlabelled_status == 1)[0]
                labelled_idx = torch.where(unlabelled_status == 0)[0]
                if uncertainty == 'evidential':
                    if unlabelled_idx.shape[0] == 0:
                        loss = evidential_regresssion_loss(batch_data.y.unsqueeze(1), out, evi_lambda)
                    elif labelled_idx.shape[0] == 0:
                        out_unlabeled = tuple([i[unlabelled_idx] for i in out])
                        if weighted_loss == 'True':
                            loss = evidential_regresssion_loss(batch_data.y[unlabelled_idx].unsqueeze(1), out_unlabeled, evi_lambda, batch_data.loss_weight[unlabelled_idx])
                        else:
                            loss = evidential_regresssion_loss(batch_data.y[unlabelled_idx].unsqueeze(1), out_unlabeled, evi_lambda, None)
                    else:
                        out_labeled = tuple([i[labelled_idx] for i in out])
                        out_unlabeled = tuple([i[unlabelled_idx] for i in out])
                        if weighted_loss == 'True':
                            loss = evidential_regresssion_loss(batch_data.y[labelled_idx].unsqueeze(1), out_labeled, evi_lambda) + evidential_regresssion_loss(batch_data.y[unlabelled_idx].unsqueeze(1), out_unlabeled, evi_lambda, batch_data.loss_weight[unlabelled_idx])
                           
                        else:
                            loss = evidential_regresssion_loss(batch_data.y[labelled_idx].unsqueeze(1), out_labeled, evi_lambda) + evidential_regresssion_loss(batch_data.y[unlabelled_idx].unsqueeze(1), out_unlabeled, evi_lambda, None)
                elif uncertainty == 'gaussian':
                    if unlabelled_idx.shape[0] == 0:
                        loss = gaussian_loss(out, batch_data.y.unsqueeze(1))
                    else:
                        out_labeled = tuple([i[labelled_idx] for i in out])
                        out_unlabeled = tuple([i[unlabelled_idx] for i in out])
                        if weighted_loss == 'True':
                            loss = gaussian_loss(out_labeled, batch_data.y[labelled_idx].unsqueeze(1)) + gaussian_loss(out_unlabeled, batch_data.y[unlabelled_idx].unsqueeze(1), batch_data.loss_weight[unlabelled_idx])
                        else:
                            loss = gaussian_loss(out_labeled, batch_data.y[labelled_idx].unsqueeze(1)) + gaussian_loss(out_unlabeled, batch_data.y[unlabelled_idx].unsqueeze(1), None)
                else:
                    if weighted_loss == 'True':
                        loss = loss_func(out[labelled_idx], batch_data.y[labelled_idx].unsqueeze(1)) + \
                        loss_func(out[unlabelled_idx] *  batch_data.uncertainty[unlabelled_idx], batch_data.y[unlabelled_idx].unsqueeze(1) *  batch_data.loss_weight[unlabelled_idx])
                    else:
                        loss = loss_func(out[labelled_idx], batch_data.y[labelled_idx].unsqueeze(1)) + \
                        loss_func(out[unlabelled_idx], batch_data.y[unlabelled_idx].unsqueeze(1))
                    
            else:
                if uncertainty == 'evidential':
                    loss = evidential_regresssion_loss(batch_data.y.unsqueeze(1), out, evi_lambda)
                elif uncertainty == 'gaussian':
                    loss = gaussian_loss(out, batch_data.y.unsqueeze(1))
                else:    
                    loss = loss_func(out, batch_data.y.unsqueeze(1))

            loss.backward()
            optimizer.step()
            loss_accum += loss.detach().cpu().item()
        return loss_accum / (step + 1)
    
    
    def evaluate(self, model, data_loader, energy_and_force, p, evaluation, device, uncertainty, ood_status, return_raw = False, uncertainty_type = 'aleatoric'):
        model.eval()
        
        preds = torch.Tensor([])
        uncertaintys = torch.Tensor([])
        targets = torch.Tensor([])
        out_all = []
        
        for step, batch_data in enumerate(tqdm(data_loader)):
            batch_data = batch_data.to(device)
            out = model(batch_data, add_pos_noise = False)
            if uncertainty == 'evidential':
                if return_raw:
                    out_all.append(tuple([i.detach().cpu() for i in out]))
                if uncertainty_type == 'aleatoric':
                    out, var = get_pred_evidential_aleatoric(out)
                elif uncertainty_type == 'epistemic':
                    out, var = get_pred_evidential_epistemic(out)
                uncertaintys = torch.cat([uncertaintys, var.detach().cpu()], dim = 0)
            elif uncertainty == 'gaussian':
                out, var = out
                std = torch.sqrt(torch.exp(var))
                uncertaintys = torch.cat([uncertaintys, std.detach().cpu()], dim = 0)
            preds = torch.cat([preds, out.detach().cpu()], dim=0) 
            if ood_status == 'False':
                t = batch_data.y.unsqueeze(1)
            else:
                t = batch_data.y_ood.unsqueeze(1)
            targets = torch.cat([targets, t.cpu()], dim=0)
            
        input_dict = {"y_true": targets, "y_pred": preds}
        
        if return_raw:
            return out_all, targets.cpu()
        
        if ood_status == 'True':
            return evaluation.eval(input_dict)['mae']
        
        if uncertainty in ['evidential', 'gaussian']:
            return preds, uncertaintys
        else:
            return preds

    
    def val(self, model, data_loader, energy_and_force, p, evaluation, device, uncertainty, uncertainty_type):
        model.eval()

        preds = torch.Tensor([]).to(device)
        targets = torch.Tensor([]).to(device)

        if energy_and_force:
            preds_force = torch.Tensor([]).to(device)
            targets_force = torch.Tensor([]).to(device)
        
        for step, batch_data in enumerate(tqdm(data_loader)):
            batch_data = batch_data.to(device)
            out = model(batch_data, add_pos_noise = False)
            if uncertainty == 'evidential':
                if uncertainty_type == 'aleatoric':
                    out, _ = get_pred_evidential_aleatoric(out)
                elif uncertainty_type == 'epistemic':
                    out, _ = get_pred_evidential_epistemic(out)
            elif uncertainty == 'gaussian':
                out, _ = out
            preds = torch.cat([preds, out.detach()], dim=0)
            targets = torch.cat([targets, batch_data.y.unsqueeze(1)], dim=0)

        input_dict = {"y_true": targets, "y_pred": preds}
        
        return evaluation.eval(input_dict)['mae']
