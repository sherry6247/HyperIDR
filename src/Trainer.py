'''
Author: Sicen Liu
Date: 2025-05-16 15:42:32
LastEditTime: 2026-01-18 17:29:42
FilePath: /liusicen/methods/Prompt_pool_IDP/src/Trainer.py
Description: 

Copyright (c) 2025 by ${liusicen_cs@outlook.com}, All Rights Reserved. 
'''

import os
import torch
import shutil
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset
# import wandb
from collections import defaultdict

from src.evaluate import eval, predict

import json
import pickle
from collections import OrderedDict

from torch.utils.tensorboard import SummaryWriter


"""adjust_learning_rate"""
def lr_poly(base_lr, iter, max_iter, power):
    # ratio_length = 1 - (float(current_length) / 30) 
    # iter = iter + ratio_length
    # iter = iter + current_length
    if iter > max_iter:
        iter = iter % max_iter
    return base_lr * ((1 - float(iter) / max_iter) ** (power))#+ (float(current_length) / 30) ** (power))
    # return base_lr * (((1 - float(iter) / max_iter) ** (power))+ 0.1*((1 - (float(current_length) / 30) ** (power))))

def adjust_learning_rate(optimizer, i_iter, args):
    lr = lr_poly(args.lr, i_iter, 100000, 0.9)
    optimizer.param_groups[0]['lr'] = np.min(np.around(lr,8))
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

class Trainer:

    def __init__(
            self, args, 
            model, 
            train_dataloader:DataLoader, 
            val_dataloader:DataLoader, 
            test_dataloader:DataLoader
    ):
        self.args = args
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = args.optimizer
    
    def train(self):
        print("Start Training...")
        patients, t_patients = 0, 0
        model = self.model
        global_step = 0
        nb_tr_setp = 0
        tr_loss = 0
        val_loss = 0
        dev_auc_best, dev_mcc_best, dev_bacc_best, dev_best, val_test_auc_best = 0, 0, 0, 0, 0
        test_auc_best = 0
        best_results = defaultdict(list)
        cu_iter = 0
        # for epoch in trange(int(self.args.epochs), desc="Epoch:", ncols=100):
        for epoch in range(int(self.args.epochs)):
            print("\n--------K: {} epoch: {} seed: {} model_name:{} train_data:{}/{} test_data:{} --------".format(self.args.k, epoch, self.args.seed, self.args.model_name, self.args.train_dataset, len(self.train_dataloader.dataset), self.args.test_dataset))
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            prog_iter = tqdm(self.train_dataloader, desc='Training', ncols=100, colour='blue')
            model = model.to(self.args.device)
            model.train()
            for _, batch in enumerate(prog_iter):
                self.optimizer.zero_grad()

                id, seq, ori_labels, input_features, input_mask, labels = batch
                # Move tensors to GPU
                input_features = input_features.to(self.args.device)
                input_mask = input_mask.to(self.args.device)
                labels = labels.to(self.args.device)

                if self.args.model_name == 'ConditionalIDRModel':
                    logits_dict = model(input_features)
                    logits = logits_dict['logits']
                
                # padding 的部分不计算loss
                # logits = logits * input_mask
                loss = self.args.criterion(logits, labels)
                loss = (loss * input_mask).mean()

                
                #清零梯度
                self.optimizer.zero_grad()
                
                loss.backward()
                self.optimizer.step()

                # train_results = cal_batch(self.args, labels, logits, residue_masks, sequences_masks)

                tr_loss += loss.item()
                nb_tr_examples += 1
                nb_tr_steps += 1

                # Display loss
                # if self.is_local_0 == 0:
                prog_iter.set_postfix({"train_loss" :'%.4f' % (tr_loss / nb_tr_steps), "lr" : "%.6f" % (self.optimizer.param_groups[0]['lr'])})
                # prog_iter.set_postfix(train_loss='%.4f' % (loss.item()))

                torch.cuda.empty_cache()
            
        
            # Validation 
            print("=================VALIDATION =================")
            val_results = eval(self.args, model, self.val_dataloader)
            
            feature_params = {"esm2":self.args.esm2, "drBERT":self.args.drBERT, "pssm":self.args.pssm}
            feature_name = [k for k, v in feature_params.items() if v == 1]

            if self.args.test_dataset in ["MXD494", "DISORDER723_test"]:
                val_met = val_results['Bacc']
            if self.args.test_dataset in ["SL329", "CASP", "Disprot504","CAID3_disorder_nox","CAID3_disorder_pdb"]:
                val_met = val_results['AUC']
            # val_met = val_results['AUC']

            if val_met > dev_best:
                patients = 0
                dev_best = val_met
                save_path = os.path.join(self.args.save_result, "{}".format(self.args.model_name), "{}".format(self.args.test_dataset), "{}".format(self.args.train_dataset), "SEED_{}".format(self.args.seed), "{}".format("-".join(feature_name)), "lr_{}_tk_{}_ps_{}_pl_{}_pk_{}_lamda_{}".format(self.args.lr, self.args.top_k, self.args.pool_size, self.args.prompt_length,self.args.use_prompt_key, self.args.lambda_orth), "K_{}".format(self.args.k))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save({
                    "epoch" : epoch,
                    "k" : self.args.k,
                    "model_state_dict" : model.state_dict()
                }, os.path.join(save_path,"model.pth"))
                try:
                    val_selected_keys = ['AUC', 'Bacc', 'MCC', 'Sn', 'Sp', 'F1', 'F_max', 'APS-PRAUC']
                    val_selected_results = {k: val_results[k] for k in val_selected_keys if k in val_results}
                    val_selected_results['Epoch'] = epoch
                    val_selected_results['K'] = self.args.k
                    val_selected_results['Seed'] = self.args.seed
                    val_selected_results['Model'] = self.args.model_name
                    val_selected_results['Test_Dataset'] = self.args.test_dataset
                    val_selected_results['hidden_size'] = self.args.hidden_size
                    val_selected_results['num_layers'] = self.args.num_layers
                    val_selected_results['learning_rate'] = self.args.lr
                    val_selected_results["train_dataset_len"] = (len(self.train_dataloader.dataset)+len(self.val_dataloader.dataset))
                    val_selected_results["top_k"] = self.args.top_k
                    val_selected_results["pool_size"] = self.args.pool_size
                    val_selected_results["backbone"] = self.args.backbone
                    val_selected_results["warmup_sdr"] = self.args.warmup_sdr
                    val_selected_results["warmup_ldr"] = self.args.warmup_ldr
                    with open(os.path.join(save_path, 'best_val_results.json'), 'w') as f:
                        json.dump(val_selected_results, f, indent=4)
                    
                except Exception as e:
                    print(f"Error saving model: {e}")

               
            else:
                patients += 1
                if patients == 10:
                    print("Early stopping at epoch {}.".format(epoch))
                    break
                
            
            
            