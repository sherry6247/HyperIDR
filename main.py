'''
Date: 2025-07-18 17:05:41
LastEditors: Please set LastEditors
LastEditTime: 2025-12-19 15:55:01
FilePath: /liusicen/methods/Prompt_pool_IDP/main.py
Description:  This is the main entry point for training the Prompt Pool IDP model.
'''

import os
import sys
import sys, getopt
import torch
import numpy as np
import re
import os
import random
import torch.nn as nn
from functools import partial

from src.model import ConditionalIDRModel
from torch.optim import AdamW
from utils.args_config import parse_args
from utils.data_processing import load_dataset, proteinDataset, kfold_split, collate_fn_batch, CAID3_collate_fn_batch, collect_span_embeddings_from_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from src.Trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
# import wandb

# os.environ["WANDB_MODE"] = "offline"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        use_cuda = True
        print("Running on CUDA")
    else:
        use_cuda = False
        print("Running on CPU")

    setup_seed(args.seed)

    device = torch.device('cuda')
    args.device = device
    
    criterion = {
        "ConditionalIDRModel" : nn.BCEWithLogitsLoss(reduction='none'),
    }
    args.criterion = criterion[args.model_name]

    # if not args.debug:
    #     run_name = f"seed_{args.seed}_{args.model_name}_lr_{args.lr}_thres_{args.threshold}"
    #     wandb.init(project="{}-{}".format(args.model_name, args.test_dataset), name=run_name, config=args,settings=wandb.Settings(init_timeout=120))

    # writer = SummaryWriter(log_dir="runs/my_exp")
    
    # feature 选择concatenate
    feature_dim_dict = {
        "pssm":20,
        "esm2":1280,
        "drBERT":768,
    }
    feature_dims = []
    if args.pssm:
        feature_dims.append(feature_dim_dict['pssm'])
    if args.esm2:
        feature_dims.append(feature_dim_dict['esm2'])
    if args.drBERT:
        feature_dims.append(feature_dim_dict['drBERT'])
    
    feature_dim = np.sum(feature_dims)

    ####################################################################
    ## Load the dataset
    #####################################################################
    train_dataset = load_dataset(args, args.train_dataset_path, dataset_type=args.train_dataset)
    
    collate_fn_with_params = partial(collate_fn_batch, args=args)
    
    # K-Fold: return dataset lists
    train_Datas, val_Datas = kfold_split(train_dataset, seed=args.seed)
    
    for k, (train_Dataset, val_Dataset) in  enumerate(zip(train_Datas, val_Datas)):
        #############################Train#######################################
        train_Dataset = proteinDataset(train_Dataset)
        train_sampler = RandomSampler(train_Dataset)
        train_dataloader = DataLoader(train_Dataset,
                                      sampler=train_sampler,
                                      batch_size=args.batch_size,
                                      collate_fn=collate_fn_with_params)
        val_Dataset = proteinDataset(val_Dataset)
        val_dataloader = DataLoader(val_Dataset,
                                    sampler=SequentialSampler(val_Dataset),
                                    batch_size=1,
                                    collate_fn=collate_fn_with_params)
        
        #############################Test#######################################
        testDataset = proteinDataset(test_dataset)
        if args.test_dataset in ['CASP','SL329','CAID3_disorder_pdb', 'CAID3_disorder_nox', 'CAID2_disorder_nox', 'CAID2_disorder_pdb']:
            test_collate_fn_with_params = partial(CAID3_collate_fn_batch, args=args)
        else:
            test_collate_fn_with_params = partial(collate_fn_batch, args=args)
        test_dataloader = DataLoader(testDataset,
                                    sampler=SequentialSampler(testDataset),
                                    batch_size=1,
                                    collate_fn=test_collate_fn_with_params)
        

        if args.model_name == 'Multi_scale_expert_dynamic_Model':    
            model = Multi_scale_expert_dynamic_Model(d_model=feature_dim, num_prompts=args.pool_size, top_k=args.top_k, branch_hidden=args.hidden_size, backbone=args.backbone, backbone_depth=args.num_layers)
            if args.prompt_warmup:
                print("Using prompt warmup")
                span_emb, span_lab = collect_span_embeddings_from_dataset(train_dataset)
                model.prompt_pool.warm_start(span_emb, span_lab, n_short=args.warmup_sdr, n_long=args.warmup_ldr)
        elif args.model_name == 'ConditionalIDRModel':
            model = ConditionalIDRModel(d_model=feature_dim, num_prompts=args.pool_size, top_k=args.top_k, branch_hidden=args.hidden_size, backbone=args.backbone, backbone_depth=args.num_layers)
            if args.prompt_warmup:
                print("Using prompt warmup")
                span_emb, span_lab = collect_span_embeddings_from_dataset(train_dataset)
                model.prompt_pool.warm_start(span_emb, span_lab, n_short=args.warmup_sdr, n_long=args.warmup_ldr)


        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
            model.to('cuda')
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 

        args.optimizer = optimizer
        args.k = k
        
        trainer = Trainer(args, model, train_dataloader, val_dataloader)
        trainer.train()
    


if __name__ == "__main__":
    main()