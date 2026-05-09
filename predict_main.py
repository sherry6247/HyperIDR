'''
Author: Sicen Liu
Date: 2024-12-05 11:47:26
LastEditTime: 2026-01-04 14:47:24
FilePath: /liusicen/methods/Prompt_pool_IDP/predict_main.py
Description: the train part 

Copyright (c) 2024 by ${liusicen_cs@outlook.com}, All Rights Reserved. 
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
import json

from src.model import Multi_scale_expert_dynamic_Model,ConditionalIDRModel
from torch.optim import AdamW
from utils.args_config import parse_args
from utils.data_processing import load_dataset, proteinDataset, kfold_split, collate_fn_batch, CAID3_collate_fn_batch, collect_span_embeddings_from_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from src.Trainer import Trainer
from src.evaluate import predict


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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

    device = torch.device('cpu')
    args.device = device
    
    criterion = {
        "ConditionalIDRModel" : nn.BCEWithLogitsLoss(reduction='none'),
    }
    args.criterion = criterion[args.model_name]
    
    # 每种特征分别进行训练并保存模型，对比实验结果选择性能最好的traditional biology features和PSSM-based features
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

    test_dataset = load_dataset(args, args.test_dataset_path, dataset_type=args.test_dataset)
    testDataset = proteinDataset(test_dataset)
    if args.test_dataset in ['CASP','SL329','CAID3_disorder_pdb', 'CAID3_disorder_nox', 'CAID2_disorder_nox', 'CAID2_disorder_pdb']:
        test_collate_fn_with_params = partial(CAID3_collate_fn_batch, args=args)
    else:
        test_collate_fn_with_params = partial(collate_fn_batch, args=args)
    test_dataloader = DataLoader(testDataset,
                                sampler=SequentialSampler(testDataset),
                                batch_size=1,
                                collate_fn=test_collate_fn_with_params)
    
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
    
    
    if args.model_name == 'ConditionalIDRModel':
        model = ConditionalIDRModel(d_model=feature_dim, num_prompts=args.pool_size, top_k=args.top_k, branch_hidden=args.hidden_size, backbone=args.backbone, backbone_depth=args.num_layers)

    
    try:
        model_dict = torch.load(args.saved_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict['model_state_dict'])
        print("=====EPOCH:{}====\n======K:{}======\n=======SEED:{}======".format(model_dict['epoch'], model_dict['k'], model_dict['seed']))
    except Exception as e:
        model.load_state_dict(args.saved_model_path)
    
    
    print("=================TEST =================")
    results, sorted_total_samples = predict(args, model, test_dataloader)
    
    result_path = os.path.join(os.path.dirname(args.saved_model_path), 'predict_result_thr-{}.txt'.format(args.threshold))
    with open(result_path, 'w') as file:
        new_results = {}
        for  k, v in results.items():
            new_results[k] = float(v)
        json.dump(new_results, file)  # 写入字典
        file.write('\n')
        file.write('====='*5) 
        for id, sample in sorted_total_samples.items():
            file.write(f'\n>{id}:\n')
            seq, label, logit = sample
            for sq, ll , lt in zip(seq, label, logit):
                file.write("{}\t{}\t{}\n".format(sq, ll, lt))



if __name__ == "__main__":
    main()