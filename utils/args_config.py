'''
Date: 2025-07-18 17:16:48
LastEditors: Please set LastEditors
LastEditTime: 2025-12-22 16:39:01
FilePath: /liusicen/methods/Prompt_pool_IDP/utils/args_config.py
Description:  
Copyright: © 2025 liusicen@smbu.edu.cn. All rights reserved.
'''

import os
import numpy as np
import torch
from tqdm import tqdm, trange


DATASET_PATH = "PutyouOwnPath/Datasets/datasets/"
FEATURE_PATH = "PutyouOwnPath/Datasets/"
CAID3_PATH = "PutyouOwnPath/Datasets/CAID3-analysis/"

import argparse
def seed(s):
    if s.isdigit():
        s = int(s)
        if 0 <= s <= 9999:
            return s
        else:
            raise argparse.ArgumentTypeError("Seed must be between 0 and 9999")
    elif s == "random":
        return np.random.randint(0, 9999)
    else:
        raise ValueError("Invalid seed value")

def parse_args():
    parser = argparse.ArgumentParser(description='Hybirds')
    
    # Task
    parser.add_argument("--model_name", type=str, default="Prompt_pool_IDP")
    parser.add_argument("--pssm", type=int, default=0, help="Utilize pssm feature")
    parser.add_argument("--drBERT", type=int, default=0, help="Utilize DrBERT feature")
    parser.add_argument("--esm2", type=int, default=0, help="Utilize esm2 feature")
    
    # Dataset parameters
    parser.add_argument("--train_dataset_path", type=str, default=DATASET_PATH + "DM4229_training.fasta")
    parser.add_argument("--test_dataset", type=str, default="DISORDER723_test", help="[DISORDER723_test,CASP,SL329]")
    parser.add_argument("--train_dataset", type=str, default= "DM4229_training")
    parser.add_argument("--evo_path", type=str, default=FEATURE_PATH + "Evo_features")
    parser.add_argument("--preT_feature_path", type=str, default=FEATURE_PATH + "PreT_features")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default='0')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--lambda_orth", type=float, default=0.001, help="Orthogonality loss weight")
    parser.add_argument("--kernel_sizes", type=str, default='3,5,7', help="kernel sizes for MixConv, e.g., '3,5,7' or '3, 7, 15, 31, 63")
    parser.add_argument("--prompt_warmup", type=int, default=1, help="If use prompt warmup, 0 for no, 1 for yes")
    parser.add_argument("--warmup_sdr", type=int, default=8, help="The number of warmup SDR prompts, default 8")
    parser.add_argument("--warmup_ldr", type=int, default=8, help="The number of warmup LDR prompts, default 8")
    
    
    # Architecture parameters
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=seed, default='random', help="random seed")
    parser.add_argument("--output_dim", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=1, help="keep the top-k prompt")
    parser.add_argument("--pool_size", type=int, default=16, help="the size of prompt")
    parser.add_argument("--prompt_length", type=int, default=10, help="the prompt token length")
    parser.add_argument("--use_prompt_key", type=int, default=0, help="If use separate prompt key parameters or not, 0 for no, 1 for yes")
    parser.add_argument("--backbone", type=str, default='bilstm', help="backbone model, 'tcn' | 'transformer' | 'mamba' | 'bilstm'")
    parser.add_argument("--tau", type=float, default=1.0, help="temperature for softmax in ConditionalIDRModel")
    
    # Evaluation parameters
    parser.add_argument("--eval_every", type=int, default=100)
    
    # Dropout parameters
    
    # Save parameters
    parser.add_argument("--save_result", type=str, default="/disk3/liusicen/Prompt_pool_IDP/saved_models")
    parser.add_argument("--saved_embedding", type=int, default=1000)
    parser.add_argument("--saved_model_path", type=str, default="/home/liusicen/methods")

    train_data_dict = {
        "Train_723_bc25" : "PutyouOwnPath/Datasets/datasets/training_dataset_25/Train_723_bc25.fasta",
        "Train_494_bc25" : "PutyouOwnPath/Datasets/datasets/training_dataset_25/Train_494_bc25.fasta",
        "Train_329_bc25" : "PutyouOwnPath/Datasets/datasets/training_dataset_25/Train_329_bc25.fasta",
        "Train_casp_bc25" : "PutyouOwnPath/Datasets/datasets/training_dataset_25/Train_casp_bc25.fasta",
        "Train_CAID3_pdb_bc25" : "PutyouOwnPath/Datasets/datasets/training_dataset_25/Train_update_caid3_pdb_bc25.fasta",
        
    }
    
    test_data_dict = {
        "DISORDER723_test": DATASET_PATH + "DISORDER723_test.fasta",
        "DisProt832_test": DATASET_PATH +  "DisProt832_test.fasta",
        "S1_test": DATASET_PATH + "S1_test.fasta",
        "CASP": DATASET_PATH + "CASP.fasta",
        "Disprot504": DATASET_PATH + "Disprot504.fasta",
        "SL329": DATASET_PATH + "SL329.fasta",
        "MXD494": DATASET_PATH + "MXD494.fasta",
        "CAID3_disorder_nox" : CAID3_PATH + "Datasets_updated/disorder_nox.fasta",
        "CAID3_disorder_pdb" : CAID3_PATH + "Datasets_updated/disorder_pdb.fasta",
       
    }

    args = parser.parse_args()

    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

    args.train_dataset_path = train_data_dict[args.train_dataset]
    
    test_dataset_path = test_data_dict[args.test_dataset]
    args.test_dataset_path = test_dataset_path
    
    return args