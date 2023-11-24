import os
import json
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
from models.HGAug import HyperGAug
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from itertools import combinations
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import pickle
from copy import deepcopy
import torch.optim as optim
from dhg import Hypergraph
from dhg.data import Cooking200
from dhg.random import set_seed
import dhg
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from dhg.structure.hypergraphs import Hypergraph
import math
import time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from scipy.sparse import csr_matrix
# from HGNN import HGNN_model
# from VHGAE import VHGAE_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single')
    parser.add_argument('--dataset', type=str, default='cooking200')
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--gpu', type=str, default='-1')
    parser.add_argument('--hidden_size',  default=128)
    parser.add_argument('--emb_size', default=32)
    parser.add_argument('--epochs', default=200)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--lr', default=1e-2)
    parser.add_argument('--weight_decay', default=5e-4)
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--beta', default=0.5)
    parser.add_argument('--temperature', default=0.2)
    
    parser.add_argument('--warmup', default=3)
    parser.add_argument('--gnnlayer_type', default='gcn')
    parser.add_argument('--alpha', default=1)
    parser.add_argument('--sample_type', default='add_sample')
    parser.add_argument('--use_bn', default=False)

    args = parser.parse_args()

    if args.gpu == '-1':
        gpu = -1
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpu = 0

    data = Cooking200()
    args.dataset = data

    # params_all = json.load(open('best_parameters.json', 'r'))
    # params = params_all['HGAugO'][args.dataset][args.gnn]

    gnn = args.gnn
    layer_type = args.gnn
    # jk = False
    # if gnn == 'jknet':
    #     layer_type = 'gsage'
    #     jk = True
    # feat_norm = 'row'
    if args.dataset == 'ppi':
        feat_norm = 'col'
    elif args.dataset in ('blogcatalog', 'flickr'):
        feat_norm = 'none'
    lr = 0.005 if layer_type == 'gat' else 0.01
    n_layers = 1

    

    accs = []
    for _ in range(30):
        model = HyperGAug(data, args.use_bn, gpu, args.hidden_size, args.emb_size, args.epochs, args.lr, args.weight_decay, args.beta, args.temperature, False, name='debug', gnnlayer_type=args.gnnlayer_type, alpha=args.alpha, sample_type=args.sample_type)
        acc = model.fit(pretrain_ep=200, pretrain_nc=20)
        accs.append(acc)
    print(f'Micro F1: {np.mean(accs):.6f}, std: {np.std(accs):.6f}')
