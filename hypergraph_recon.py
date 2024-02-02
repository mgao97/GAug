import os
import pickle
import argparse
import numpy as np
from collections import Counter
from models.HGAug import HyperGAug
import torch
import optuna
import scipy.sparse as sp
from itertools import combinations
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import pickle
from copy import deepcopy
import torch.optim as optim
from dhg import Hypergraph
from dhg.data import *
from dhg.random import set_seed
import dhg
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from dhg.structure.hypergraphs import Hypergraph
import math
import time
import gc
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from scipy.sparse import csr_matrix
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from vgae.hg_dataloader import DataLoader
from vgae.utils import *
from vgae.models import *
import pickle



parser = argparse.ArgumentParser(description='single')
parser.add_argument('--dataset', type=str, default='CoauthorshipCora')
parser.add_argument('--gnn', type=str, default='gcn')
parser.add_argument('--gpu', type=str, default='-1')
# parser.add_argument('--layers', type=int, default=-1)
# parser.add_argument('--add_train', type=int, default=-1)
# parser.add_argument('--feat_norm', type=str, default='row')
parser.add_argument('--hidden_size',  default=128)
parser.add_argument('--emb_size', default=32)
parser.add_argument('--epochs', default=400)
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
parser.add_argument('--val_frac', type=float, default=0.05)
parser.add_argument('--test_frac', type=float, default=0.1)
#parser.add_argument('--dataset', type=str, default='zkc')
parser.add_argument('--criterion', type=str, default='roc')
parser.add_argument('--no_mask', action='store_true')
parser.add_argument('--gae', action='store_true')
parser.add_argument('--gen_graphs', type=int, default=1)
    # # tmp args for debuging
parser.add_argument("--w_r", type=float, default=1)
parser.add_argument("--w_kl", type=float, default=1)

args = parser.parse_args()


if args.gpu == '-1':
    gpu = -1
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpu = 0

'''

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
'''


args.device = torch.device(f'cuda:{args.gpu}' if int(args.gpu)>=0 else 'cpu')
if args.seed > 0:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

data = CocitationCora()
dataname = 'CocitationCora'
dl = DataLoader(args, data)

if args.gae: args.w_kl = 0

vgae = VGAE(dl.adj_norm.to(args.device), dl.features.size(1), args.hidden_size, args.emb_size, args.gae)

vgae.to(args.device)
vgae = train_model(args, dl, vgae)
adj_matrix =  gen_graphs(args, dl, vgae)



with open('hypergraph_recon/CocitationCora_probability_upd.pkl', 'wb') as file:
    pickle.dump(adj_matrix, file)

