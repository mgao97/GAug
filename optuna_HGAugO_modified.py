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
from dhg.data import Cooking200
from dhg.random import set_seed
import dhg
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from dhg.structure.hypergraphs import Hypergraph
import math
import time
import gc
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser(description='single')
parser.add_argument('--dataset', type=str, default='cooking200')
parser.add_argument('--gnn', type=str, default='gcn')
parser.add_argument('--gpu', type=str, default='-1')
# parser.add_argument('--layers', type=int, default=-1)
# parser.add_argument('--add_train', type=int, default=-1)
# parser.add_argument('--feat_norm', type=str, default='row')
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

args = parser.parse_args()

if args.gpu == '-1':
    gpu = -1
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpu = 0

data = Cooking200()
args.dataset = data
print(data['labels'])
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


def adjacency_matrix(hg, s=1, weight=False):
        r"""
        The :term:`s-adjacency matrix` for the dual hypergraph.

        Parameters
        ----------
        s : int, optional, default 1

        Returns
        -------
        adjacency_matrix : scipy.sparse.csr.csr_matrix

        """

        tmp_H = hg.H.to_dense().numpy()
        A = tmp_H @ (tmp_H.T)
        A[np.diag_indices_from(A)] = 0
        if not weight:
            A = (A >= s) * 1

        del tmp_H
        gc.collect()

        return csr_matrix(A)

def objective(trial):
    data = Cooking200()
    # hg = Hypergraph(data["num_vertices"], data["edge_list"])
    # features = torch.eye(data['num_vertices'])
    # adj_matrix = adjacency_matrix(hg, s=1, weight=False)
    # train_index, val_index, test_index = np.where(data['train_mask'])[0], np.where(data['val_mask'])[0], np.where(data['test_mask'])[0]
    # train_nid = torch.tensor(train_index,dtype=torch.long)
    # val_nid = torch.tensor(val_index,dtype=torch.long)
    # test_nid = torch.tensor(test_index,dtype=torch.long)

    lr = 0.005 if layer_type == 'gat' else 0.01
    # if args.layers > 0:
    #     n_layers = args.layers
    # else:
    #     n_layers = 1
    #     if jk:
    #         n_layers = 3
    # feat_norm = args.feat_norm
    # if data == 'ppi':
    #     feat_norm = 'col'
    # elif ds in ('blogcatalog', 'flickr'):
    #     feat_norm = 'none'
    change_frac = trial.suggest_discrete_uniform('alpha', 0, 1, 0.01)
    beta = trial.suggest_discrete_uniform('beta', 0.0, 4.0, 0.1)
    temp = trial.suggest_discrete_uniform('temp', 0.1, 2.1, 0.1)
    warmup = trial.suggest_int('warmup', 0, 10)
    pretrain_ep = trial.suggest_discrete_uniform('pretrain_ep', 5, 300, 5)
    pretrain_nc = trial.suggest_discrete_uniform('pretrain_nc', 5, 300, 5)
    accs = []
    for _ in range(30):
        model = HyperGAug(data, args.use_bn, gpu, args.hidden_size, args.emb_size, args.epochs, args.seed, args.lr, args.weight_decay, args.dropout, beta, temp, False, name='debug', warmup=warmup, gnnlayer_type=args.gnnlayer_type, alpha=change_frac, sample_type=args.sample_type)
        acc = model.fit(pretrain_ep=int(pretrain_ep), pretrain_nc=int(pretrain_nc))
        accs.append(acc)
    acc = np.mean(accs)
    std = np.std(accs)
    trial.suggest_categorical('dataset', [dataname])
    trial.suggest_categorical('gnn', [gnn])
    trial.suggest_uniform('acc', acc, acc)
    trial.suggest_uniform('std', std, std)
    
    return acc

if __name__ == "__main__":
    db_url = "sqlite:///Cooking200.db"
    
    study = optuna.create_study(study_name = 'cooking200_study', storage = db_url, load_if_exists = True,direction="maximize")
    
    study.optimize(objective, n_trials=400)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    

