import os
import time
import copy
import torch
import pickle
import logging
import argparse
import numpy as np
import scipy.sparse as sp
from collections import Counter
import optuna
import torch

from models.GCN_dgl import GCN
from models.GAT_dgl import GAT
from models.GSAGE_dgl import GraphSAGE
from models.JKNet_dgl import JKNet

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
import time
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from dhg import Graph, Hypergraph
from dhg.data import *
from dhg.models import HGNN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
import pickle


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description='single')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--gnn', type=str, default='gcn')
parser.add_argument('--gpu', type=str, default='-1')
parser.add_argument('--eval_orig', type=int, default=0)
parser.add_argument('--nlayers', type=int, default=-1)
parser.add_argument('--add_train', type=int, default=-1)
args = parser.parse_args()

gpu = args.gpu
if gpu == '-1':
    gpuid = -1
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    gpuid = 0

def sample_graph_det(adj_orig, A_pred, remove_pct, add_pct):
    if remove_pct == 0 and add_pct == 0:
        return copy.deepcopy(adj_orig)
    orig_upper = sp.triu(adj_orig, 1)
    n_edges = orig_upper.nnz
    edges = np.asarray(orig_upper.nonzero()).T
    if remove_pct:
        n_remove = int(n_edges * remove_pct / 100)
        pos_probs = A_pred[edges.T[0], edges.T[1]]
        e_index_2b_remove = np.argpartition(pos_probs, n_remove)[:n_remove]
        mask = np.ones(len(edges), dtype=bool)
        mask[e_index_2b_remove] = False
        edges_pred = edges[mask]
    else:
        edges_pred = edges

    if add_pct:
        n_add = int(n_edges * add_pct / 100)
        # deep copy to avoid modifying A_pred
        A_probs = np.array(A_pred)
        # make the probabilities of the lower half to be zero (including diagonal)
        A_probs[np.tril_indices(A_probs.shape[0])] = 0
        # make the probabilities of existing edges to be zero
        A_probs[edges.T[0], edges.T[1]] = 0
        all_probs = A_probs.reshape(-1)
        e_index_2b_add = np.argpartition(all_probs, -n_add)[-n_add:]
        new_edges = []
        for index in e_index_2b_add:
            i = int(index / A_probs.shape[0])
            j = index % A_probs.shape[0]
            new_edges.append([i, j])
        edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
    adj_pred = sp.csr_matrix((np.ones(len(edges_pred)), edges_pred.T), shape=adj_orig.shape)
    adj_pred = adj_pred + adj_pred.T
    return adj_pred


def tensor_2d_to_csr(tensor_2d):
    nonzero_coords = torch.nonzero(tensor_2d)

    rows, cols = nonzero_coords[:, 0], nonzero_coords[:, 1]
    values = tensor_2d[nonzero_coords[:, 0], nonzero_coords[:, 1]]

    csr_matrix_result = csr_matrix((values.numpy(), (rows.numpy(), cols.numpy())), shape=tensor_2d.shape)

    return csr_matrix_result


def test_gaugm(trial):
    ###data
    data = CocitationCora()
    
    ds = args.dataset
    gnn = args.gnn
    eval_orig = args.eval_orig
    t = time.time()
    #tvt_nids = [torch.nonzero(data['train_mask']).squeeze().numpy(), torch.nonzero(data['val_mask']).squeeze().numpy(), torch.nonzero(data['test_mask']).squeeze().numpy()]
    
    n = data['num_vertices']
    m = len(data['edge_list'])
    num_train = int(0.8 * n)
    num_val = int(0.1 * n)
    num_test = n - num_train - num_val

    # 创建训练、验证和测试节点列表
    train_nids = np.arange(num_train)
    val_nids = np.arange(num_train, num_train + num_val)
    test_nids = np.arange(num_train + num_val, n)


    # 组合成tvt_nids列表
    tvt_nids = [train_nids, val_nids, test_nids]
    

    
    edges = []
    features = data['features']
    labels = data['labels']
    for i in range(m):
        features_tensor = []
        for x in data['edge_list'][i]:
            edges.append([x,i+n])
            features_tensor.append(data['features'][x])
        features_tensor = torch.stack(features_tensor)
        mean_features = torch.mean(features_tensor, dim = 0)
        features = torch.vstack((features, mean_features))
        tensor_list = data['labels'][list(data['edge_list'][i])]
        labels_list = [label.item() for label in tensor_list]
        labels_counter = Counter(labels_list)
        most_common_label = labels_counter.most_common(1)[0][0]
        labels = torch.cat([labels, torch.tensor([most_common_label])])

    edges = np.asarray(edges)
    row = edges.T[0]
    col = edges.T[1]
    adj_mat = sp.csr_matrix((np.ones_like(row), (row, col)), shape=(n+m, n+m))
    adj_orig = adj_mat + adj_mat.T

    features = tensor_2d_to_csr(features)
    
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())

    #data
    with open('hypergraph_recon/CocitationCora_probability_upd.pkl', 'rb') as file:
        A_pred = pickle.load(file)
    
    A_pred = A_pred.numpy()


    # sample the graph
    remove_pct = trial.suggest_int('remove_pct', 0, 80)
    add_pct = trial.suggest_int('add_pct', 0, 80)
    adj_pred = sample_graph_det(adj_orig, A_pred, remove_pct, add_pct)
  
    if gnn == 'gcn':
        GNN = GCN
    elif gnn == 'gat':
        GNN = GAT
    elif gnn == 'gsage':
        GNN = GraphSAGE
    elif gnn == 'jknet':
        GNN = JKNet
    accs = []
    
    for _ in range(30):
        if eval_orig > 0:
            if args.nlayers > 0:
                model = GNN(adj_pred, copy.deepcopy(adj_orig), features, labels, tvt_nids, print_progress=True, cuda=gpuid, epochs=200, n_layers=args.nlayers)
            else:
                model = GNN(adj_pred, copy.deepcopy(adj_orig), features, labels, tvt_nids, print_progress=True, cuda=gpuid, epochs=200)
        else:
            if args.nlayers > 0:
                model = GNN(adj_pred, adj_pred, features, labels, tvt_nids, print_progress=False, cuda=gpuid, epochs=400, n_layers=args.nlayers)
            else:
                model = GNN(adj_pred, adj_pred, features, labels, tvt_nids, print_progress=False, cuda=gpuid, epochs=400)
        acc, _, _ = model.fit()
        accs.append(acc)
    acc = np.mean(accs)
    std = np.std(accs)
    # print results
    ev = 'e-orig' if eval_orig else 'e-pred'
    trial.suggest_categorical('dataset', [ds])
    trial.suggest_categorical('gnn', [gnn])
    trial.suggest_categorical('eval_orig', [eval_orig])
    return acc
    

if __name__ == "__main__":
    logging.info('start')
    study = optuna.create_study(direction='maximize')
    study.optimize(test_gaugm, n_trials=400)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))





