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




parser = argparse.ArgumentParser(description='single')
parser.add_argument('--dataset', type=str, default='CoauthorshipCora')
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



def objective(trial):

    def train(net, X, G, lbls, train_idx, optimizer, epoch):
        net.train()

        st = time.time()
        optimizer.zero_grad()
        outs = net(X, G)
        outs, lbls = outs[train_idx], lbls[train_idx]
        loss = F.cross_entropy(outs, lbls)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
        return loss.item()


    @torch.no_grad()
    def infer(net, X, G, lbls, idx, test=False):
        net.eval()
        outs = net(X, G)
        outs, lbls = outs[idx], lbls[idx]
        if not test:
            res = evaluator.validate(lbls, outs)
        else:
            res = evaluator.test(lbls, outs)
        return res

    data = CoauthorshipCora()
    n = data['num_vertices']
    m = len(data['edge_list'])
    
    with open('hypergraph_recon/CoauthorshipCora_probability.pkl', 'rb') as file:
        adj_matrix = pickle.load(file)
    
    border = trial.suggest_discrete_uniform('border', 0.95, 1, 0.001)
    
    edge_recon = data['edge_list']
    for i in range(n, n+m):
        edge = []
        for j in range(n):
            if adj_matrix[i][j] > border:
                edge.append(j)
        edge_recon.append(edge)
    for i in range(0,n):
        for j in range(i+1,n):
            if adj_matrix[i][j]>border:
                edge_recon.append([i,j])
    
    set_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])


    X, lbl = data["features"], data["labels"]
    #HG = Hypergraph(data["num_vertices"],edge_recon)
    HG = Hypergraph(data['num_vertices'], data['edge_list'])
    print(HG)
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]
    net = HGNN(data["dim_features"], 128, data["num_classes"])
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    '''
    '''
    X, lbl = X.to(device), lbl.to(device)
    HG = HG.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        # train
        train(net, X, HG, lbl, train_mask, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, HG, lbl, val_mask)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)
    res = infer(net, X, HG, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
    return res['accuracy']

if __name__ == "__main__":
    
    study = optuna.create_study(study_name = 'HGAugM_CoauthorshipCora_study',direction="maximize")
    
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    

