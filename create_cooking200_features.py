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
import os


feature_dim = 100
feature_noise = 0.1

data = Cooking200()
labels = data['labels']
num_nodes = data['num_vertices']
labels = labels.tolist()

    # then create node features.
num_classes = data['num_classes']
features = np.zeros((num_nodes, num_classes))


features[np.arange(num_nodes), labels] = 1

print(features)    

if feature_dim is not None:
    num_row, num_col = features.shape
    zero_col = np.zeros((num_row, feature_dim - num_col), dtype = features.dtype)
    features = np.hstack((features, zero_col))

features = np.random.normal(features, feature_noise, features.shape)
print(f'number of nodes:{num_nodes}, feature dimension: {features.shape[1]}')

features = torch.FloatTensor(features)
labels = torch.LongTensor(labels)

with open(os.path.join('data', "cooking200_features.pkl"), "wb") as f:
    pickle.dump(features, f)
