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

data = CocitationPubmed()

edges = data['edge_list']
labels = data['labels']

dic = {}

for edge in edges:
    edge= list(edge)
    edge_label = labels[edge]
    labels_list = set([label.item() for label in edge_label])
    x = len(labels_list)
    if x not in dic:
        dic[x] = 0
    dic[x] += 1

total = sum(dic.values())

dic = {key: (value / total) for key, value in dic.items()}
dic = {key: dic[key] for key in sorted(dic)}

dic = {key: f'{value * 100:.2f}%' for key, value in dic.items()}

print(dic)

di = {}

for edge in edges:
    edge= list(edge)
    
    for x in edge:
        if x not in di:
            di[x] = set()
        for y in edge:
            di[x].add(y)

stat = {}

for x in di:
    neibor_label = labels[list(di[x])]
    labels_list = set([label.item() for label in neibor_label])
    num = len(labels_list)
    if (num-1) not in stat:
        stat[num-1] = 0
    stat[num-1] += 1

total = sum(stat.values())

stat = {key: (value / total) for key, value in stat.items()}
stat = {key: stat[key] for key in sorted(stat)}
stat = {key: f'{value * 100:.2f}%' for key, value in stat.items()}

print(stat)
