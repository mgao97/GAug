import gc
import math
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from sklearn.metrics import f1_score

import logging
import pyro
from itertools import combinations
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle

import os
import json
import argparse
from typing import Optional
from itertools import chain
from functools import partial
import random
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from graphmae.utils import create_norm, drop_edge

# scaled cosine similarity loss
def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

# 计算特征重构损失
def feature_reconstruction_loss(original_features, reconstructed_features):
    mse_loss = nn.MSELoss()
    loss = mse_loss(original_features, reconstructed_features)
    # 归一化损失
    num_features = original_features.shape[1]
    loss /= num_features
    return loss

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True

def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng



def scipysp_to_pytorchsp(sp_mx):
    """ converts scipy sparse matrix to pytorch sparse matrix """
    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx


def eval_node_cls(logits, labels):
        # preds = torch.argmax(logits, dim=1)
        # correct = torch.sum(preds == labels)
        # acc = correct.item() / len(labels)
        if len(labels.size()) == 2:
            preds = torch.round(torch.sigmoid(logits))
        else:
            preds = torch.argmax(logits, dim=1)
        micro_f1 = f1_score(labels, preds, average='micro')
        # calc confusion matrix
        # conf_mat = np.zeros((self.n_class, self.n_class))
        # for i in range(len(preds)):
        #     conf_mat[labels[i], preds[i]] += 1
        return micro_f1, 1


class MultipleOptimizer():
    """ a class that wraps multiple optimizers """
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def update_lr(self, op_index, new_lr):
        """ update the learning rate of one optimizer
        Parameters: op_index: the index of the optimizer to update
                    new_lr:   new learning rate for that optimizer """
        for param_group in self.optimizers[op_index].param_groups:
            param_group['lr'] = new_lr


class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class CeilNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.ceil()

    @staticmethod
    def backward(ctx, g):
        return g

# GCN Layer的实现：通过图的消息传递操作，更新节点的特征
# 定义参数包括：输入特征数、输出特征数、激活函数、丢弃概率、是否使用偏执项
class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h): # 输入是DGL图对象g和输入节点特征h
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        h = h * g.ndata['norm']
        g.ndata['h'] = h
        g.update_all(fn.copy_u('h', 'm'),
                     fn.sum(msg='m', out='h'))
        h = g.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * g.ndata['norm']
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h

# 多层GCN模型用于节点分类：
# 定义参数包括：输入特征数量、隐藏层特征数量、输出类别数、GCN模型层数、激活函数、丢弃率
class GCN(nn.Module):
    'model for node classification'
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(in_feats, n_hidden, activation, 0.))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.append(GCNLayer(n_hidden, n_classes, None, dropout))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h


class GCNEncoder(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 activation,
                 dropout):
        super(GCNEncoder, self).__init__()
        self.layers = nn.ModuleList()
        # self.gcn_base = GCNLayer(in_feats, n_hidden, None, dropout)

        # self.gcn_mean = GCNLayer(n_hidden, n_hidden, activation, dropout)
        # self.gcn_logstd = GCNLayer(n_hidden, n_hidden, activation, dropout)

        # input layer
        self.layers.append(GCNLayer(in_feats, n_hidden, activation, 0.))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h

class GCNDecoder(nn.Module):
    def __init__(self,
                 n_hidden,
                 in_feats,
                 n_layers,
                 activation,
                 dropout):
        super(GCNDecoder, self).__init__()
        self.layers = nn.ModuleList()
        # self.gcn_base = GCNLayer(in_feats, n_hidden, None, dropout)

        # input layer
        self.layers.append(GCNLayer(n_hidden, n_hidden, activation, 0.))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(n_hidden, in_feats, activation, dropout))

    def forward(self, g, h):
        for layer in self.layers:
            h = layer(g, h)
        return h
    

class AMGAE(nn.Module):
    """ GAE/VGAE as edge prediction model """
    def __init__(self, adj_matrix, features, labels, tvt_nids, in_feats, n_hidden, n_layers, n_classes, activation, feat_drop, gamma=0.5, gae=True, mask_rate=0.3, replace_rate=0.1, loss_fn='sce', alpha_l=2.0, beta_l=1, theta_l=2, sample_type='add_sample',lr=1e-2, weight_decay=5e-4, epochs=500,dropedge=0, drop_edge_rate=0.0, seed=-1, feat_norm='row',temperature=0.2,gnnlayer_type='gcn'):
        super(AMGAE, self).__init__()
        self.gamma=gamma
        self.alpha_l=alpha_l
        self.beta_l=beta_l
        self.theta_l=theta_l
        self.sample_type=sample_type
        self.gae=gae
        self.lr=lr
        self.weight_decay=weight_decay
        self.epochs=epochs
        # self.norm_w = norm_w
        self.dropedge = dropedge
        self.feat_norm=feat_norm
        self.temperature=temperature
        self.gnnlayer_type=gnnlayer_type
        self._drop_edge_rate = drop_edge_rate
        self._mask_rate=mask_rate
        self._replace_rate=replace_rate
        self._mask_token_rate=1-self._replace_rate
        self.encoder = GCNEncoder(in_feats, n_hidden, n_layers, activation, dropout=feat_drop)
        self.decoder = GCNDecoder(n_hidden, in_feats, n_layers, activation, dropout=feat_drop)
        self.nc = GCN(in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout=0)

        self.gcn_base = GCNLayer(in_feats, n_hidden, None, 0, bias=False)
        self.gcn_mean = GCNLayer(n_hidden, n_hidden, activation, 0, bias=False)
        self.gcn_logstd = GCNLayer(n_hidden, n_hidden, activation, 0, bias=False)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_feats))
        # setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

        # fix random seeds if needed
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.load_data(adj_matrix, features, labels, tvt_nids)

    def load_data(self, adj_matrix, features, labels, tvt_nids):
        # prepare data
        # features (torch.FloatTensor)
        if isinstance(features, torch.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)

        # normalize feature matrix if needed
        if self.feat_norm == 'row':
            self.features = F.normalize(self.features, p=1, dim=1)
        elif self.feat_norm == 'col':
            self.features = self.col_normalization(self.features)

        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels
        if len(self.labels.size()) == 1:
            self.n_class = len(torch.unique(self.labels))
        else:
            self.n_class = labels.size(1)
        self.train_nid = tvt_nids[0]
        self.val_nid = tvt_nids[1]
        self.test_nid = tvt_nids[2]

        # original adj_matrix for training vgae (torch.FloatTensor)
        assert sp.issparse(adj_matrix)
        if not isinstance(adj_matrix, sp.coo_matrix):
            adj_matrix = sp.coo_matrix(adj_matrix)
        adj_matrix.setdiag(1)
        self.adj = adj_matrix
        adj = sp.csr_matrix(adj_matrix)
        self.G = DGLGraph(self.adj)

        # weights for log_lik loss when training EP net
        adj_t = self.adj
        norm_w = adj_t.shape[0]**2 / float((adj_t.shape[0]**2 - adj_t.sum()) * 2)
        pos_weight = torch.FloatTensor([float(adj_t.shape[0]**2 - adj_t.sum()) / adj_t.sum()])
        self.pos_weight=pos_weight
        self.norm_w=norm_w

        # self.adj_orig = scipysp_to_pytorchsp(adj_matrix).to_dense()

        # # normalized adj_matrix used as input for ep_net (torch.sparse.FloatTensor)
        # degrees = np.array(adj_matrix.sum(1))
        # degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        # adj_norm = degree_mat_inv_sqrt @ adj_matrix @ degree_mat_inv_sqrt
        # self.adj_norm = scipysp_to_pytorchsp(adj_norm)
        # self.adj = scipysp_to_pytorchsp(adj_norm)
        # self.G = DGLGraph(self.adj)
        # normalization (D^{-1/2})
        degs = self.G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        
        self.G.ndata['norm'] = norm.unsqueeze(1)

        # labels (torch.LongTensor) and train/validation/test nids (np.ndarray)
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels
        self.train_nid = tvt_nids[0]
        self.val_nid = tvt_nids[1]
        self.test_nid = tvt_nids[2]
        # number of classes
        if len(self.labels.size()) == 1:
            self.out_size = len(torch.unique(self.labels))
        else:
            self.out_size = labels.size(1)
        # sample the edges to evaluate edge prediction results
        # sample 10% (1% for large graph) of the edges and the same number of no-edges
        if labels.size(0) > 5000:
            edge_frac = 0.01
        else:
            edge_frac = 0.1
        adj_matrix = sp.csr_matrix(adj_matrix)
        n_edges_sample = int(edge_frac * adj_matrix.nnz / 2)
        # sample negative edges
        neg_edges = []
        added_edges = set()
        while len(neg_edges) < n_edges_sample:
            i = np.random.randint(0, adj_matrix.shape[0])
            j = np.random.randint(0, adj_matrix.shape[0])
            if i == j:
                continue
            if adj_matrix[i, j] > 0:
                continue
            if (i, j) in added_edges:
                continue
            neg_edges.append([i, j])
            added_edges.add((i, j))
            added_edges.add((j, i))
        neg_edges = np.asarray(neg_edges)
        # sample positive edges
        nz_upper = np.array(sp.triu(adj_matrix, k=1).nonzero()).T
        np.random.shuffle(nz_upper)
        pos_edges = nz_upper[:n_edges_sample]
        self.val_edges = np.concatenate((pos_edges, neg_edges), axis=0)
        self.edge_labels = np.array([1]*n_edges_sample + [0]*n_edges_sample)


    def forward(self, adj, x):

        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        self.adj = adj
        adj = sp.csr_matrix(adj)
        self.G = DGLGraph(self.adj)

        # ---- attribute reconstruction ----

        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(self.G, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        # GCN encoder
        hidden = self.encoder(use_g, x)

        self.mean = self.gcn_mean(use_g, hidden)
        if self.gae:
            # GAE (no sampling at bottleneck)
            Z = self.mean
        else:
            # VGAE
            self.logstd = self.gcn_logstd(use_g, hidden)
            gaussian_noise = torch.randn_like(self.mean)
            sampled_Z = gaussian_noise*torch.exp(self.logstd) + self.mean
            Z = sampled_Z
        # remask
        Z[mask_nodes] = 0
        feats_logits = self.decoder(use_g, Z)
        # inner product decoder
        adj_logits = Z @ Z.T

        if self.sample_type == 'edge':
            adj_new = self.sample_adj_edge(adj_logits, adj_orig, self.alpha)
        elif self.sample_type == 'add_round':
            adj_new = self.sample_adj_add_round(adj_logits, adj_orig, self.alpha)
        elif self.sample_type == 'rand':
            adj_new = self.sample_adj_random(adj_logits)
        elif self.sample_type == 'add_sample':
            if self.alpha_l == 1:
                adj_new = self.sample_adj(adj_logits)
            else:
                adj_new = self.sample_adj_add_bernoulli(adj_logits, adj_orig, self.alpha)

        adj_new_normed = self.normalize_adj(adj_new)

        # adj_new_normed = adj_new_normed[mask_nodes]
        # print(adj_new_normed.shape)
        # print('-'*10)
        # assert sp.issparse(adj_new_normed)
        if not isinstance(adj_new_normed, sp.coo_matrix):
            adj_new_normed = sp.coo_matrix(adj_new_normed.long().detach().numpy())
        adj_new_normed.setdiag(1.0)
        self.new_G = DGLGraph(adj_new_normed)
        # normalization (D^{-1/2})
        degs = self.new_G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        self.new_G.ndata['norm'] = norm.unsqueeze(1)


        x_init = x
        x_rec = feats_logits
        # x_init = x[mask_nodes]
        # x_rec = feats_logits[mask_nodes]

        # print(x_rec.shape, x_init.shape)

        if self.gamma:
            x_new = self.sample_feats(x_init, x_rec)

        x_new_normed = self.normalize_feats(x_new)
        
        # print(x_new_normed, x_new_normed.shape)
        # print('+'*10)
        

        nc_logits = self.nc(self.new_G, x_new_normed)

        return x_init, x_rec, adj_logits, nc_logits

    def sample_feats(self, x, x_rec):
        return self.gamma*x+(1-self.gamma)*x_rec

    def sample_adj(self, adj_logits):
        """ sample an adj from the predicted edge probabilities of ep_net """
        edge_probs = adj_logits / torch.max(adj_logits)
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_add_bernoulli(self, adj_logits, adj_orig, alpha):
        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = alpha*edge_probs + (1-alpha)*adj_orig
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_add_round(self, adj_logits, adj_orig, alpha):
        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = alpha*edge_probs + (1-alpha)*adj_orig
        # sampling
        adj_sampled = RoundNoGradient.apply(edge_probs)
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_random(self, adj_logits):
        adj_rand = torch.rand(adj_logits.size())
        adj_rand = adj_rand.triu(1)
        adj_rand = torch.round(adj_rand)
        adj_rand = adj_rand + adj_rand.T
        return adj_rand

    def sample_adj_edge(self, adj_logits, adj_orig, change_frac):
        adj = adj_orig.to_dense() if adj_orig.is_sparse else adj_orig
        n_edges = adj.nonzero().size(0)
        n_change = int(n_edges * change_frac / 2)
        # take only the upper triangle
        edge_probs = adj_logits.triu(1)
        edge_probs = edge_probs - torch.min(edge_probs)
        edge_probs = edge_probs / torch.max(edge_probs)
        adj_inverse = 1 - adj
        # get edges to be removed
        mask_rm = edge_probs * adj
        nz_mask_rm = mask_rm[mask_rm>0]
        if len(nz_mask_rm) > 0:
            n_rm = len(nz_mask_rm) if len(nz_mask_rm) < n_change else n_change
            thresh_rm = torch.topk(mask_rm[mask_rm>0], n_rm, largest=False)[0][-1]
            mask_rm[mask_rm > thresh_rm] = 0
            mask_rm = CeilNoGradient.apply(mask_rm)
            mask_rm = mask_rm + mask_rm.T
        # remove edges
        adj_new = adj - mask_rm
        # get edges to be added
        mask_add = edge_probs * adj_inverse
        nz_mask_add = mask_add[mask_add>0]
        if len(nz_mask_add) > 0:
            n_add = len(nz_mask_add) if len(nz_mask_add) < n_change else n_change
            thresh_add = torch.topk(mask_add[mask_add>0], n_add, largest=True)[0][-1]
            mask_add[mask_add < thresh_add] = 0
            mask_add = CeilNoGradient.apply(mask_add)
            mask_add = mask_add + mask_add.T
        # add edges
        adj_new = adj_new + mask_add
        return adj_new

    def normalize_adj(self, adj):
        if self.gnnlayer_type == 'gcn':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
            # normalize adj with A = D^{-1/2} @ A @ D^{-1/2}
            D_norm = torch.diag(torch.pow(adj.sum(1), -0.5))
            adj = D_norm @ adj @ D_norm
        elif self.gnnlayer_type == 'gat':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
        elif self.gnnlayer_type == 'gsage':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
            adj = F.normalize(adj, p=1, dim=1)
        return adj
    
    def normalize_feats(self, feats):
        mean = feats.mean()
        std = feats.std()
        normed_feats = (feats - mean) / std
        return normed_feats
    
    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def dropEdge(self, adj):
        upper = sp.triu(adj, 1) # 将输入矩阵转换为上三角矩阵，即只保留矩阵的上三角部分（不包括对角线）
        n_edge = upper.nnz # 计算上三角矩阵中非零元素的个数，即图中的边数
        n_edge_left = int((1 - self.dropedge) * n_edge) #根据给定的参数计算需要保留的边数
        index_edge_left = np.random.choice(n_edge, n_edge_left, replace=False) #从边的索引中随机选择n_edge_left个索引，用于保留这些边
        data = upper.data[index_edge_left]
        row = upper.row[index_edge_left]
        col = upper.col[index_edge_left]
        adj = sp.coo_matrix((data, (row, col)), shape=adj.shape)
        adj = adj + adj.T # 将稀疏矩阵 adj 与其转置相加，得到对称的邻接矩阵
        adj.setdiag(1) # 将邻接矩阵的对角线元素设置为 1，表示每个节点与自身存在连接
        self.G = DGLGraph(adj)
        # normalization (D^{-1/2})
        degs = self.G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.device)
        self.G.ndata['norm'] = norm.unsqueeze(1)

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)
            
    def train_nc(self, model, adj_orig, features, tvt_nids, labels):
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        
        # loss function for node classification
        if len(labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()

        best_vali_acc = 0.0
        best_logits = None
        for epoch in tqdm(range(self.epochs)):
            if self.dropedge > 0:
                self.dropEdge()
            model.train()
            x_init, x_rec, adj_logits, nc_logits = model(adj_orig, features)
            # node classification losses
            nc_l = nc_criterion(nc_logits[tvt_nids[0]], labels[tvt_nids[0]])
            # adj reconstruct losses
            print('*'*50)
            print(adj_logits.shape, adj_orig.shape, self.pos_weight.shape)
            adj_l = self.norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=self.pos_weight)
            if not self.gae:
                mu = self.model.ep_net.mean
                lgstd = self.model.ep_net.logstd
                kl_divergence = 0.5/adj_logits.size(0) * (1 + 2*lgstd - mu**2 - torch.exp(2*lgstd)).sum(1).mean()
                adj_l -= kl_divergence
            # feature reconstruct losses
            feats_l = feature_reconstruction_loss(x_init, x_rec)

            # total losses
            l = self.alpha_l * nc_l + self.beta_l *adj_l + self.theta_l * feats_l

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            # validate with original graph (without dropout)
            model.eval()
            with torch.no_grad():
                x_init_eval, x_rec_eval, adj_logits_eval, nc_logits_eval = model(adj_orig, features)
            vali_acc, _ = eval_node_cls(nc_logits_eval[tvt_nids[1]], labels[tvt_nids[1]].cpu())
            if epoch % 10 == 0:
                print('Epoch [{:2}/{}]: loss: {:.4f}, vali acc: {:.4f}'.format(epoch+1, self.epochs, l.item(), vali_acc))
            if vali_acc > best_vali_acc:
                best_vali_acc = vali_acc
                torch.save(model.state_dict(), 'AMGAE_parameters.pth')
                best_logits = nc_logits_eval
                test_acc, conf_mat = eval_node_cls(nc_logits_eval[tvt_nids[2]], labels[tvt_nids[2]].cpu())
                if epoch % 50 == 0:
                    print(f'                 test acc: {test_acc:.4f}')
        
        print(f'Final test results: acc: {test_acc:.4f}')
        del model, features, labels
        torch.cuda.empty_cache()
        gc.collect()
        t = time.time() - self.t
        return test_acc, best_vali_acc, best_logits

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    if args.gpu == '-1':
        gpu = -1
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpu = 0

    tvt_nids = pickle.load(open(f'data/graphs/{args.dataset}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'data/graphs/{args.dataset}_adj.pkl', 'rb'))
    features = pickle.load(open(f'data/graphs/{args.dataset}_features.pkl', 'rb'))
    labels = pickle.load(open(f'data/graphs/{args.dataset}_labels.pkl', 'rb'))
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())

    device = args.gpu
    # txt_nids = tvt_nids.to(device)
    # adj_orig = adj_orig.to(device)
    # features = features.to(device)
    # labels = labels.to(device)
    # print(labels)


    # params_all = json.load(open('best_parameters.json', 'r'))
    # params = params_all['GAugO'][args.dataset][args.gnn]
    # print(params)

    gnn = args.gnn
    layer_type = args.gnn
    jk = False
    if gnn == 'jknet':
        layer_type = 'gsage'
        jk = True
    feat_norm = 'row'
    if args.dataset == 'ppi':
        feat_norm = 'col'
    elif args.dataset in ('blogcatalog', 'flickr'):
        feat_norm = 'none'
    lr = 0.005 if layer_type == 'gat' else 0.01
    n_layers = 1
    if jk:
        n_layers = 3

    model = AMGAE(adj_orig, features, labels, tvt_nids, in_feats=features.shape[1], n_hidden=128, n_layers=2, n_classes=len(list(torch.unique(labels))), activation=F.relu, feat_drop=0.3, gamma=0.5, gae=True, mask_rate=0.3, replace_rate=0.1, loss_fn='sce', alpha_l=1.0, beta_l=0.8, theta_l=2, sample_type='add_sample',lr=1e-2, weight_decay=5e-4, epochs=500, feat_norm=feat_norm)
    # model = model.to(device)
    print('*'*30)
    print(model)
    print('*'*30)

    accs = []
    for _ in range(3):
        test_acc, best_vali_acc, best_logits = model.train_nc(model, adj_orig, features, tvt_nids, labels)

