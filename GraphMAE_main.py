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

    def forward(self, g, h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        h = h * g.ndata['norm']
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'),
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




# class SAGELayer(nn.Module):
#     """ one layer of GraphSAGE with gcn aggregator """
#     def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
#         super(SAGELayer, self).__init__()
#         self.linear_neigh = nn.Linear(input_dim, output_dim, bias=False)
#         # self.linear_self = nn.Linear(input_dim, output_dim, bias=False)
#         self.activation = activation
#         if dropout:
#             self.dropout = nn.Dropout(p=dropout)
#         else:
#             self.dropout = 0
#         self.init_params()

#     def init_params(self):
#         """ Initialize weights with xavier uniform and biases with all zeros """
#         for param in self.parameters():
#             if len(param.size()) == 2:
#                 nn.init.xavier_uniform_(param)
#             else:
#                 nn.init.constant_(param, 0.0)

#     def forward(self, adj, h):
#         # using GCN aggregator
#         if self.dropout:
#             h = self.dropout(h)
#         x = adj @ h
#         x = self.linear_neigh(x)
#         # x_neigh = self.linear_neigh(x)
#         # x_self = self.linear_self(h)
#         # x = x_neigh + x_self
#         if self.activation:
#             x = self.activation(x)
#         # x = F.normalize(x, dim=1, p=2)
#         return x


# class GATLayer(nn.Module):
#     """ one layer of GAT """
#     def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
#         super(GATLayer, self).__init__()
#         self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
#         self.activation = activation
#         self.n_heads = n_heads
#         self.attn_l = nn.Linear(output_dim, self.n_heads, bias=False)
#         self.attn_r = nn.Linear(output_dim, self.n_heads, bias=False)
#         self.attn_drop = nn.Dropout(p=0.6)
#         if dropout:
#             self.dropout = nn.Dropout(p=dropout)
#         else:
#             self.dropout = 0
#         if bias:
#             self.b = nn.Parameter(torch.FloatTensor(output_dim))
#         else:
#             self.b = None
#         self.init_params()

#     def init_params(self):
#         """ Initialize weights with xavier uniform and biases with all zeros """
#         for param in self.parameters():
#             if len(param.size()) == 2:
#                 nn.init.xavier_uniform_(param)
#             else:
#                 nn.init.constant_(param, 0.0)

#     def forward(self, adj, h):
#         if self.dropout:
#             h = self.dropout(h)
#         x = h @ self.W # torch.Size([2708, 128])
#         # calculate attentions, both el and er are n_nodes by n_heads
#         el = self.attn_l(x)
#         er = self.attn_r(x) # torch.Size([2708, 8])
#         if isinstance(adj, torch.sparse.FloatTensor):
#             nz_indices = adj._indices()
#         else:
#             nz_indices = adj.nonzero().T
#         attn = el[nz_indices[0]] + er[nz_indices[1]] # torch.Size([13264, 8])
#         attn = F.leaky_relu(attn, negative_slope=0.2).squeeze()
#         # reconstruct adj with attentions, exp for softmax next
#         attn = torch.exp(attn) # torch.Size([13264, 8]) NOTE: torch.Size([13264]) when n_heads=1
#         if self.n_heads == 1:
#             adj_attn = torch.zeros(size=(adj.size(0), adj.size(1)), device=adj.device)
#             adj_attn.index_put_((nz_indices[0], nz_indices[1]), attn)
#         else:
#             adj_attn = torch.zeros(size=(adj.size(0), adj.size(1), self.n_heads), device=adj.device)
#             adj_attn.index_put_((nz_indices[0], nz_indices[1]), attn) # torch.Size([2708, 2708, 8])
#             adj_attn.transpose_(1, 2) # torch.Size([2708, 8, 2708])
#         # edge softmax (only softmax with non-zero entries)
#         adj_attn = F.normalize(adj_attn, p=1, dim=-1)
#         adj_attn = self.attn_drop(adj_attn)
#         # message passing
#         x = adj_attn @ x # torch.Size([2708, 8, 128])
#         if self.b is not None:
#             x = x + self.b
#         if self.activation:
#             x = self.activation(x)
#         if self.n_heads > 1:
#             x = x.flatten(start_dim=1)
#         return x # torch.Size([2708, 1024])


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

class VGAE(nn.Module):
    """ GAE/VGAE as edge prediction model """
    def __init__(self, in_feats, n_hidden, dim_z, activation, gae=False):
        super(VGAE, self).__init__()
        self.gae = gae
        self.gcn_base = GCNLayer(in_feats, n_hidden, 1, None, 0, bias=False)
        self.gcn_mean = GCNLayer(n_hidden, dim_z, 1, activation, 0, bias=False)
        self.gcn_logstd = GCNLayer(n_hidden, dim_z, 1, activation, 0, bias=False)

    def forward(self, adj, features):
        # GCN encoder
        hidden = self.gcn_base(adj, features)
        self.mean = self.gcn_mean(adj, hidden)
        if self.gae:
            # GAE (no sampling at bottleneck)
            Z = self.mean
        else:
            # VGAE
            self.logstd = self.gcn_logstd(adj, hidden)
            gaussian_noise = torch.randn_like(self.mean)
            sampled_Z = gaussian_noise*torch.exp(self.logstd) + self.mean
            Z = sampled_Z
        # inner product decoder
        adj_logits = Z @ Z.T
        return adj_logits
    
class GCN_model(nn.Module):
    """ GNN as node classification model """
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN_model, self).__init__()
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


class GNN(nn.Module):
    """ GNN as node classification model """
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, gnnlayer_type='gcn'):
        super(GNN, self).__init__()
        heads = [1] * (n_layers + 1)
        if gnnlayer_type == 'gcn':
            gnnlayer = GCNLayer
        # elif gnnlayer_type == 'gsage':
        #     gnnlayer = SAGELayer
        # elif gnnlayer_type == 'gat':
        #     gnnlayer = GATLayer
        #     if dim_feats in (50, 745, 12047): # hard coding n_heads for large graphs
        #         heads = [2] * n_layers + [1]
        #     else:
        #         heads = [8] * n_layers + [1]
        #     dim_h = int(dim_h / 8)
        #     dropout = 0.6
        #     activation = F.elu
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(gnnlayer(in_feats, n_hidden, heads[0], activation, 0))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(gnnlayer(n_hidden*heads[i], n_hidden, activation, dropout))
        # output layer
        self.layers.append(gnnlayer(n_hidden*heads[-2], n_classes, None, dropout))

    def forward(self, adj, features):
        h = features
        for layer in self.layers:
            h = layer(adj, h)
        return h
    
class GAugMAE_model(nn.Module):
    def __init__(self, 
                 in_feats,
                 n_hidden,
                 dim_z,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 device,
                 gnnlayer_type,
                 temperature=1,
                 gae=False,
                 sample_type='add_sample',
                 alpha = 1):
        super(GAugMAE_model, self).__init__()
        self.device = device
        self.gnnlayer_type= gnnlayer_type
        self.sample_type=sample_type

        # edge prediction network
        self.ep_net = VGAE(in_feats, n_hidden, dim_z, activation, gae=gae)

        # node classification network
        self.nc_net = GNN(in_feats, n_hidden, n_classes, n_layers, activation, dropout, gnnlayer_type=gnnlayer_type)

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
            D_norm = torch.diag(torch.pow(adj.sum(1), -0.5)).to(self.device)
            adj = D_norm @ adj @ D_norm
        elif self.gnnlayer_type == 'gat':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
        elif self.gnnlayer_type == 'gsage':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
            adj = F.normalize(adj, p=1, dim=1)
        return adj

    def forward(self, adj, adj_orig, features):
        adj_logits = self.ep_net(adj, features)
        if self.sample_type == 'edge':
            adj_new = self.sample_adj_edge(adj_logits, adj_orig, self.alpha)
        elif self.sample_type == 'add_round':
            adj_new = self.sample_adj_add_round(adj_logits, adj_orig, self.alpha)
        elif self.sample_type == 'rand':
            adj_new = self.sample_adj_random(adj_logits)
        elif self.sample_type == 'add_sample':
            if self.alpha == 1:
                adj_new = self.sample_adj(adj_logits)
            else:
                adj_new = self.sample_adj_add_bernoulli(adj_logits, adj_orig, self.alpha)
        adj_new_normed = self.normalize_adj(adj_new)
        nc_logits = self.nc_net(adj_new_normed, features)
        return nc_logits, adj_logits


    
class GAugMAE(object):
    def __init__(self, adj, features, labels, tvt_nids, alpha=1, gae=True, cuda=-1, hidden_size=128, n_layers=1, epochs=200, seed=-1, lr=1e-2, weight_decay=5e-4, dropout=0.5, print_progress=True, dropedge=0, pos_weight=0.3, norm_w=1):
        self.t = time.time()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.print_progress = print_progress
        self.dropedge = dropedge
        self.pos_weight = pos_weight
        self.norm_w = norm_w
        self.gae=gae
        self.alpha=alpha
        # config device
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda%8}' if cuda>=0 else 'cpu')
        # fix random seeds if needed
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.load_data(adj, features, labels, tvt_nids)

        self.model = GAugMAE_model(self.features.size(1),
                               hidden_size,
                               self.n_class,
                               n_layers,
                               F.relu,
                               dropout)
        # move everything to device
        self.model.to(self.device)

    def load_data(self, adj, adj_eval, features, labels, tvt_nids):
        if isinstance(features, torch.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)
        if self.features.size(1) in (1433, 3703):
            self.features = F.normalize(self.features, p=1, dim=1)
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
        # adj for training
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        self.adj = adj
        adj = sp.csr_matrix(adj)
        self.G = DGLGraph(self.adj)
        # normalization (D^{-1/2})
        degs = self.G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.device)
        self.G.ndata['norm'] = norm.unsqueeze(1)
        # adj for inference
        assert sp.issparse(adj_eval)
        if not isinstance(adj_eval, sp.coo_matrix):
            adj_eval = sp.coo_matrix(adj_eval)
        adj_eval.setdiag(1)
        adj_eval = sp.csr_matrix(adj_eval)
        self.adj_eval = adj_eval
        self.G_eval = DGLGraph(self.adj_eval)
        # normalization (D^{-1/2})
        degs_eval = self.G_eval.in_degrees().float()
        norm_eval = torch.pow(degs_eval, -0.5)
        norm_eval[torch.isinf(norm_eval)] = 0
        norm_eval = norm_eval.to(self.device)
        self.G_eval.ndata['norm'] = norm_eval.unsqueeze(1)

    def dropEdge(self):
        upper = sp.triu(self.adj, 1) # 将输入矩阵转换为上三角矩阵，即只保留矩阵的上三角部分（不包括对角线）
        n_edge = upper.nnz # 计算上三角矩阵中非零元素的个数，即图中的边数
        n_edge_left = int((1 - self.dropedge) * n_edge) #根据给定的参数计算需要保留的边数
        index_edge_left = np.random.choice(n_edge, n_edge_left, replace=False) #从边的索引中随机选择n_edge_left个索引，用于保留这些边
        data = upper.data[index_edge_left]
        row = upper.row[index_edge_left]
        col = upper.col[index_edge_left]
        adj = sp.coo_matrix((data, (row, col)), shape=self.adj.shape)
        adj = adj + adj.T # 将稀疏矩阵 adj 与其转置相加，得到对称的邻接矩阵
        adj.setdiag(1) # 将邻接矩阵的对角线元素设置为 1，表示每个节点与自身存在连接
        self.G = DGLGraph(adj)
        # normalization (D^{-1/2})
        degs = self.G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.device)
        self.G.ndata['norm'] = norm.unsqueeze(1)

    # def pretrain_ep_net(self, model, adj_orig, norm_w, pos_weight, n_epochs):
    #     """ pretrain the edge prediction network """
    #     adj = self.adj
    #     features = self.features

    #     optimizer = torch.optim.Adam(model.ep_net.parameters(),
    #                                  lr=self.lr)
    #     model.train()
    #     for epoch in range(n_epochs):
    #         adj_logits = model.ep_net(adj, features)
    #         loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
    #         if not self.gae:
    #             mu = model.ep_net.mean
    #             lgstd = model.ep_net.logstd
    #             kl_divergence = 0.5/adj_logits.size(0) * (1 + 2*lgstd - mu**2 - torch.exp(2*lgstd)).sum(1).mean()
    #             loss -= kl_divergence
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
    #         ep_auc, ep_ap = self.eval_edge_pred(adj_pred, self.val_edges, self.edge_labels)
    #         self.logger.info('EPNet pretrain, Epoch [{:3}/{}]: loss {:.4f}, auc {:.4f}, ap {:.4f}'
    #                     .format(epoch+1, n_epochs, loss.item(), ep_auc, ep_ap))
            
    def fit(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        # data
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        # loss function for node classification
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()

        best_vali_acc = 0.0
        best_logits = None
        for epoch in range(self.epochs):
            if self.dropedge > 0:
                self.dropEdge()
            self.model.train()
            nc_logits, adj_logits = self.model(self.G_eval, self.G, features)
            # losses
            nc_l = nc_criterion(nc_logits[self.train_nid], labels[self.train_nid])
            adj_l = self.norm_w * F.binary_cross_entropy_with_logits(adj_logits, self.adj_orig, pos_weight=self.pos_weight)
            if not self.gae:
                mu = self.model.ep_net.mean
                lgstd = self.model.ep_net.logstd
                kl_divergence = 0.5/adj_logits.size(0) * (1 + 2*lgstd - mu**2 - torch.exp(2*lgstd)).sum(1).mean()
                adj_l -= kl_divergence

            l = (1-self.alpha)*nc_l + self.alpha*adj_l
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # validate with original graph (without dropout)
            self.model.eval()
            with torch.no_grad():
                nc_logits_eval, adj_logits_eval = self.model(self.G_eval, features).detach().cpu()
            vali_acc, _ = self.eval_node_cls(nc_logits_eval[self.val_nid], labels[self.val_nid].cpu())
            if self.print_progress:
                print('Epoch [{:2}/{}]: loss: {:.4f}, vali acc: {:.4f}'.format(epoch+1, self.epochs, l.item(), vali_acc))
            if vali_acc > best_vali_acc:
                best_vali_acc = vali_acc
                best_logits = nc_logits_eval
                test_acc, conf_mat = self.eval_node_cls(nc_logits_eval[self.test_nid], labels[self.test_nid].cpu())
                if self.print_progress:
                    print(f'                 test acc: {test_acc:.4f}')
        if self.print_progress:
            print(f'Final test results: acc: {test_acc:.4f}')
        del self.model, features, labels, self.G
        torch.cuda.empty_cache()
        gc.collect()
        t = time.time() - self.t
        return test_acc, best_vali_acc, best_logits

    def eval_node_cls(self, logits, labels):
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

    params_all = json.load(open('best_parameters.json', 'r'))
    params = params_all['GAugO'][args.dataset][args.gnn]
    print(params)

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

    accs = []
    for _ in range(3):
        model = GAugMAE(adj_orig, features, labels, tvt_nids, alpha=1, gae=True, cuda='gpu', hidden_size=128, n_layers=1, epochs=200, seed=-1, lr=1e-2, weight_decay=5e-4, dropout=0.5, print_progress=True, dropedge=0, pos_weight=0.3, norm_w=1)
        print(model)