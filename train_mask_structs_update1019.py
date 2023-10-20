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
from torch import optim as optim
import logging
import pyro
from itertools import combinations
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
import copy
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
from dgl.utils import expand_as_pair
from graphmae.utils import create_norm, drop_edge

from collections import namedtuple, Counter
import numpy as np

import torch

import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix

import dgl
from dgl.data import (
    load_data, 
    TUDataset, 
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader

from sklearn.preprocessing import StandardScaler

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)

from graphmae.models.gat import GAT
from graphmae.models.gin import GIN
from graphmae.models.dot_gat import DotGAT

GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset
}

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

class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)

def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer

def node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False):
    model.eval()
    if linear_prob:
        with torch.no_grad():
            x = model.embed(graph.to(device), x.to(device))
            in_feat = x.shape[1]
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    
    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_acc, estp_acc = linear_probing_for_transductive_node_classiifcation(encoder, graph, x, optimizer_f, max_epoch_f, device, mute)
    return final_acc, estp_acc


def linear_probing_for_transductive_node_classiifcation(model, graph, feat, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.CrossEntropyLoss()

    # graph = graph.to(device)
    # x = feat.to(device)
    x = feat

    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"]

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(graph, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(graph, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(graph, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    # (final_acc, es_acc, best_acc)
    return test_acc, estp_test_acc


def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

# scaled cosine similarity loss
def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss 



def load_dataset(dataset_name):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()

    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) % 200 == 0:


            node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)

    # return best_model
    return model

# GCN Layer的实现：通过图的消息传递操作，更新节点的特征
# 定义参数包括：输入特征数、输出特征数、激活函数、丢弃概率、是否使用偏执项
class GraphConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 norm=None,
                 activation=None,
                 residual=True,
                 ):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim

        self.fc = nn.Linear(in_dim, out_dim)

        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._in_feats, self._out_feats, bias=False)
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        self.norm = norm
        if norm is not None:
            self.norm = norm(out_dim)
        self._activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, graph, feat):
        with graph.local_scope():
            aggregate_fn = fn.copy_u('h', 'm')
            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            # if self._norm in ['left', 'both']:
            degs = graph.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm

            
            # aggregate first then mult W
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            
            rst = self.fc(rst)

            # if self._norm in ['right', 'both']:
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)

            if self.norm is not None:
                rst = self.norm(rst)

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

# 多层GCN模型用于节点分类：
# 定义参数包括：输入特征数量、隐藏层特征数量、输出类别数、GCN模型层数、激活函数、丢弃率
class GCN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False
                 ):
        super(GCN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            self.gcn_layers.append(GraphConv(
                in_dim, out_dim, residual=last_residual, norm=last_norm, activation=last_activation))
        else:
            # input projection (no residual)
            self.gcn_layers.append(GraphConv(
                in_dim, num_hidden, residual=residual, norm=norm, activation=create_activation(activation)))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gcn_layers.append(GraphConv(
                    num_hidden, num_hidden, residual=residual, norm=norm, activation=create_activation(activation)))
            # output projection
            self.gcn_layers.append(GraphConv(
                num_hidden, out_dim, residual=last_residual, activation=last_activation, norm=last_norm))

        self.norms = None
        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcn_layers[l](g, h)
            if self.norms is not None and l != self.num_layers - 1:
                h = self.norms[l](h)
            hidden_list.append(h)
        # output projection
        if self.norms is not None and len(self.norms) == self.num_layers:
            h = self.norms[-1](h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)

    def get_embeddings(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcn_layers[l](g, h)
        return h
    
        return h

    def reconstruct_adj_matrix(self, g, h):
        # 在这里生成重构的邻接矩阵
        # g: 图数据，h: GCN层的输出

        # 你需要定义生成邻接矩阵的逻辑，这取决于你的具体需求
        # 以下是一个示例，假设你想要生成一个对称的二元邻接矩阵：
        adj_reconstructed = torch.mm(h, h.t())  # 做内积来生成邻接矩阵
        adj_reconstructed = torch.sigmoid(adj_reconstructed)  # 将内积结果映射到 [0, 1]

        return adj_reconstructed


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


def relabel_tensor(tensor):
    # 获取原始张量中的唯一值（去重）
    unique_values = torch.unique(tensor)

    # 创建一个从0到N-1的映射关系，其中N是唯一值的数量
    mapping = {value.item(): index for index, value in enumerate(unique_values)}

    # 使用映射将原始张量中的值替换为新的映射值
    remapped_tensor = torch.tensor([mapping[value.item()] for value in tensor])

    return remapped_tensor


class MGAE(nn.Module):
    def __init__(self, 
                 in_dim: int,
                num_hidden: int,
                num_layers: int,
                
                nhead: int,
                nhead_out: int,
                activation: str,
                feat_drop: float,
                attn_drop: float,
                negative_slope: float,
                residual: bool,
                norm: Optional[str],
                mask_rate: float = 0.3,
                encoder_type: str = "gcn",
                decoder_type: str = "gcn",
                loss_fn: str = "sce",
                drop_edge_rate: float = 0.0,
                replace_rate: float = 0.1,
                alpha_l: float = 2,
                concat_hidden: bool = False,):
        super(MGAE, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 


        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=dec_num_hidden,
            num_layers=2,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)


    def forward(self, g, x):
        # ---- attribute reconstruction ----
        loss = self.mask_edge_prediction(g, x)
        loss_item = {"loss": loss.item()}
        return loss, loss_item
    

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def add_noise(self, x, mask_nodes, mask_rate=0.3):
        num_nodes = x.shape[0]
        num_mask_nodes = int(mask_rate * num_nodes)
        perm = torch.randperm(num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        noise = torch.randn(num_mask_nodes, x.shape[1])
        x[mask_nodes] = noise
        return x, mask_nodes, keep_nodes
    
    def mask_edge_prediction(self, g, x, mask_rate=0.3):
        num_edges = g.num_edges()
        num_mask_edges = int(mask_rate * num_edges)

        perm = torch.randperm(num_edges)
        mask_edges = perm[:num_mask_edges]
        keep_edges = perm[num_mask_edges:]

        # print('-'*20)
        # print(mask_edges, mask_edges.shape)

        pos_mask_g = g.edge_subgraph(mask_edges)

        num_nodes = pos_mask_g.num_nodes()
        
        # 生成负样本的连边，节点对存在但没有实际边
        neg_mask_edges = []
        while len(neg_mask_edges) < num_mask_edges:
            # 随机选择两个节点
            node1 = torch.randint(0, num_nodes, (num_mask_edges,))
            node2 = torch.randint(0, num_nodes, (num_mask_edges,))
            
            # 确保所选节点不在正样本连边中
            is_duplicate = torch.any(torch.eq(node1, node2))
            
            if not is_duplicate:
                neg_mask_edges.append((node1, node2))

       
        neg_mask_g = DGLGraph()
        
        neg_mask_g.add_edges(neg_mask_edges[0][0].tolist(), neg_mask_edges[0][1].tolist())
        

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pos_mask_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pos_mask_g

        use_nodes = use_g.nodes()
        use_x = x[use_nodes]


        if self._drop_edge_rate > 0:
            neg_use_g, neg_masked_edges = drop_edge(neg_mask_g, self._drop_edge_rate, return_edges=True)
        else:
            neg_use_g = neg_mask_g

        neg_use_nodes = neg_use_g.nodes()
        neg_use_x = x[neg_use_nodes]
        
        use_g = dgl.add_self_loop(use_g)
        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        neg_use_g = dgl.add_self_loop(neg_use_g)
        neg_enc_rep, neg_all_hidden = self.encoder(neg_use_g, neg_use_x, return_hidden=True)
        if self._concat_hidden:
            neg_enc_rep = torch.cat(neg_all_hidden, dim=1)

        # ---- structure reconstruction ----
        # print('-'*50)
        # print(enc_rep.shape)
        rep = self.encoder_to_decoder(enc_rep)
        neg_rep = self.encoder_to_decoder(neg_enc_rep)

        # print(rep.shape)

        
        pos_recon = self.decoder(pos_mask_g, rep)
        neg_recon = self.decoder(neg_mask_g, neg_rep)

        

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        edges_recon_pos = cos(pos_recon[pos_mask_g.edges()[0]], pos_recon[pos_mask_g.edges()[1]])
        edges_recon_neg = cos(neg_recon[neg_mask_g.edges()[0]], neg_recon[neg_mask_g.edges()[1]])
        

        # Concatenate positive and negative edge predictions
        edges_recon = torch.cat((edges_recon_pos, edges_recon_neg))

        # print(edges_recon, edges_recon.shape)
        edges_recon = edges_recon.requires_grad_()

        # calculate loss
        criterion = nn.BCEWithLogitsLoss()

        loss = criterion(edges_recon.float(), torch.cat((torch.ones_like(edges_recon_pos), torch.zeros_like(edges_recon_neg))))
        # loss = criterion(edges_recon, torch.ones_like(edges_recon))
        return loss

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep
    
    def deembed(self, g, rep):
        derep = self.decoder(g, rep)
        return derep


    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
    
class GAug(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers, feat_drop, activation, residual, norm, encoding=False, sample_type="add_sample", alpha=1.0, temperature=0.2, gnnlayer_type="gcn"):
        super(GAug, self).__init__()
        self.sample_type=sample_type
        self.alpha=alpha
        self.temperature = temperature
        self.gnnlayer_type = gnnlayer_type

        self.norms = None
        self.head = nn.Identity()
        self.linear = LogisticRegression(in_dim, hidden_dim)
        self.sample_type=sample_type
        self.num_layers = num_layers
        self.dropout=feat_drop
        self.gcn_layers = nn.ModuleList()
        self.activation = activation

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            self.gcn_layers.append(GraphConv(
                in_dim, num_classes, residual=last_residual, norm=last_norm, activation=last_activation))
        else:
            # input projection (no residual)
            self.gcn_layers.append(GraphConv(
                in_dim, hidden_dim, residual=residual, norm=norm, activation=create_activation(activation)))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gcn_layers.append(GraphConv(
                    hidden_dim, hidden_dim, residual=residual, norm=norm, activation=create_activation(activation)))
            # output projection
            self.gcn_layers.append(GraphConv(
                hidden_dim, num_classes, residual=last_residual, activation=last_activation, norm=last_norm))


    def forward(self, adj_rec, graph, x, lr_f, weight_decay_f, max_epoch_f,return_hidden=False, linear_prob=True, mute=False):

        adj = graph.adjacency_matrix()

        if self.sample_type == 'edge':
            adj_new = self.sample_adj_edge(adj_rec, adj, self.alpha)
        elif self.sample_type == 'add_round':
            adj_new = self.sample_adj_add_round(adj_rec, adj, self.alpha)
        elif self.sample_type == 'rand':
            adj_new = self.sample_adj_random(adj_rec)
        elif self.sample_type == 'add_sample':
            if self.alpha == 1:
                adj_new = self.sample_adj(adj_rec)
            else:
                adj_new = self.sample_adj_add_bernoulli(adj_rec, adj, self.alpha)
        adj_new_normed = self.normalize_adj(adj_new)

        sparse_adj = sp.csr_matrix(adj_new_normed)

        new_graph = dgl.from_scipy(sparse_adj)
        new_graph.ndata["train_mask"] = graph.ndata["train_mask"]
        new_graph.ndata["val_mask"] = graph.ndata["val_mask"]
        new_graph.ndata["test_mask"] = graph.ndata["test_mask"]
        new_graph.ndata["label"] = graph.ndata["label"]
        new_graph.ndata["feat"] = graph.ndata["feat"]

        h = x
        hidden_list = []
        for l in range(self.num_layers):
            print(l)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcn_layers[l](new_graph, h)
            if self.norms is not None and l != self.num_layers - 1:
                h = self.norms[l](h)
            hidden_list.append(h)
        # output projection
        if self.norms is not None and len(self.norms) == self.num_layers:
            h = self.norms[-1](h)
        if return_hidden:
            head, hidden_list = self.head(h), hidden_list
            
        else:
            head = self.head(h)

        encoder = self.linear
        num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
        if not mute:
            print(f"num parameters for finetuning: {sum(num_finetune_params)}")

        optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
        final_acc, estp_acc = linear_probing_for_transductive_node_classiifcation(encoder, new_graph, x, optimizer_f, max_epoch_f, mute)
        return final_acc, estp_acc


    def linear_probing_for_transductive_node_classiifcation(model, graph, feat, optimizer, max_epoch, device, mute=False):
        criterion = torch.nn.CrossEntropyLoss()

        graph = graph.to(device)
        x = feat.to(device)

        train_mask = graph.ndata["train_mask"]
        val_mask = graph.ndata["val_mask"]
        test_mask = graph.ndata["test_mask"]
        labels = graph.ndata["label"]

        best_val_acc = 0
        best_val_epoch = 0
        best_model = None

        if not mute:
            epoch_iter = tqdm(range(max_epoch))
        else:
            epoch_iter = range(max_epoch)

        for epoch in epoch_iter:
            model.train()
            out = model(graph, x)
            loss = criterion(out[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

            with torch.no_grad():
                model.eval()
                pred = model(graph, x)
                val_acc = accuracy(pred[val_mask], labels[val_mask])
                val_loss = criterion(pred[val_mask], labels[val_mask])
                test_acc = accuracy(pred[test_mask], labels[test_mask])
                test_loss = criterion(pred[test_mask], labels[test_mask])
            
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch
                best_model = copy.deepcopy(model)

            if not mute:
                epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

        best_model.eval()
        with torch.no_grad():
            pred = best_model(graph, x)
            estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
        if mute:
            print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
        else:
            print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

        # (final_acc, es_acc, best_acc)
        return test_acc, estp_test_acc


    def sample_adj(self, adj_logits):
        """ sample an adj from the predicted edge probabilities of ep_net """
        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = torch.clamp(edge_probs, 0, 1) # 限制在 [0, 1] 的范围内
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

def build_model(args):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder
    mask_rate = args.mask_rate
    drop_edge_rate = args.drop_edge_rate
    replace_rate = args.replace_rate


    activation = args.activation
    loss_fn = args.loss_fn
    alpha_l = args.alpha_l
    concat_hidden = args.concat_hidden
    num_features = args.num_features


    model = MGAE(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        norm=norm,
        loss_fn=loss_fn,
        drop_edge_rate=drop_edge_rate,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
    )
    return model


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    graph, (num_features, num_classes) = load_dataset(dataset_name)
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        print('8'*20)
        print(model)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        x = graph.ndata["feat"]
        if not load_model:
            model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            
            
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint_struct.pt"))
        if save_model:
            logging.info("Saving Model ...")
            torch.save(model.state_dict(), "checkpoint_struct.pt")
        
        model = model.to(device)
        model.eval()

        rep = model.encoder.get_embeddings(graph, x)
        # print('='*50)
        # print(rep.shape)
        # print('='*50)
        derep = model.encoder.reconstruct_adj_matrix(graph, rep)
        # print('='*50)
        # print(rep.shape, derep.shape)
        # print('='*50)

        # rep = model.embed(graph, x)
        # derep = model(graph, rep)

        # model2 = GAug(args.in_dim, args.hidden_dim, args.num_classes,lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False)

        model2 = GAug(args.num_features, args.num_hidden, num_classes, args.num_layers, args.in_drop, args.activation, args.residual, create_norm(args.norm), encoding=False, sample_type="add_sample", alpha=1.0, temperature=0.2, gnnlayer_type="gcn")
        print('8'*50)
        print(model2)
        

        final_acc, estp_acc = model2(derep, graph, x, lr_f, weight_decay_f, max_epoch_f,return_hidden=False, linear_prob=True, mute=False)

        

        # final_acc, estp_acc = node_classification_evaluation(model2, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)