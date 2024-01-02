import gc
import logging
import numpy as np
import scipy.sparse as sp
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
from models.HGNN import HGNN_model
from models.VHGAE import VHGAE_model
import random
from tqdm import tqdm

class HyperGAug(object):
    def __init__(self, data, use_bn, cuda=-1, hidden_size=128, emb_size=32, epochs=200, seed=42, lr=1e-2, weight_decay=5e-4, dropout=0.5, beta=0.5, temperature=0.2, log=True, name='debug', warmup=3, gnnlayer_type='gcn', alpha=1, sample_type='add_sample'):
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = epochs
        # self.gae = gae
        self.beta = beta
        self.warmup = warmup
        # self.feat_norm = feat_norm
        self.use_bn = use_bn
        self.gnnlayer_type = gnnlayer_type
        # create a logger, logs are saved to GAug-[name].log when name is not None
        if log:
            self.logger = self.get_logger(name)
        else:
            # disable logger if wanted
            # logging.disable(logging.CRITICAL)
            self.logger = logging.getLogger()
        # config device (force device to cpu when cuda is not available)
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda}' if cuda>=0 else 'cpu')
        # log all parameters to keep record
        all_vars = locals()
        self.log_parameters(all_vars)
        # fix random seeds if needed
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # load data
        self.load_data(data, gnnlayer_type)
        # setup the model
        self.model = HGAug_model(self.features.size(1),
                                hidden_size,
                                emb_size,
                                self.out_size,
                                use_bn,
                                dropout,
                                self.device,
                                temperature=temperature,
                                alpha=alpha,
                                sample_type=sample_type)

        
    
    def load_data(self, data, gnnlayer_type):
        """ preprocess data """
        hg = Hypergraph(data["num_vertices"], data["edge_list"])
        #print("ddddd", hg)
        features = torch.eye(data['num_vertices'])
        labels = data['labels']
        train_index, val_index, test_index = np.where(data['train_mask'])[0], np.where(data['val_mask'])[0], np.where(data['test_mask'])[0]

        # features (torch.FloatTensor)
        if isinstance(features, torch.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)
        # # normalize feature matrix if needed
        # if self.feat_norm == 'row':
        #     self.features = F.normalize(self.features, p=1, dim=1)
        # elif self.feat_norm == 'col':
        #     self.features = self.col_normalization(self.features)
        # original adj_matrix for training vgae (torch.FloatTensor)

        adj_matrix = adjacency_matrix(hg, s=1, weight=False)
        assert sp.issparse(adj_matrix)
        if not isinstance(adj_matrix, sp.coo_matrix):
            adj_matrix = sp.coo_matrix(adj_matrix)
        #print("tttt", adj_matrix)
        #print(data["edge_list"],adjacency_matrix[163][0])
        adj_matrix.setdiag(1)
        #print("adj_matrix", adj_matrix(0,384))
        '''rows, cols = adj_matrix.shape
        I = np.identity(min(rows, cols))
        adj_matrix += I
        print(adj_matrix)'''
        
        #print("tttt", adj_matrix)
        self.adj_orig = scipysp_to_pytorchsp(adj_matrix).to_dense()
        # normalized adj_matrix used as input for ep_net (torch.sparse.FloatTensor)
        degrees = np.array(adj_matrix.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj_matrix @ degree_mat_inv_sqrt
        self.adj_norm = scipysp_to_pytorchsp(adj_norm)
        # adj_matrix used as input for nc_net (torch.sparse.FloatTensor)
        
        if gnnlayer_type == 'gcn':
            self.adj = scipysp_to_pytorchsp(adj_norm)
        elif gnnlayer_type == 'gsage':
            adj_matrix_noselfloop = sp.coo_matrix(adj_matrix)
            # adj_matrix_noselfloop.setdiag(0)
            # adj_matrix_noselfloop.eliminate_zeros()
            adj_matrix_noselfloop = sp.coo_matrix(adj_matrix_noselfloop / adj_matrix_noselfloop.sum(1))
            self.adj = scipysp_to_pytorchsp(adj_matrix_noselfloop)
        elif gnnlayer_type == 'gat':
            # self.adj = scipysp_to_pytorchsp(adj_matrix)
            self.adj = torch.FloatTensor(adj_matrix.todense())
        # labels (torch.LongTensor) and train/validation/test nids (np.ndarray)
        
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels
        self.train_nid = torch.tensor(train_index,dtype=torch.long)
        self.val_nid = torch.tensor(val_index, dtype=torch.long)
        self.test_nid = torch.tensor(test_index, dtype=torch.long)
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
        #n_edges_sample = int(edge_frac * adj_matrix.nnz / 2)
        n_edges_sample = int(edge_frac * len(data['edge_list']) / 2)
        # sample negative edges
        hyperedges = []
        for x in data['edge_list']:
            hyperedges.append(frozenset(x))
        nodes_to_neighbors = csr_to_nodes_to_neighbors(adj_matrix)
        list_hyperedges = list(hyperedges)
        node_set = set(range(0,data['num_vertices']))
        neg_edges = []
        for i in tqdm(range(n_edges_sample)):
            sampled_edge = clique_negative_sampling(
                hyperedges, nodes_to_neighbors, list_hyperedges,
                node_set)
            neg_edges.append(sampled_edge)
        # sample positive edges
        
        selected_edges = random.sample(data['edge_list'], n_edges_sample)
        pos_edges = [list(edge) for edge in selected_edges]
        #print(edge_array)
        #pos_edges = np.array(selected_edges, dtype = object)
        
        self.val_edges = pos_edges + neg_edges
        self.edge_labels = np.array([1]*n_edges_sample + [0]*n_edges_sample)
        self.hg = hg

    def pretrain_ep_net(self, model, hg, features, adj_orig, norm_w, pos_weight, n_epochs):
        """ pretrain the edge prediction network """
        optimizer = torch.optim.Adam(model.ep_net.parameters(),
                                    lr=self.lr)
        model.train()
        for epoch in range(n_epochs):
            adj_logits = model.ep_net(features, hg)

            loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
            print('Epoch: {:04d}'.format(epoch+1),'ep_loss_pretrain: {:.4f}'.format(loss.item()))
            # if not self.gae:
            mu = model.ep_net.mean
            lgstd = model.ep_net.logstd
            
            kl_divergence = 0.5/adj_logits.size(0) * (1 + 2*lgstd - mu**2 - torch.exp(2*lgstd)).sum(1).mean()
            loss -= kl_divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
            ep_auc, ep_ap = self.eval_edge_pred(adj_pred, self.val_edges, self.edge_labels)
            self.logger.info('EPNet pretrain, Epoch [{:3}/{}]: loss {:.4f}, auc {:.4f}, ap {:.4f}'
                        .format(epoch+1, n_epochs, loss.item(), ep_auc, ep_ap))

    def pretrain_nc_net(self, model, hg, features, labels, n_epochs):
        """ pretrain the node classification network """
        optimizer = torch.optim.Adam(model.nc_net.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay)
        # loss function for node classification
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.
        for epoch in range(n_epochs):
            model.train()
            nc_logits = model.nc_net(features, hg)
            # losses
            loss = nc_criterion(nc_logits[self.train_nid], labels[self.train_nid])
            #print('Epoch: {:04d}'.format(epoch+1),'nc_loss_pretrain: {:.4f}'.format(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            model.eval()
            with torch.no_grad():
                nc_logits_eval = model.nc_net(features, hg)
            val_acc = self.eval_node_cls(nc_logits_eval[self.val_nid], labels[self.val_nid])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = self.eval_node_cls(nc_logits_eval[self.test_nid], labels[self.test_nid])
                self.logger.info('NCNet pretrain, Epoch [{:2}/{}]: loss {:.4f}, val acc {:.4f}, test acc {:.4f}'
                            .format(epoch+1, n_epochs, loss.item(), val_acc, test_acc))
            else:
                self.logger.info('NCNet pretrain, Epoch [{:2}/{}]: loss {:.4f}, val acc {:.4f}'
                            .format(epoch+1, n_epochs, loss.item(), val_acc))

    def fit(self, pretrain_ep=200, pretrain_nc=20):
        """ train the model """
        # move data to device
        hg = self.hg.to(self.device)
        adj_norm = self.adj_norm.to(self.device)
        adj = self.adj.to(self.device)
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        adj_orig = self.adj_orig.to(self.device)
        model = self.model.to(self.device)
        # weights for log_lik loss when training EP net 
        adj_t = self.adj_orig
        norm_w = adj_t.shape[0]**2 / float((adj_t.shape[0]**2 - adj_t.sum()) * 2)
        pos_weight = torch.FloatTensor([float(adj_t.shape[0]**2 - adj_t.sum()) / adj_t.sum()]).to(self.device)
        # pretrain VGAE if needed
        
        if pretrain_ep:
            self.pretrain_ep_net(model, hg, features, adj_orig, norm_w, pos_weight, pretrain_ep)
        # pretrain GCN if needed

        
        if pretrain_nc:
            self.pretrain_nc_net(model, hg, features, labels, pretrain_nc)
        # optimizers
        
        optims = MultipleOptimizer(torch.optim.Adam(model.ep_net.parameters(),
                                                    lr=self.lr),
                                torch.optim.Adam(model.nc_net.parameters(),
                                                    #lr=self.lr,
                                                    lr = 1e-5,
                                                    weight_decay=self.weight_decay))
        # get the learning rate schedule for the optimizer of ep_net if needed
        if self.warmup:
            ep_lr_schedule = self.get_lr_schedule_by_sigmoid(self.n_epochs, self.lr, self.warmup)
        # loss function for node classification
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()
        # keep record of the best validation accuracy for early stopping
        best_val_acc = 0.
        patience_step = 0
        # train model
        for epoch in range(self.n_epochs):
            # update the learning rate for ep_net if needed
            if self.warmup:
                optims.update_lr(0, ep_lr_schedule[epoch])

            model.train()
            nc_logits, adj_logits = model(features, hg)

            # losses
            loss = nc_loss = nc_criterion(nc_logits[self.train_nid], labels[self.train_nid])
            ep_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
            loss += self.beta * ep_loss
            optims.zero_grad()
            loss.backward()
            optims.step()
            
            # validate (without dropout)
            model.eval()
            with torch.no_grad():
                nc_logits_eval = model.nc_net(features, hg)
            val_acc = self.eval_node_cls(nc_logits_eval[self.val_nid], labels[self.val_nid])
            adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
            ep_auc, ep_ap = self.eval_edge_pred(adj_pred, self.val_edges, self.edge_labels)
            print('Epoch: {:04d}'.format(epoch+1),'ep_loss_val: {:.4f}'.format(ep_loss.item()),'nc_loss_val: {:.4f}'.format(nc_loss.item()))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = self.eval_node_cls(nc_logits_eval[self.test_nid], labels[self.test_nid])
                self.logger.info('Epoch [{:3}/{}]: ep loss {:.4f}, nc loss {:.4f}, ep auc: {:.4f}, ep ap {:.4f}, val acc {:.4f}, test acc {:.4f}'
                            .format(epoch+1, self.n_epochs, ep_loss.item(), nc_loss.item(), ep_auc, ep_ap, val_acc, test_acc))
                patience_step = 0
            else:
                self.logger.info('Epoch [{:3}/{}]: ep loss {:.4f}, nc loss {:.4f}, ep auc: {:.4f}, ep ap {:.4f}, val acc {:.4f}'
                            .format(epoch+1, self.n_epochs, ep_loss.item(), nc_loss.item(), ep_auc, ep_ap, val_acc))
                patience_step += 1
                if patience_step == 100:
                    self.logger.info('Early stop!')
                    break
        # get final test result without early stop
        with torch.no_grad():
            nc_logits_eval = model.nc_net(features, hg)
        test_acc_final = self.eval_node_cls(nc_logits_eval[self.test_nid], labels[self.test_nid])
        # log both results
        self.logger.info('Final test acc with early stop: {:.4f}, without early stop: {:.4f}'
                    .format(test_acc, test_acc_final))
        # release RAM and GPU memory
        del adj, features, labels, adj_orig
        torch.cuda.empty_cache()
        gc.collect()
        return test_acc
        
        
    def log_parameters(self, all_vars):
        """ log all variables in the input dict excluding the following ones """
        # del all_vars['self']
        # del all_vars['adj_matrix']
        # del all_vars['features']
        # del all_vars['labels']
        # del all_vars['tvt_nids']
        self.logger.info(f'Parameters: {all_vars}')

    @staticmethod
    def eval_edge_pred(adj_pred, val_edges, edge_labels):
        logits = []
        for x in val_edges:
            #print(x)
            combinations_list = list(combinations(x, 2))
            val_edge_T = list(map(list, zip(*combinations_list)))
            logits.append(adj_pred[val_edge_T].mean().item())
            #print(adj_pred[val_edge_T].mean().item())
        #print(len(val_edges))
        logits = np.array(logits)
        #print(logits)
        #logits = adj_pred[val_edges.T]
        logits = np.nan_to_num(logits)
        #print(len(logits),logits)
        roc_auc = roc_auc_score(edge_labels, logits)
        ap_score = average_precision_score(edge_labels, logits)
        return roc_auc, ap_score

    @staticmethod
    def eval_node_cls(nc_logits, labels):
        """ evaluate node classification results """
        if len(labels.size()) == 2:
            preds = torch.round(torch.sigmoid(nc_logits))
            tp = len(torch.nonzero(preds * labels))
            tn = len(torch.nonzero((1-preds) * (1-labels)))
            fp = len(torch.nonzero(preds * (1-labels)))
            fn = len(torch.nonzero((1-preds) * labels))
            pre, rec, f1 = 0., 0., 0.
            if tp+fp > 0:
                pre = tp / (tp + fp)
            if tp+fn > 0:
                rec = tp / (tp + fn)
            if pre+rec > 0:
                fmeasure = (2 * pre * rec) / (pre + rec)
        else:
            preds = torch.argmax(nc_logits, dim=1)
            correct = torch.sum(preds == labels)
            fmeasure = correct.item() / len(labels)
        return fmeasure

    @staticmethod
    def get_lr_schedule_by_sigmoid(n_epochs, lr, warmup):
        """ schedule the learning rate with the sigmoid function.
        The learning rate will start with near zero and end with near lr """
        factors = torch.FloatTensor(np.arange(n_epochs))
        factors = ((factors / factors[-1]) * (warmup * 2)) - warmup
        factors = torch.sigmoid(factors)
        # range the factors to [0, 1]
        factors = (factors - factors[0]) / (factors[-1] - factors[0])
        lr_schedule = factors * lr
        return lr_schedule

    @staticmethod
    def get_logger(name):
        """ create a nice logger """
        logger = logging.getLogger(name)
        # clear handlers if they were created in other runs
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        # create console handler add add to logger
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # create file handler add add to logger when name is not None
        if name is not None:
            fh = logging.FileHandler(f'GAug-{name}.log')
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
        return logger

    @staticmethod
    def col_normalization(features):
        """ column normalization for feature matrix """
        features = features.numpy()
        m = features.mean(axis=0)
        s = features.std(axis=0, ddof=0, keepdims=True) + 1e-12
        features -= m
        features /= s
        return torch.FloatTensor(features)


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

def adjacency_matrix_to_hypergraph(adj_matrix):
    hyperedge = []
    num_nodes = adj_matrix.shape[0]

    for i in range(num_nodes):
        
        hyperedge_nodes = np.where(adj_matrix[i] != 0)[0].tolist()
        hyperedge.append(hyperedge_nodes)
    num_nodes = adj_matrix.shape[0]
    hyperg = Hypergraph(num_v=num_nodes, e_list=hyperedge)


    return hyperg

class HGAug_model(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hid_channels1: int,
                 hid_channels2: int,
                 num_classes: int,
                 use_bn,
                 dropout,
                 device,
                 temperature=1,
                 alpha=1,
                 sample_type='add_sample'):
        super(HGAug_model, self).__init__()
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.sample_type=sample_type
        # edge prediction network
        self.ep_net = VHGAE_model(in_channels,
                 hid_channels1,
                 hid_channels2,
                 use_bn= False,
                 drop_rate = 0.5,
                )
        # node classification network
        # print(dropout)
        # print('*'*100)
        
        self.nc_net = HGNN_model(in_channels, hid_channels1, num_classes, use_bn, dropout)

    def sample_adj(self, adj_logits):
        """ sample an adj from the predicted edge probabilities of ep_net """
        edge_probs = adj_logits / torch.max(adj_logits)
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_add_bernoulli(self, adj_logits, alpha):
        adj_orig = self.adj_orig
        #adj_orig = torch.sparse_coo_tensor(adj_orig.nonzero(), adj_orig.data, adj_orig.shape).to_dense()
        coords = np.array(adj_orig.nonzero())
        data = np.array(adj_orig.data)
        adj_orig = torch.sparse_coo_tensor(coords, data, adj_orig.shape)
        
        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = alpha*edge_probs + (1-alpha)*adj_orig
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_add_round(self, adj_logits, alpha):
        adj_orig = self.adj_orig
        adj_orig = torch.sparse_coo_tensor(adj_orig.nonzero(), adj_orig.data, adj_orig.shape).to_dense()
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

    def sample_adj_edge(self, adj_logits, change_frac):
        adj_orig = self.adj_orig
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
        # if self.gnnlayer_type == 'gcn':
        adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
        adj.fill_diagonal_(1)
        # normalize adj with A = D^{-1/2} @ A @ D^{-1/2}
        D_norm = torch.diag(torch.pow(adj.sum(1), -0.5)).to(self.device)
        adj = D_norm @ adj @ D_norm
        # elif self.gnnlayer_type == 'gat':
        #     # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
        #     adj.fill_diagonal_(1)
        # elif self.gnnlayer_type == 'gsage':
        #     # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
        #     adj.fill_diagonal_(1)
        #     adj = F.normalize(adj, p=1, dim=1)
        return adj

    def forward(self, features, hg):
        adj_orig = adjacency_matrix(hg, s=1, weight=False)
        self.adj_orig = adj_orig
        adj_logits = self.ep_net(features, hg)
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
                adj_new = self.sample_adj_add_bernoulli(adj_logits, self.alpha)
        adj_new_normed = self.normalize_adj(adj_new)
        hg_new = adjacency_matrix_to_hypergraph(adj_new_normed)
        nc_logits = self.nc_net(features, hg_new)
        return nc_logits, adj_logits



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




def clique_negative_sampling(hyperedges, nodes_to_neighbors,list_hyperedges, node_set):
    edgeidx = np.random.choice(len(hyperedges), size=1)[0]
    neg = list_hyperedges[edgeidx]

    while neg in hyperedges:
        edgeidx = np.random.choice(len(hyperedges), size=1)[0]
        edge = list(list_hyperedges[edgeidx])
        node_to_remove = np.random.choice(len(edge), size=1)[0]
        nodes_to_keep = edge[:node_to_remove] + edge[node_to_remove+1:]
        probable_neighbors = node_set
        for node in nodes_to_keep:
            probable_neighbors = probable_neighbors.intersection(
                nodes_to_neighbors[node])
        
        if len(probable_neighbors) == 0:
            continue
        probable_neighbors = list(probable_neighbors)
        neighbor_node = np.random.choice(probable_neighbors, size=1)[0]
        
        nodes_to_keep.append(neighbor_node)
        neg = list(nodes_to_keep)
    '''
    edges = {
        frozenset([node1, node2])
        for node1 in neg for node2 in neg if node1 < node2
    }
    '''
    return neg


def csr_to_nodes_to_neighbors(csr_matrix):
    nodes_to_neighbors = {}
    for i in range(csr_matrix.shape[0]):
        neighbors = set(csr_matrix.getrow(i).nonzero()[1]) - {i}
        nodes_to_neighbors[i] = neighbors

    return nodes_to_neighbors
