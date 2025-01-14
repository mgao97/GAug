import os
import sys
import time
import pickle
import warnings
import numpy as np
import networkx as nx
import scipy.sparse as sp
import dgl
from dgl import DGLGraph
import torch
from collections import defaultdict
from sklearn.preprocessing import normalize

from vgae.utils import sparse_to_tuple

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CUR_DIR)

class DataLoader():
    def __init__(self, args, data):
        self.args = args

        

        self.load_data(data)
        
        self.mask_test_edges(args.val_frac, args.test_frac, args.no_mask)
        self.normalize_adj()
        self.to_pyt_sp()
        
        
    def load_data(self,data):
        n = data['num_vertices']
        m = len(data['edge_list'])
        
        edges = []
        features = data['features']
        for i in range(m):
            features_tensor = []
            for x in data['edge_list'][i]:
                edges.append([x,i+n])
                features_tensor.append(data['features'][x])
            features_tensor = torch.stack(features_tensor)
            mean_features = torch.mean(features_tensor, dim = 0)
            features = torch.vstack((features, mean_features))
        edges = np.asarray(edges)
        row = edges.T[0]
        col = edges.T[1]
        adj_mat = sp.csr_matrix((np.ones_like(row), (row, col)), shape=(n+m, n+m))
        adj_mat = adj_mat + adj_mat.T
        features = sp.coo_matrix(features.numpy())
        self.adj_orig = adj_mat
        self.features_orig = features
        
    
    def mask_test_edges(self, val_frac, test_frac, no_mask):
        adj = self.adj_orig
        assert adj.diagonal().sum() == 0

        adj_triu = sp.triu(adj)
        edges = sparse_to_tuple(adj_triu)[0]
        edges_all = sparse_to_tuple(adj)[0]
        num_test = int(np.floor(edges.shape[0] * test_frac))
        num_val = int(np.floor(edges.shape[0] * val_frac))

        all_edge_idx = list(range(edges.shape[0]))
        np.random.shuffle(all_edge_idx)
        val_edge_idx = all_edge_idx[:num_val]
        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges[test_edge_idx]
        val_edges = edges[val_edge_idx]
        if no_mask:
            train_edges = edges
        else:
            train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)

        test_edges_false = []
        while len(test_edges_false) < len(test_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])

        val_edges_false = []
        while len(val_edges_false) < len(val_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], train_edges):
                continue
            if ismember([idx_j, idx_i], train_edges):
                continue
            if ismember([idx_i, idx_j], val_edges):
                continue
            if ismember([idx_j, idx_i], val_edges):
                continue
            if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
            val_edges_false.append([idx_i, idx_j])

        # assert ~ismember(test_edges_false, edges_all)
        # assert ~ismember(val_edges_false, edges_all)
        # assert ~ismember(val_edges, test_edges)
        # if not no_mask:
        #     assert ~ismember(val_edges, train_edges)
        #     assert ~ismember(test_edges, train_edges)

        # Re-build adj matrix
        adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        self.adj_train = adj_train + adj_train.T
        self.adj_label = adj_train + sp.eye(adj_train.shape[0])
        # NOTE: these edge lists only contain single direction of edge!
        self.val_edges = val_edges
        self.val_edges_false = np.asarray(val_edges_false)
        self.test_edges = test_edges
        self.test_edges_false = np.asarray(test_edges_false)

    def normalize_adj(self):
        adj_ = sp.coo_matrix(self.adj_train)
        adj_.setdiag(1)
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        self.adj_norm = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

    def to_pyt_sp(self):
        adj_norm_tuple = sparse_to_tuple(self.adj_norm)
        adj_label_tuple = sparse_to_tuple(self.adj_label)
        features_tuple = sparse_to_tuple(self.features_orig)
        self.adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_tuple[0].T),
                                                torch.FloatTensor(adj_norm_tuple[1]),
                                                torch.Size(adj_norm_tuple[2]))
        self.adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label_tuple[0].T),
                                                torch.FloatTensor(adj_label_tuple[1]),
                                                torch.Size(adj_label_tuple[2]))
        self.features = torch.sparse.FloatTensor(torch.LongTensor(features_tuple[0].T),
                                                torch.FloatTensor(features_tuple[1]),
                                                torch.Size(features_tuple[2]))

