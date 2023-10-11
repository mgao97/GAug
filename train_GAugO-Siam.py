import os
import json
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
from models.GAug-Siam import SiameseNet
import torch

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

    # params_all = json.load(open('best_parameters.json', 'r'))
    # params = params_all['GAugO'][args.dataset][args.gnn]

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
    for _ in range(30):
        model = SiameseNet(
                dim_feats, 
                 dim_h,
                 dim_z,
                 n_layers, 
                 activation, 
                 dropout, 
                 gnnlayer_type='gcn',
                 dim_h2,
                 device,
                 gnnlayer_type,
                 temperature=1,
                 gae=False)

                 
        acc = model.fit(pretrain_ep=params['pretrain_ep'], pretrain_nc=params['pretrain_nc'])
        accs.append(acc)
    print(f'Micro F1: {np.mean(accs):.6f}, std: {np.std(accs):.6f}')
