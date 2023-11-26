import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torch.optim as optim
from dhg import Hypergraph
from dhg.data import Cooking200
from dhg.random import set_seed
import dhg
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from dhg.structure.hypergraphs import Hypergraph
import gc
import math
import time
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score

class HGNNConv(nn.Module):
    r"""The HGNN convolution layer proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).
    Matrix Format:

    .. math::
        \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} 
        \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X} \mathbf{\Theta} \right).

    where :math:`\mathbf{X}` is the input vertex feature matrix, :math:`\mathbf{H}` is the hypergraph incidence matrix, 
    :math:`\mathbf{W}_e` is a diagonal hyperedge weight matrix, :math:`\mathbf{D}_v` is a diagonal vertex degree matrix, 
    :math:`\mathbf{D}_e` is a diagonal hyperedge degree matrix, :math:`\mathbf{\Theta}` is the learnable parameters.

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        X = self.theta(X)
        X = hg.smoothing_with_HGNN(X)
        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X



class HGNN_model(nn.Module):
    r"""The HGNN model proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        print(drop_rate)
        print('='*100)
        self.layers.append(
            HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X

class HGNN(object):
    def __init__(self, hg, data, hid_dim, lr, seed, weight_decay, use_bn, dropout, epochs, test, evaluator, device):
        self.device = device
        
        self.hg = hg
        self.features = data['features']
        self.labels = data['labels']
        self.hid_dim = hid_dim
        self.classes = data['num_classes']
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.model = HGNN_model(self.features.shape[1], self.hid_dim, self.num_classes, use_bn, dropout)
        # move everything to device
        self.model.to(self.device)

    def fit(self, data, evaluator, test=False):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        
    
        for epoch in range(self.epochs):
            self.model.train()
            logits = self.model(features, self.hg)
            logits, lbls = logits[data['train_mask']], labels[data['train_mask']]
            loss = F.cross_entropy(logits, lbls)
            loss.backward()
            optimizer.step()
            print(f'epoch:{epoch}, time: {time.time()}s, loss: {loss.item():.4f}')

            self.model.eval()
            with torch.no_grad():
                logits_eval = self.model(features, self.hg)
                logits_eval, labels_eval = logits_eval[data['val_mask']], labels[data['val_mask']]
            
            if not test:
                res = evaluator.validate(labels_eval, logits_eval)
            else:
                res = evaluator.test(labels_eval, logits_eval)
        torch.cuda.empty_cache()
        gc.collect()

        return res


    
