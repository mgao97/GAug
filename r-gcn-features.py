import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from dgl import function as fn
import dgl.init
from dgl import DGLGraph
from dgl.contrib.data import load_data
import numpy as np
import networkx as nx
from scipy import sparse

class RGCN(nn.Module):
    """
    Relational Graph Convolutional Network for entity classification.
    
    Parameters
    ----------
    graph: dgl.DGLGraph
        The graph on which the model is applied.
    features: torch.FloatTensor
        Feature matrix of size n_nodes * n_in_feats.
    n_hidden_feats: int
        The number of features for the input and hidden layers.
    n_hidden_layers: int
        The number of hidden layers.
    activation: torch.nn.functional
        The activation function used by the input and hidden layers.
    dropout: float
        The dropout rate.
    n_rels: int
        The number of relations in the graph.
    n_bases: int
        The number of bases used by the model.
    self_loop: boolean
        Use self-loop in the model
    References
    ----------
    M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, and M. Welling,
    “Modeling Relational Data with Graph Convolutional Networks,” arXiv:1703.06103 [cs, stat], Mar. 2017.
    """

    def __init__(
        self,
        graph,
        features,
        n_hidden_feats,
        n_hidden_layers,
        n_classes,
        activation,
        dropout,
        n_rels,
        n_bases,
        self_loop
    ):
        super().__init__()
        self.features = features
        n_in_feats = features.size(1)

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            RGCNLayer(
                graph=graph,
                n_in_feats=n_in_feats,
                n_out_feats=n_hidden_feats,
                activation=activation,
                dropout=0,
                bias=True,
                n_rels=n_rels,
                n_bases=n_bases,
                self_loop=self_loop
            )
        )

        # Hidden layers
        for _ in range(n_hidden_layers):
            self.layers.append(
                RGCNLayer(
                    graph=graph,
                    n_in_feats=n_hidden_feats,
                    n_out_feats=n_hidden_feats,
                    activation=activation,
                    dropout=dropout,
                    bias=True,
                    n_rels=n_rels,
                    n_bases=n_bases,
                    self_loop=self_loop
                )
            )

        # Output layer
        self.layers.append(
            RGCNLayer(
                graph=graph,
                n_in_feats=n_hidden_feats,
                n_out_feats=n_classes,
                activation=None,
                dropout=dropout,
                bias=True,
                n_rels=n_rels,
                n_bases=n_bases,
                self_loop=self_loop
            )
        )

    def forward(self, x):
        """
        Defines how the model is run, from input to output.
        
        Parameters
        ----------
        x: torch.FloatTensor
            (Input) feature matrix of size n_nodes * n_in_feats.
        
        Return
        ------
        h: torch.FloatTensor
            Output matrix of size n_nodes * n_classes.
        """
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    def fit(self, train_labels, train_mask, epochs, lr, weight_decay):
        """
        Trains the model.
        
        Parameters
        ----------
        train_labels: torch.LongTensor
            Tensor of target data of size n_train_nodes.
        train_mask: torch.ByteTensor
            Boolean mask of size n_nodes indicating the nodes used in training.
        epochs: int
            Number of epochs.
        lr: float
            Learning rate.
        weight_decay: float
            Weight decay (L2 penalty).
        """
        loss_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        self.train()
        for epoch in range(1, epochs + 1):
            # Forward
            optimizer.zero_grad()
            logits = self(self.features)
            loss = loss_criterion(logits[train_mask], train_labels)

            # Backward
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}/{epochs} | Loss {loss.item():.5f}")

    def evaluate(self, test_labels, test_mask):
        """
        Evaluates the accuracy of the model against a set of test nodes.
        
        Parameters
        ----------
        test_labels: torch.LongTensor
            Tensor of target data of size n_test_nodes.
        test_mask: torch.ByteTensor
            Boolean mask of size n_nodes indicating the nodes used in testing.
        Returns
        -------
        accuracy: float
            The accuracy of the model on the set of test nodes.
        """
        self.eval()
        with torch.no_grad():
            logits = self(self.features)
            logits = logits[test_mask]
            total = test_labels.size(0)
            predicted = torch.argmax(logits, dim=1)
            correct = (predicted == test_labels).sum().item()
            return correct / total


class RGCNLayer(nn.Module):
    def __init__(
        self,
        graph,
        n_in_feats,
        n_out_feats,
        activation,
        dropout,
        bias,
        n_rels,
        n_bases,
        self_loop
    ):
        super().__init__()
        self.graph = graph

        self.n_in_feats = n_in_feats
        self.n_out_feats = n_out_feats
        self.activation = activation
        self.self_loop = self_loop

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_out_feats))
        else:
            self.bias = None

        self.n_rels = n_rels
        self.n_bases = n_bases

        if self.n_bases <= 0 or self.n_bases > self.n_rels:
            self.n_bases = self.n_rels

        self.loop_weight = nn.Parameter(torch.Tensor(n_in_feats, n_out_feats))

        # Add basis weights
        self.weight = nn.Parameter(
            torch.Tensor(self.n_bases, self.n_in_feats, self.n_out_feats)
        )

        if self.n_bases < self.n_rels:
            # Linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.n_rels, self.n_bases))

        self.reset_parameters()

    def reset_parameters(self):
        if self.self_loop:
            nn.init.xavier_uniform_(self.loop_weight)

        nn.init.xavier_uniform_(self.weight)

        if self.n_bases < self.n_rels:
            nn.init.xavier_uniform_(self.w_comp)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def propagate(self, h):
        if self.n_bases < self.n_rels:
            # Generate all weights from bases
            weight = self.weight.view(self.n_bases, self.n_in_feats * self.n_out_feats)
            weight = torch.matmul(self.w_comp, weight).view(
                self.n_rels, self.n_in_feats, self.n_out_feats
            )
        else:
            weight = self.weight

        def msg_func(edges):
            w = weight.index_select(dim=0, index=edges.data["type"])
            msg = torch.bmm(edges.src["h"].unsqueeze(1), w).squeeze()
            msg = msg * edges.data["norm"]
            return {"msg": msg}

        self.graph.update_all(msg_func, fn.sum(msg="msg", out="h"))

    def forward(self, h):
        if self.self_loop:
            loop_message = torch.mm(h, self.loop_weight)

            if self.dropout:
                loop_message = self.dropout(loop_message)

        self.graph.ndata["h"] = h

        # Send messages through all edges and update all nodes
        self.propagate(h)

        h = self.graph.ndata.pop("h")

        if self.self_loop:
            h = h + loop_message

        if self.bias is not None:
            h = h + self.bias

        if self.activation:
            h = self.activation(h)

        return h


def main():
    # Create dataset from Zachary's Karate Club network
    g = nx.karate_club_graph()
    g = g.to_directed()  # to directed graph

    n_edges = g.number_of_edges()
    n_nodes = g.number_of_nodes()

    labels = [0 if data["club"] == "Mr. Hi" else 1 for _, data in g.nodes(data=True)]

    edge_list = nx.to_edgelist(g)
    src = [edge[0] for edge in edge_list]
    dst = [edge[1] for edge in edge_list]
    
    weights = np.ones(n_edges)
    adj_matrix = sparse.coo_matrix((weights, (src, dst)), shape=(n_nodes, n_nodes)).toarray()
    
    # Normalize adjacency matrix
    degs = np.sum(adj_matrix, axis=1)
    degs[degs == 0] = 1
    adj_matrix = adj_matrix / degs[:, None]

    # Get edges and normalization weights
    sp_matrix = sparse.coo_matrix(adj_matrix)
    src = sp_matrix.row
    dst = sp_matrix.col
    norm = sp_matrix.data

    labels = torch.LongTensor(labels)
    src = torch.LongTensor(src)
    dst = torch.LongTensor(dst)
    weights = torch.FloatTensor(weights)
    norm = torch.FloatTensor(norm).unsqueeze(1)

    train_mask = torch.zeros(n_nodes, dtype=torch.uint8)
    train_mask[0] = train_mask[33] = 1

    # Create graph
    graph = dgl.DGLGraph()
    graph.add_nodes(num=n_nodes)

    # Add edges
    graph.add_edges(src, dst, {"norm": norm, "type": torch.zeros(n_edges, dtype=torch.long)})
    graph.add_edges(src, dst, {"norm": norm, "type": torch.ones(n_edges, dtype=torch.long)})

    graph.set_n_initializer(dgl.init.zero_initializer)

    model = RGCN(
        graph=graph,
        features=torch.eye(n_nodes, dtype=torch.float),
        n_hidden_feats=10,
        n_hidden_layers=0,
        n_classes=2,
        activation=F.relu,
        dropout=0,
        n_rels=2,
        n_bases=-1,
        self_loop=True,
    )

    model.fit(
        train_labels=labels[train_mask],
        train_mask=train_mask,
        epochs=200,
        lr=0.01,
        weight_decay=0
    )

    accuracy = model.evaluate(test_labels=labels[~train_mask], test_mask=~train_mask)

    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
