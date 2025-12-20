"""
Pattern-aware extensions to GRAPE for incomplete tabular data.

References
----------
.. [1] You, J., Ma, X., Ding, D.Y., Kochenderfer, M., and Leskovec, J. 2020.
       "Handling Missing Data with Graph Representation Learning." NeurIPS.
.. [2] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., and
       Bengio, Y. 2018. "Graph Attention Networks." ICLR.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class BipartiteMessagePassing(MessagePassing):
    """Message passing on bipartite observation-feature graphs.
    
    This is the core GRAPE mechanism from [1].
    """

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='mean')
        self.message_nn = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.update_nn = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU())

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr), edge_attr

    def message(self, x_j, edge_attr):
        return self.message_nn(torch.cat([x_j, edge_attr], dim=1))

    def update(self, aggr_out, x):
        return self.update_nn(torch.cat([x, aggr_out], dim=1))


class GRAPEBase(nn.Module):
    """
    GRAPE model for incomplete tabular data.

    Constructs bipartite graphs with edges only for observed values.
    No pattern awareness - two samples with identical observed values
    produce identical embeddings regardless of missingness pattern.

    Parameters
    ----------
    num_features : int
        Number of input features.
    hidden_dim : int
        Hidden layer dimension.
    num_classes : int
        Number of output classes.
    num_layers : int
        Number of message passing layers.
    dropout : float
        Dropout probability.
    edge_dropout : float
        Edge dropout during training, see [1] for rationale.

    References
    ----------
    .. [1] You et al. 2020, NeurIPS.
    """

    def __init__(self, num_features, hidden_dim, num_classes, num_layers=3,
                 dropout=0.2, edge_dropout=0.3):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.edge_dropout = edge_dropout

        self.node_proj = nn.Linear(num_features, hidden_dim)
        self.edge_proj = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList([
            BipartiteMessagePassing(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes))
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch)

        if self.training and edge_index.size(1) > 0:
            mask = torch.rand(edge_index.size(1), device=edge_index.device)
            edge_index = edge_index[:, mask > self.edge_dropout]
            edge_attr = edge_attr[mask > self.edge_dropout]

        h = self.node_proj(x)
        e = self.edge_proj(edge_attr) if edge_attr.size(0) > 0 else edge_attr

        for layer in self.layers:
            h, e = layer(h, edge_index, e)
            h = self.dropout(h)

        # GRAPE: extract observation node (idx 0) from each graph
        # this node aggregates info from all observed features
        obs_h = self._get_obs_embeddings(h, batch)
        if obs_h is None:
            return torch.zeros((0, self.num_classes), device=h.device)
        return self.classifier(obs_h)

    def _get_obs_embeddings(self, h, batch):
        num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 0
        if num_graphs == 0:
            return None
        emb = []
        for i in range(num_graphs):
            nodes = torch.where(batch == i)[0]
            emb.append(h[nodes[0]])
        return torch.stack(emb)


class GRAPEPatternAware(nn.Module):
    """
    GRAPE with missingness pattern encoding.

    Extends GRAPEBase by incorporating pattern information via
    concatenation with observation embedding before classification.

    Parameters
    ----------
    num_features : int
    hidden_dim : int
    num_classes : int
    pattern_encoder : str
        One of {'learned', 'random', 'statistical', 'hierarchical'}.
        - learned: trainable MLP
        - random: frozen random projection (Johnson-Lindenstrauss)
        - statistical: handcrafted features (miss rate, entropy, runs)
        - hierarchical: bottleneck MLP
    num_layers : int
    dropout : float
    edge_dropout : float

    Notes
    -----
    Random embeddings perform comparably to learned (see experiments),
    suggesting pattern distinction matters more than optimization.
    """

    def __init__(self, num_features, hidden_dim, num_classes,
                 pattern_encoder='learned', num_layers=3,
                 dropout=0.2, edge_dropout=0.3):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.edge_dropout = edge_dropout
        self.pattern_encoder_type = pattern_encoder

        self.node_proj = nn.Linear(num_features, hidden_dim)
        self.edge_proj = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList([
            BipartiteMessagePassing(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)])

        self._init_pattern_encoder(pattern_encoder, num_features,
                                   hidden_dim, dropout)

        # classifier input: GRAPE embedding + pattern embedding (our extension)
        clf_dim = hidden_dim + (hidden_dim // 2 if pattern_encoder == 'hierarchical'
                                else hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(clf_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

    def _init_pattern_encoder(self, enc_type, n_feat, h_dim, dropout):
        if enc_type == 'random':
            self.pattern_enc = nn.Sequential(
                nn.Linear(n_feat, h_dim), nn.ReLU(),
                nn.Linear(h_dim, h_dim))
            for p in self.pattern_enc.parameters():
                p.requires_grad = False
        elif enc_type == 'learned':
            self.pattern_enc = nn.Sequential(
                nn.Linear(n_feat, h_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(h_dim, h_dim))
        elif enc_type == 'statistical':
            self.pattern_enc = nn.Sequential(
                nn.Linear(4, h_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(h_dim, h_dim))
        elif enc_type == 'hierarchical':
            self.pattern_enc = nn.Sequential(
                nn.Linear(n_feat, h_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(h_dim, h_dim // 2))
        else:
            raise ValueError(f'unknown pattern encoder: {enc_type}')

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch)

        # edge dropout from GRAPE paper
        if self.training and edge_index.size(1) > 0:
            mask = torch.rand(edge_index.size(1), device=edge_index.device)
            edge_index = edge_index[:, mask > self.edge_dropout]
            edge_attr = edge_attr[mask > self.edge_dropout]

        # standard GRAPE message passing
        h = self.node_proj(x)
        e = self.edge_proj(edge_attr) if edge_attr.size(0) > 0 else edge_attr

        for layer in self.layers:
            h, e = layer(h, edge_index, e)
            h = self.dropout(h)

        obs_h = self._get_obs_embeddings(h, batch)
        if obs_h is None:
            return torch.zeros((0, self.num_classes), device=h.device)

        # our extension: encode missingness pattern and concatenate
        patterns = self._get_batch_patterns(data).to(x.device)
        if self.pattern_encoder_type == 'statistical':
            patterns = _compute_statistical_features(patterns)
        pat_h = self.pattern_enc(patterns)

        return self.classifier(torch.cat([obs_h, pat_h], dim=1))

    def _get_obs_embeddings(self, h, batch):
        num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 0
        if num_graphs == 0:
            return None
        emb = []
        for i in range(num_graphs):
            nodes = torch.where(batch == i)[0]
            emb.append(h[nodes[0]])
        return torch.stack(emb)

    def _get_batch_patterns(self, data):
        if not hasattr(data, 'mask') or data.mask is None:
            ng = data.num_graphs if hasattr(data, 'num_graphs') else 1
            return torch.zeros(ng, self.num_features, device=data.x.device)

        ng = data.num_graphs if hasattr(data, 'num_graphs') else (
            data.batch.max().item() + 1)
        nf = self.num_features
        patterns = []
        for i in range(ng):
            s, e = i * nf, (i + 1) * nf
            if s < len(data.mask) and e <= len(data.mask):
                patterns.append(data.mask[s:e])
            else:
                patterns.append(torch.zeros(nf, device=data.mask.device))
        return torch.stack(patterns)


def _compute_statistical_features(mask):
    """
    Compute summary statistics from missingness patterns.
    
    One of our pattern encoding strategies (Table III in paper).
    Returns 4 features per sample: missing rate, entropy,
    max consecutive run (normalized), num blocks (normalized).
    """
    B, D = mask.shape
    feats = []
    for i in range(B):
        m = mask[i].cpu().numpy()
        miss_rate = m.mean()

        # entropy
        if 0 < miss_rate < 1:
            p0, p1 = 1 - miss_rate, miss_rate
            ent = -(p0 * np.log2(p0 + 1e-10) + p1 * np.log2(p1 + 1e-10))
        else:
            ent = 0.0

        # consecutive runs
        runs = []
        run = 0
        for v in m:
            if v == 1:
                run += 1
            else:
                if run > 0:
                    runs.append(run)
                run = 0
        if run > 0:
            runs.append(run)
        max_run = (max(runs) / D) if runs else 0.0
        n_blocks = len(runs) / D

        feats.append([miss_rate, ent, max_run, n_blocks])
    return torch.tensor(feats, dtype=torch.float32, device=mask.device)
