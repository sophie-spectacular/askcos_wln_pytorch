import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import dgl
from rdkit import Chem

class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(100, hidden_dim)  # Atom types
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, atom_types):
        return self.linear(self.embed(atom_types))

class BondEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(10, hidden_dim)  # Bond types
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, bond_types):
        return self.linear(self.embed(bond_types))

class WLN(nn.Module):
    def __init__(self, hidden_dim, num_message_passing=3):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        self.convs = nn.ModuleList([GraphConv(hidden_dim, hidden_dim) for _ in range(num_message_passing)])
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Score for a bond change
        )

    def forward(self, g, atom_types, bond_types, candidates):
        h = self.atom_encoder(atom_types)
        e = self.bond_encoder(bond_types)

        for conv in self.convs:
            h = conv(g, h)
            h = F.relu(h)

        # Get node pair embeddings for each candidate bond change
        scores = []
        for u, v in candidates:  # list of (i, j) atom indices
            pair_emb = torch.cat([h[u], h[v]], dim=-1)
            score = self.scorer(pair_emb)
            scores.append(score)

        return torch.stack(scores).squeeze()