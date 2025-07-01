import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from WLN.layers import WLN_Layer, WLN_Edit, Global_Attention, WL_DiffNet


class WLN_Regressor(nn.Module):
    '''
    A simple NN regressor that uses the WLN graph convolution procedure as the embedding layer
    '''
    def __init__(self, hidden_size, depth, max_nb=10):
        super(WLN_Regressor, self).__init__()
        self.WLN = WLN_Layer(hidden_size, depth, max_nb)
        self.layer1 = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        x, _ = self.WLN(inputs)
        x = torch.sum(x, dim=-2)
        x = self.layer1(x)
        return x

class WLNPairwiseAtomClassifier(nn.Module):
    '''
    PyTorch version of the Keras WLNPairwiseAtomClassifier
    '''
    def __init__(self, hidden_size, depth, output_dim=5, max_nb=10):
        super(WLNPairwiseAtomClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.WLN = WLN_Layer(hidden_size, depth, max_nb)
        self.attention = Global_Attention(hidden_size)
        self.atom_feature = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bin_feature = nn.Linear(11, hidden_size, bias=False)
        self.ctx_feature = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, output_dim)

    def forward(self, inputs):
        atom_hidden, _ = self.WLN(inputs)
        att_context, atom_pair = self.attention(atom_hidden, inputs[-1])
        att_context1 = att_context.unsqueeze(1)
        att_context2 = att_context.unsqueeze(2)
        att_pair = att_context1 + att_context2
        atom_pair = atom_pair.float()
        att_pair = att_pair.float()
        inputs[-1] = inputs[-1].float()
        pair_hidden = self.atom_feature(atom_pair) + self.bin_feature(inputs[-1]) + self.ctx_feature(att_pair)
        pair_hidden = F.relu(pair_hidden)
        pair_hidden = pair_hidden.view(pair_hidden.size(0), -1, self.hidden_size)
        score = self.score(pair_hidden)
        score = score.view(score.size(0), -1)
        return score

class WLNCandidateRanker(nn.Module):
    '''
    PyTorch version of the Keras WLNCandidateRanker
    '''
    def __init__(self, hidden_size, depth, max_nb=10):
        super(WLNCandidateRanker, self).__init__()
        self.hidden_size = hidden_size
        self.WLN = WLN_Edit(hidden_size, depth, max_nb)
        self.WL_DiffNet = WL_DiffNet(hidden_size, depth=1, max_nb=max_nb)
        self.rex_hidden = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(500, 1)

def forward(self, inputs):
    """
    inputs: tuple of tensors, see your data loader spec.
        inputs[0-4]: graph tensors
        inputs[5]: core_bias, shape [batch, n_candidates] or [n_candidates]
    """
    graph_inputs = tuple(inp for inp in inputs[:5])
    core_bias = inputs[5]

    fp_all_atoms = self.WLN(graph_inputs)
    reactant = fp_all_atoms[:, 0:1, :] 
    candidates = fp_all_atoms[:, 1:, :]
    candidates = candidates - reactant

    reaction_fp = self.WL_DiffNet(graph_inputs, candidates)
    reaction_fp = self.rex_hidden(reaction_fp) 
    scores = self.score(reaction_fp).squeeze(-1)

    if core_bias is not None:
        core_bias = core_bias.squeeze()
        if core_bias.shape == scores.shape:
            scores = scores + core_bias
        elif core_bias.shape[-1] == scores.shape[-1]:
            scores = scores + core_bias.unsqueeze(0)
        else:
            # fallback: broadcast or ignore
            scores = scores

    return scores
