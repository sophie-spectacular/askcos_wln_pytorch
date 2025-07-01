import torch
import torch.nn as nn
import torch.nn.functional as F

def ensure_float32(tensor):
    if tensor.dtype == torch.float64:
        return tensor.float()
    return tensor

class WLN_Layer(nn.Module):
    def __init__(self, hidden_size, depth, max_nb=10):
        super(WLN_Layer, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.max_nb = max_nb

        self.atom_features = nn.Linear(82, hidden_size, bias=False)
        self.nei_atom = nn.Linear(hidden_size, hidden_size, bias=False)
        self.nei_bond = nn.Linear(6, hidden_size, bias=False)
        self.self_atom = nn.Linear(hidden_size, hidden_size, bias=False)
        self.label_U2 = nn.Linear(hidden_size + 6, hidden_size)
        self.label_U1 = nn.Linear(2 * hidden_size, hidden_size)


    def forward(self, graph_inputs):
        graph_inputs = tuple(map(ensure_float32, graph_inputs))
        input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask, _ = graph_inputs
        atom_features = self.atom_features(input_atom)
        batch_size, num_atoms, _ = atom_features.shape

        for _ in range(self.depth):
            atom_graph_exp = atom_graph.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size)
            atom_graph_exp = atom_graph_exp.contiguous().view(batch_size, num_atoms * self.max_nb, self.hidden_size)
            fatom_nei = torch.gather(atom_features, 1, atom_graph_exp)
            fatom_nei = fatom_nei.view(batch_size, num_atoms, self.max_nb, self.hidden_size)

            bond_graph_exp = bond_graph.unsqueeze(-1).expand(-1, -1, -1, 6)
            bond_graph_exp = bond_graph_exp.contiguous().view(batch_size, num_atoms * self.max_nb, 6)
            fbond_nei = torch.gather(input_bond, 1, bond_graph_exp)
            fbond_nei = fbond_nei.view(batch_size, num_atoms, self.max_nb, 6)

            h_nei_atom = self.nei_atom(fatom_nei)
            h_nei_bond = self.nei_bond(fbond_nei)
            h_nei = h_nei_atom * h_nei_bond

            mask_nei = torch.arange(self.max_nb).to(num_nbs.device)
            mask_nei = (mask_nei.unsqueeze(0).unsqueeze(0) < num_nbs.unsqueeze(-1)).float()
            mask_nei = mask_nei.unsqueeze(-1)

            f_nei = torch.sum(h_nei * mask_nei, dim=2)
            f_self = self.self_atom(atom_features)
            node_mask_reshaped = node_mask.unsqueeze(-1)
            kernel = f_nei * f_self * node_mask_reshaped

            l_nei = torch.cat([fatom_nei, fbond_nei], dim=3)
            pre_label = self.label_U2(l_nei)
            nei_label = torch.sum(pre_label * mask_nei, dim=2)
            new_label = torch.cat([atom_features, nei_label], dim=2)
            atom_features = self.label_U1(new_label)

        return kernel, atom_features

import torch
import torch.nn as nn
import torch.nn.functional as F
class WLN_Edit(nn.Module):
    def __init__(self, hidden_size, depth, max_nb=10):
        super(WLN_Edit, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.max_nb = max_nb

        self.atom_features = nn.Linear(89, hidden_size, bias=False)
        self.label_U2 = nn.Linear(hidden_size + 5, hidden_size)
        self.label_U1 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, graph_inputs):
        input_atom, input_bond, atom_nei_idx, bond_nei_idx, num_nbs = graph_inputs
        B, N, K, F_bond, H = 76, 151, 10, 5, self.hidden_size

        print("input_atom:", input_atom.shape)        # [76, 151, 89]
        print("input_bond:", input_bond.shape)        # [76, 151, 10, 5]
        print("atom_nei_idx:", atom_nei_idx.shape)    # [76, 151, 10]
        print("bond_nei_idx:", bond_nei_idx.shape)    # [76, 151, 10]
        print("num_nbs:", num_nbs.shape)              # [76, 151]

        atom_features = self.atom_features(input_atom)  # [76, 151, H]
        print("atom_features:", atom_features.shape)

        for step in range(self.depth):
            print(f"\n--- Layer {step+1}/{self.depth} ---")

            atom_features_exp = atom_features.unsqueeze(2).expand(B, N, K, H)
            print("atom_features_exp:", atom_features_exp.shape)

            if atom_nei_idx.dim() == 4 and atom_nei_idx.size(-1) == 2:
                atom_nei_idx_exp = atom_nei_idx[..., 0]
            else:
                atom_nei_idx_exp = atom_nei_idx

            atom_nei_idx_exp = atom_nei_idx_exp.unsqueeze(-1).expand(B, N, K, H)
            print("atom_nei_idx_exp:", atom_nei_idx_exp.shape)

            fatom_nei = torch.gather(atom_features_exp, dim=1, index=atom_nei_idx_exp)
            print("fatom_nei:", fatom_nei.shape)

            bond_nei_idx_exp = bond_nei_idx[..., 0].unsqueeze(-1).expand(B, N, K, F_bond)

            # Gather bond features
            input_bond_exp = input_bond.unsqueeze(2).expand(B, N, K, F_bond)
            print("input_bond_exp:", input_bond_exp.shape)
            print("bond_nei_idx_exp:", bond_nei_idx_exp.shape)

            fbond_nei = torch.gather(
                input_bond_exp,  # [B, N, F_bond]
                dim=1,
                index=bond_nei_idx[..., 0].unsqueeze(-1).expand(-1, -1, -1, F_bond)
            ) 
            print("fbond_nei:", fbond_nei.shape)

            mask_nei = (torch.arange(K, device=num_nbs.device).view(1, 1, -1) < num_nbs.unsqueeze(-1)).float()
            mask_nei = mask_nei.unsqueeze(-1)  # [B, N, K, 1]
            print("mask_nei:", mask_nei.shape)

            l_nei = torch.cat([fatom_nei, fbond_nei], dim=-1)  # [B, N, K, H + F_bond]
            print("l_nei:", l_nei.shape)

            pre_label = self.label_U2(l_nei)
            print("pre_label:", pre_label.shape)

            nei_label = (pre_label * mask_nei).sum(dim=2)
            print("nei_label:", nei_label.shape)

            new_label = torch.cat([atom_features, nei_label], dim=-1)
            print("new_label:", new_label.shape)

            atom_features = self.label_U1(new_label)
            print("updated atom_features:", atom_features.shape)

        return atom_features


class WL_DiffNet(nn.Module):
    def __init__(self, hidden_size, depth, max_nb=10):
        super(WL_DiffNet, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.max_nb = max_nb

        self.label_U2 = nn.Sequential(
            nn.Linear(hidden_size + 5, hidden_size), nn.ReLU()
        )
        self.label_U1 = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size), nn.ReLU()
        )

    def forward(self, graph_inputs, atom_features):
        graph_inputs = tuple(map(ensure_float32, graph_inputs))
        input_atom, input_bond, atom_graph, bond_graph, num_nbs = graph_inputs
        batch_size, num_atoms, _ = atom_features.shape

        for i in range(self.depth):

            atom_graph_idx = atom_graph[..., 0]  # shape: [B, N, K]

            atom_graph_exp = atom_graph_idx.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size)  # [B, N, K, H]

            atom_graph_exp = atom_graph_exp.contiguous().view(batch_size, num_atoms * self.max_nb, self.hidden_size)

            fatom_nei = torch.gather(atom_features, 1, atom_graph_exp)
            fatom_nei = fatom_nei.view(batch_size, num_atoms, self.max_nb, self.hidden_size)

            bond_graph_idx = bond_graph[..., 0]
            bond_graph_exp = bond_graph_idx.unsqueeze(-1).expand(-1, -1, -1, input_bond.shape[-1])

            bond_graph_exp = bond_graph_exp.contiguous().view(batch_size, num_atoms * self.max_nb, input_bond.shape[-1])

            fbond_nei = torch.gather(input_bond, 1, bond_graph_exp)
            fbond_nei = fbond_nei.view(batch_size, num_atoms, self.max_nb, input_bond.shape[-1])

            mask_nei = torch.arange(self.max_nb).to(num_nbs.device)
            mask_nei = (mask_nei.unsqueeze(0).unsqueeze(0) < num_nbs.unsqueeze(-1)).float()
            mask_nei = mask_nei.unsqueeze(-1)

            l_nei = torch.cat([fatom_nei, fbond_nei], dim=3)

            pre_label = self.label_U2(l_nei)

            nei_label = torch.sum(pre_label * mask_nei, dim=2)

            new_label = torch.cat([atom_features, nei_label], dim=2)

            atom_features = self.label_U1(new_label)

        final_fp = torch.sum(atom_features, dim=1)
        return final_fp


class Global_Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Global_Attention, self).__init__()
        self.att_atom_feature = nn.Linear(hidden_size, hidden_size, bias=False)
        self.att_bin_feature = nn.Linear(hidden_size, hidden_size)
        self.att_score = nn.Linear(hidden_size, 1)


    def forward(self, inputs, bin_features):
        inputs = ensure_float32(inputs)
        bin_features = ensure_float32(bin_features)

        atom_hiddens1 = inputs.unsqueeze(2)
        atom_hiddens2 = inputs.unsqueeze(1)
        atom_pair = atom_hiddens1 + atom_hiddens2

        att_hidden = F.relu(self.att_atom_feature(atom_pair) + self.att_bin_feature(bin_features))
        att_score = self.att_score(att_hidden)
        att_context = att_score * atom_hiddens1
        return att_context.sum(2), atom_pair
