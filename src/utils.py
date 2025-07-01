import torch
import numpy as np
from rdkit import Chem
from src.eval_by_smiles import edit_mol_smiles
from graph_utils.ioutils_direct import bo_to_index

def set_map(smi, sanitize=True):
    if sanitize:
        m = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smi)))
    else:
        m = Chem.MolFromSmiles(smi)
    if any(not a.HasProp('molAtomMapNumber') for a in m.GetAtoms()):
        mapnum = 1
        for a in m.GetAtoms():
            a.SetIntProp('molAtomMapNumber', mapnum)
            mapnum += 1
    return Chem.MolToSmiles(m), m.GetNumAtoms()


def topk_numpy(scores, bond_labels, nk=80, invalid_bond=-1):
    """Replicates the TensorFlow `topk_numpy` with Torch."""
    bmask = (bond_labels == invalid_bond).float() * 10000
    scores = scores - bmask
    topk_vals, topk_indices = torch.topk(scores, k=min(nk, scores.shape[0]))

    return topk_indices.detach().cpu().numpy()


def enumerate_outcomes(rsmi, conf, scores, top_n=100):
    """
    PyTorch-native enumeration of outcomes.
    """
    scores = scores.flatten()
    k = min(top_n, scores.shape[0])

    top_vals, top_indices = torch.topk(scores, k=k)
    top_indices = top_indices.detach().cpu().numpy()
    top_vals = top_vals.detach()

    idxfunc = lambda x: x.GetIntProp('molAtomMapNumber') - 1

    rmol = Chem.MolFromSmiles(rsmi)
    rbonds = {}
    for bond in rmol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom()) 
        a2 = idxfunc(bond.GetEndAtom()) 
        t = bond.GetBondTypeAsDouble()
        a1, a2 = min(a1, a2), max(a1, a2)
        rbonds[(a1, a2)] = t

    cand_smiles = []
    cand_scores = []
    predicted_edits = []

    for idx in top_indices:
        cbonds = []
        for x, y, t, v in conf[idx]:
            if ((x, y) not in rbonds and t > 0) or ((x, y) in rbonds and rbonds[(x, y)] != t):
                cbonds.append((x, y, bo_to_index[t]))

        pred_smiles = edit_mol_smiles(rmol, cbonds)
        if pred_smiles in cand_smiles:
            continue

        cand_smiles.append(pred_smiles)
        cand_scores.append(scores[idx].item())  # detach to a Python float
        predicted_edits.append(cbonds)

    cand_probs = torch.softmax(torch.tensor(cand_scores), dim=0)

    outcomes = []
    for i in range(min(len(cand_smiles), top_n)):
        outcomes.append({ 
            'rank': i + 1,
            'smiles': cand_smiles[i],
            'predicted_edits': predicted_edits[i],
            'score': cand_scores[i],
            'prob': cand_probs[i].item()
        })
    return outcomes
