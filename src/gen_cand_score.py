import json
import math
import torch
import numpy as np

from WLN.data_loading import Graph_DataLoader
from WLN.metrics import wln_loss, top_10_acc, top_20_acc, top_100_acc
from WLN.models import WLNPairwiseAtomClassifier
from graph_utils.ioutils_direct import INVALID_BOND, bindex_to_o, nbos, reactant_tracking

'''
 Run test set then output candidates.
'''

def gen_core_preds(model, filename, data_gen, nk=80, device='cpu'):
    """
    Generates candidates for train/valid/test (ie batch)
    """
    print(f'Generating candidates for {filename}')

    model.eval()
    with open(filename, 'w') as f:
        for it, batch in enumerate(data_gen):
            graph, bond_labels, sp_labels, all_ratoms, all_rbonds, rxn_smiles, labels = batch
            # Move tensors to device
            if isinstance(graph, list):
                graph = [g.to(device) for g in graph]
            else:
                graph = graph.to(device)
            bond_labels = bond_labels.to(device)
            with torch.no_grad():
                scores = model(graph)
            # Mask invalid bonds
            bmask = (bond_labels == INVALID_BOND).float() * 10000
            masked_scores = scores - bmask
            topk_scores, topk = torch.topk(masked_scores, k=nk, dim=-1)
            topk = topk.cpu().numpy()
            topk_scores = topk_scores.cpu().numpy()
            cur_dim = bond_labels.shape[1]
            cur_dim = int(math.sqrt(cur_dim / nbos))

            for i in range(len(labels)):
                ratoms = all_ratoms[i]
                rbonds = all_rbonds[i]
                f.write(f'{rxn_smiles[i]} {labels[i]} ')
                written = 0
                for j in range(nk):
                    k = topk[i, j]
                    bindex = k % nbos
                    y = ((k - bindex) / nbos) % cur_dim + 1
                    x = (k - bindex - (y - 1) * nbos) / cur_dim / nbos + 1
                    bo = bindex_to_o[bindex]
                    # Only allow atoms from reacting molecules to be part of the prediction,
                    if x < y and x in ratoms and y in ratoms and (x, y, bo) not in rbonds:
                        f.write(f'{x}-{y}-{bo:.1f} ')
                        f.write(f'{topk_scores[i, j]} ')
                        written += 1
                f.write('\n')
    return True

def gen_cand_single(scores, nk=80, smiles=None, reagents=False):
    """
    Generates candidates for inference (ie single molecule)
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    topk_indices = np.argpartition(-scores, nk)[:nk]
    topk_scores = scores[topk_indices]
    sorted_idx = np.argsort(-topk_scores)
    topk_indices = topk_indices[sorted_idx]
    topk_scores = topk_scores[sorted_idx]
    cur_dim = int(math.sqrt(len(scores) / nbos))

    cand_bonds = []

    if smiles:
        if '>' in smiles:
            if not reagents:
                ratoms, rbonds = reactant_tracking([smiles], hard=False)
        else:
            ratoms = None
    else:
        ratoms = None

    for j in range(nk):
        k = topk_indices[j]
        bindex = k % nbos
        y = ((k - bindex) / nbos) % cur_dim + 1
        x = (k - bindex - (y - 1) * nbos) / cur_dim / nbos + 1
        bo = bindex_to_o[bindex]
        if ratoms:
            if x < y and x in ratoms[0] and y in ratoms[0] and (x, y, bo) not in rbonds[0]:
                cand_bonds.append((int(x) - 1, int(y) - 1, bo, float(topk_scores[j])))
        else:
            if x < y:
                cand_bonds.append((int(x) - 1, int(y) - 1, bo, float(topk_scores[j])))

    return cand_bonds

def gen_cands_detailed(model=None, model_name=None, model_dir='models', train=None, valid=None, test=None,
                       batch_size=10, cutoff=0.05, use_multiprocessing=True, workers=1, reagents=False, device='cpu'):

    assert model or model_name, 'Either a model or model_name (path) needs to be provided'
    assert train and valid and test, 'Please provide the training, validation, and test sets'

    # Hardcoded from rexgen_direct DO NOT CHANGE
    NK3 = 80
    NK2 = 40
    NK1 = 20
    NK0 = 16
    NK = 12

    test_gen = Graph_DataLoader(test, batch_size)

    if model_name and not model:
        params_file = f'{model_dir}/{model_name}_core-params.txt'
        core_model = f'{model_dir}/{model_name}_core-weights.pt'
        try:
            with open(params_file, 'r') as f:
                model_params = json.loads(f.read())
        except Exception:
            print('!' * 100)
            print('No Params file, will use default params for loading model. Warning: this will not work if user has changed default training parameters')
            print('!' * 100)
            model_params = {}

        hidden = model_params.get('hidden', 300)
        depth = model_params.get('depth', 3)
        output_dim = model_params.get('output_dim', 5)
        model = WLNPairwiseAtomClassifier(hidden, depth, output_dim)
        model.load_state_dict(torch.load(core_model, map_location=device))
        model.to(device)

    print('~' * 100)
    print('Evaluating model performance')

    # Evaluate model on test set
    model.eval()
    total = 0
    correct_10 = 0
    correct_20 = 0
    correct_100 = 0
    with torch.no_grad():
        for batch in test_gen:
            graph, bond_labels = batch[:2]
            if isinstance(graph, list):
                graph = [g.to(device) for g in graph]
            else:
                graph = graph.to(device)
            bond_labels = bond_labels.to(device)
            scores = model(graph)
            correct_10 += top_10_acc(bond_labels, scores).item()
            correct_20 += top_20_acc(bond_labels, scores).item()
            correct_100 += top_100_acc(bond_labels, scores).item()
            total += 1

    performance = {
        'top_10_acc': correct_10 / total,
        'top_20_acc': correct_20 / total,
        'top_100_acc': correct_100 / total
    }

    print('Performance for model on test set:')
    print(performance)
    print('~' * 100)

    assert performance.get('top_100_acc', 0.0) >= cutoff, \
        f'The top 100 accuracy for the supplied test set is below the threshold that is desired for continuing the training process \nHere are the current performance metrics {performance}'

    train_gen = Graph_DataLoader(train, batch_size, detailed=True, reagents=reagents)
    val_gen = Graph_DataLoader(valid, batch_size, detailed=True, reagents=reagents)

    detailed_file = f'{model_dir}/train_{model_name}.cbond_detailed.txt'
    train_preds = gen_core_preds(model, detailed_file, train_gen, nk=NK3, device=device)

    detailed_file1 = f'{model_dir}/valid_{model_name}.cbond_detailed.txt'
    val_preds = gen_core_preds(model, detailed_file1, val_gen, nk=NK3, device=device)

    assert train_preds and val_preds, 'Predictions for either training set or validation set failed'

    print(f'Detailed output written to file {detailed_file} and {detailed_file1}')
    return model

if __name__ == '__main__':
    gen_cands_detailed(model_name='uspto_500k', train='/work/data/train_trunc.txt.proc', valid='/work/data/valid_trunc.txt.proc',
                       test='/work/data/test_trunc.txt.proc', cutoff=-1, workers=6)