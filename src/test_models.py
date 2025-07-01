import json
import torch
import numpy as np

from WLN.data_loading import Graph_DataLoader, Candidate_DataLoader
from WLN.metrics import wln_loss, top_10_acc, top_20_acc, top_100_acc
from WLN.models import WLNCandidateRanker, WLNPairwiseAtomClassifier
from src.gen_cand_score import gen_core_preds
from src.utils import enumerate_outcomes
from graph_utils.ioutils_direct import smiles2graph_list_bin
from graph_utils.mol_graph_useScores import smiles2graph as s2g_edit

def gen_cands_detailed_testing(core_model=None, model_name=None, model_dir='models', test=None,
                               batch_size=10, reagents=False, device='cpu'):
    assert core_model or model_name, 'Either a model or model_name (path) needs to be provided'
    assert test, 'Please provide the test set'

    # Hardcoded from rexgen_direct DO NOT CHANGE
    NK3 = 80

    test_gen = Graph_DataLoader(test, batch_size, detailed=True, shuffle=False, reagents=reagents)

    if model_name and not core_model:
        params_file = f'{model_dir}/{model_name}_core-params.txt'
        core_model_path = f'{model_dir}/{model_name}_core-weights.pt'
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
        core_model = WLNPairwiseAtomClassifier(hidden, depth, output_dim)
        # Hardcode a smiles to init the model params
        init_smi = '[CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14]'
        reactant_smi = init_smi.split('>')[0]
        core_input = list(smiles2graph_list_bin([reactant_smi], idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1))
        with torch.no_grad():
            dummy = [torch.from_numpy(np.array(g)).to(device) for g in core_input]
            core_model(dummy)
        core_model.load_state_dict(torch.load(core_model_path, map_location=device))
        core_model.eval()
        core_model.to(device)
        print('**********MODEL LOADED SUCCESSFULLY**********')

    detailed_file = f'{model_dir}/test_{model_name}.cbond_detailed.txt'
    gen_core_preds(core_model, detailed_file, test_gen, nk=NK3, device=device)
    print(f'Detailed output written to file {detailed_file}')

def test_wln_diffnet(test=None, batch_size=1, model_name='wln_diffnet', model_dir='models', device='cpu'):
    """
    Tests the candidate ranker.
    """
    init_smi = '[CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14]'
    init_cand ='1.0-3.0-0.0 -5.605409145355225 1.0-2.0-0.0 -14.99064826965332 1.0-4.0-0.0 -10.05514144897461 1.0-2.0-1.0 -13.740262985229492 1.0-15.0-1.0 -9.8405122756958 1.0-3.0-2.0 -7.9041523933410645 1.0-14.0-1.0 -10.052669525146484 1.0-9.0-1.0 -12.468145370483398 1.0-4.0-2.0 -10.767592430114746 1.0-12.0-1.0 -8.33215618133545'
    reactant_smi = init_smi.split('>')[0]
    cand_bonds = init_cand.split()
    cand_bonds = [cand_bonds[i].split('-') + [cand_bonds[i + 1]] for i in range(0, len(cand_bonds), 2)]
    cand_bonds = [(int(float(x)), int(float(y)), float(t), float(s)) for x, y, t, s in cand_bonds]

    diff_input, _ = s2g_edit(reactant_smi, None, cand_bonds, None, cutoff=150, core_size=16,
                             idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1, testing=True)
    diff_input = [np.expand_dims(i, axis=0) for i in diff_input]

    params_file = f'{model_dir}/{model_name}_diffnet-params.txt'
    diff_model_path = f'{model_dir}/{model_name}_diffnet-weights.pt'
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

    diff_model = WLNCandidateRanker(hidden, depth)
    with torch.no_grad():
        dummy = tuple(torch.from_numpy(np.array(i)).to(device) for i in diff_input)
        diff_model(dummy)
    diff_model.load_state_dict(torch.load(diff_model_path, map_location=device))
    diff_model.eval()
    diff_model.to(device)
    print('**********MODEL LOADED SUCCESSFULLY**********')

    assert test, 'Please specify the test set'

    test_detailed = f'{model_dir}/test_{model_name}.cbond_detailed.txt'
    test_gen = Candidate_DataLoader(test_detailed, batch_size, cutoff=1500, core_size=16, testing=True)

    assert len(test_gen) > 0, f'Test set has {len(test_gen)} examples, has to be greater than 0'
    assert model_name and model_dir, 'Model name and directory must be provided!'

    pred_path = f'{model_dir}/test_{model_name}'

    with open(pred_path + '.predictions.txt', 'w') as fpred:
        for batch in test_gen:
            inputs, conf, rsmi = batch
            with torch.no_grad():
                score = diff_model(inputs)
            outcomes = enumerate_outcomes(rsmi, conf, score.cpu().numpy())
            for outcome in outcomes:
                for x, y, t in outcome['predicted_edits']:
                    fpred.write("{}-{}-{} ".format(x, y, t))
                fpred.write(' | ')
            fpred.write(' \n')

if __name__ == '__main__':
    gen_cands_detailed_testing(model_name='pistachio_no_reagents', model_dir='/data/tstuyver/wlnfw_pistachio2/pistachio_no_reagents',
                              test='data/pistachio_mini_test.txt.proc', batch_size=10, reagents=False)
    test_wln_diffnet(batch_size=1, model_name='pistachio_no_reagents', model_dir='/data/tstuyver/wlnfw_pistachio2/pistachio_no_reagents',
                     test='data/pistachio_mini_test.txt.proc')