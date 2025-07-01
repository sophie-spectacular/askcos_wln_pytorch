import os
import torch
import numpy as np
from rdkit import Chem

from WLN.models import WLNCandidateRanker, WLNPairwiseAtomClassifier
from graph_utils.ioutils_direct import smiles2graph_list_bin
from graph_utils.mol_graph_useScores import smiles2graph as s2g_edit
from src.gen_cand_score import gen_cand_single
from src.utils import set_map, enumerate_outcomes

class FWPredictor:
    """
    Class for making predictions with PyTorch models.
    """
    def __init__(self, model_name=None, model_dir=None, hidden_bond_classifier=300, hidden_candidate_ranker=500, depth=3,
                 cutoff=1500, output_dim=5, core_size=16, debug=False, reagents=False, device='cpu'):
        self.model_name = model_name
        self.cutoff = cutoff
        self.core_size = core_size
        self.model_dir = model_dir
        self.hidden_bond_classifier = hidden_bond_classifier
        self.hidden_cand_ranker = hidden_candidate_ranker
        self.depth = depth
        self.output_dim = output_dim
        self.debug = debug
        self.reagents = reagents
        self.device = device

    def load_models(self):
        # Hardcode a smiles to init the model params
        init_smi = '[CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14]'
        init_cand ='1.0-3.0-0.0 -5.605409145355225 1.0-2.0-0.0 -14.99064826965332 1.0-4.0-0.0 -10.05514144897461 1.0-2.0-1.0 -13.740262985229492 1.0-15.0-1.0 -9.8405122756958 1.0-3.0-2.0 -7.9041523933410645 1.0-14.0-1.0 -10.052669525146484 1.0-9.0-1.0 -12.468145370483398 1.0-4.0-2.0 -10.767592430114746 1.0-12.0-1.0 -8.33215618133545'
        cand_bonds = init_cand.split()
        cand_bonds = [cand_bonds[i].split('-') + [cand_bonds[i + 1]] for i in range(0, len(cand_bonds), 2)]
        cand_bonds = [(int(float(x)), int(float(y)), float(t), float(s)) for x, y, t, s in cand_bonds]

        reactant_smi = init_smi.split('>')[0]
        core_input = list(smiles2graph_list_bin([reactant_smi], idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1))
        diff_input, _ = s2g_edit(reactant_smi, None, cand_bonds, None, cutoff=self.cutoff, core_size=self.core_size,
                                 idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1, testing=True)
        diff_input = [np.expand_dims(i, axis=0) for i in diff_input]

        core_model_path = f'{self.model_dir}/{self.model_name}_core-weights.pt'
        self.core_model = WLNPairwiseAtomClassifier(self.hidden_bond_classifier, self.depth, output_dim=self.output_dim)
        self.core_model.eval()
        self.core_model.to(self.device)
        # Dummy forward to initialize weights
        with torch.no_grad():
            dummy = [torch.from_numpy(np.array(g)).to(self.device) for g in core_input]
            self.core_model(dummy)
        self.core_model.load_state_dict(torch.load(core_model_path, map_location=self.device))

        diff_model_path = f'{self.model_dir}/{self.model_name}_diffnet-weights.pt'
        self.diff_model = WLNCandidateRanker(self.hidden_cand_ranker, self.depth)
        self.diff_model.eval()
        self.diff_model.to(self.device)
        with torch.no_grad():
            dummy = tuple(torch.from_numpy(np.array(i)).to(self.device) for i in diff_input)
            self.diff_model(dummy)
        self.diff_model.load_state_dict(torch.load(diff_model_path, map_location=self.device))
        print('**********MODELS LOADED SUCCESSFULLY**********')

    def predict_single(self, smi):
        if '>' not in smi:
            reactant_smiles, _ = set_map(smi)
        else:
            reactant_smiles = smi.split('>')[0]
            if any(not a.HasProp('molAtomMapNumber') for a in Chem.MolFromSmiles(reactant_smiles).GetAtoms()):
                reactant_smiles, _ = set_map(reactant_smiles)

        graph_inputs = list(smiles2graph_list_bin([reactant_smiles], idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1))
        graph_inputs_torch = [torch.from_numpy(np.array(g)).to(self.device) for g in graph_inputs]
        with torch.no_grad():
            score = self.core_model(graph_inputs_torch)
        self.score = score
        cand_bonds = gen_cand_single(score[0].cpu().numpy(), smiles=reactant_smiles, reagents=self.reagents)

        graph_inputs, conf = s2g_edit(reactant_smiles, None, cand_bonds, None, cutoff=self.cutoff, core_size=self.core_size,
            idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1, testing=True)
        inputs = tuple(torch.from_numpy(np.expand_dims(np.array(g), axis=0)).to(self.device) for g in graph_inputs)

        with torch.no_grad():
            score = self.diff_model(inputs)
        outcomes = enumerate_outcomes(reactant_smiles, conf, score.cpu().numpy())

        if self.debug:
            return outcomes, smi, sorted(cand_bonds, key=lambda x: x[3], reverse=True)
        else:
            return outcomes

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.core_model.state_dict(), f'{output_dir}/{self.model_name}_core-weights.pt')
        torch.save(self.diff_model.state_dict(), f'{output_dir}/{self.model_name}_diffnet-weights.pt')

if __name__ == '__main__':
    predictor = FWPredictor(model_name='uspto_500k', model_dir='work/bin/models/')
    predictor.load_models()
    res = predictor.predict_single('CCOCCCO.c1cnccc1Cl')
    print(list(zip(['.'.join(x.get('smiles', ['R'])) for x in res], [x.get('prob', -1) for x in res])))