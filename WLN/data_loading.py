import itertools
from random import shuffle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset

from graph_utils.ioutils_direct import get_bond_label, smiles2graph_list_bin, reactant_tracking
from graph_utils.mol_graph_useScores import smiles2graph as s2g_edit

import time

def time_it_and_log(filename="timing_log.txt"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            with open(filename, "a") as f:
                f.write(f"{func.__name__} took {end - start:.6f} seconds\n")
            return result
        return wrapper
    return decorator

def convert_data(filename, reactants_only=True):
    df = pd.read_csv(filename, sep=' ', names=['smiles', 'bond_edits'])
    df_edits = df['bond_edits']
    df_smiles = df['smiles']
    if reactants_only:
        return df_smiles.str.split('>>', expand=True)[0], df_edits
    else:
        return df['smiles'], df_edits

def convert_detail_data(filename, filename_detail):
    smiles, edits = convert_data(filename, reactants_only=False)
    smiles = smiles.str.split('>>', expand=True)
    r, p = smiles[0], smiles[1]
    with open(filename_detail, 'r') as f:
        cand = f.read().splitlines()
    data_dict = {x.strip("\r\n ").split()[0]:x.strip("\r\n ").split()[1:] for x in cand if len(x)>0}

    fail = 0
    train = []
    for i,j in enumerate(r):
        try:
            train.append([j + '>>' + p[i], data_dict[j], edits[i]])
        except KeyError:
            fail += 1

    if fail > 10:
        print('!' * 100)
        print('!' * 100)
        print(f'WARNING: CONVERTING DETAILED DATA HAD {fail} FAILURES OUT OF {len(smiles)} EXAMPLES')
        print('!' * 100)
        print('!' * 100)

    return zip(*train)

class FileLoader(IterableDataset):
    def __init__(self, filename, batch_size, chunk_size, shuffle_data):
        assert chunk_size % batch_size == 0, 'chunk_size should be integer multiple of batch_size'
        self.filename = filename
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle_data = shuffle_data

        with open(self.filename, 'r') as f:
            self.dataset_size = sum(1 for _ in f)
            self.chunk_indices = list(range(int(np.ceil(self.dataset_size / self.chunk_size))))

        self.on_epoch_end()
        self._current_chunk_index = 0
        self._current_chunk = None

    def __len__(self):
        return int(np.ceil(self.dataset_size / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle_data:
            shuffle(self.chunk_indices)

    def get_chunk(self, batch_index):
        chunk_index = batch_index * self.batch_size // self.chunk_size
        if self.dataset_size % self.chunk_size != 0:
            index_truncated_chunk = self.chunk_indices.index(len(self.chunk_indices) - 1)
            if chunk_index >= index_truncated_chunk:
                chunk_index = ((batch_index * self.batch_size - self.dataset_size % self.chunk_size) // self.chunk_size) + 1

        true_chunk_index = self.chunk_indices[chunk_index]

        if true_chunk_index == self._current_chunk_index and self._current_chunk is not None:
            return self._current_chunk
        else:
            start = true_chunk_index * self.chunk_size
            end = (true_chunk_index + 1) * self.chunk_size
            if end > self.dataset_size:
                end = self.dataset_size
            with open(self.filename, 'r') as f:
                chunk = list(itertools.islice(f, start, end))
            if self.shuffle_data:
                shuffle(chunk)
            self._current_chunk_index = true_chunk_index
            self._current_chunk = chunk
            return self._current_chunk

    def __iter__(self):
        self.on_epoch_end()
        num_batches = len(self)
        for batch_idx in range(num_batches):
            chunk = self.get_chunk(batch_idx)
            local_index = batch_idx % (self.chunk_size // self.batch_size)
            start = local_index * self.batch_size
            end = (local_index + 1) * self.batch_size
            batch = chunk[start:end]
            yield self.data_generation(batch)

            

class Graph_DataLoader(FileLoader):
    def __init__(self, filename, batch_size=10, chunk_size=1000, shuffle=True, detailed=False, reagents=False):
        super().__init__(filename, batch_size, chunk_size, shuffle)
        self.detailed = detailed
        self.reagents = reagents

    @time_it_and_log()
    def data_generation(self, batch):
        smiles_tmp, labels_tmp = zip(*(x.strip().split() for x in batch))
        r_tmp = [x.split('>')[0] for x in smiles_tmp]
        graph_inputs = list(smiles2graph_list_bin(r_tmp, idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1))
        assert graph_inputs[0].max() < graph_inputs[0].shape[1]
        print("max atom idx in atom_graph :", graph_inputs[0].max()) 
        print("number of atoms :", graph_inputs[0].shape[1]) 
        max_natoms = graph_inputs[0].shape[1]
        bond_labels = []
        sp_labels = []
        for smi, edit in zip(r_tmp, labels_tmp):
            l = get_bond_label(smi, edit, max_natoms)
            bond_labels.append(l[0])
            sp_labels.append(l[1])

        # Convert to torch tensors
        graph_inputs = [torch.from_numpy(np.array(g)) for g in graph_inputs]
        bond_labels = torch.from_numpy(np.array(bond_labels))
        # max_length = max(len(s) for s in sp_labels) #Here we changed
        # padded_sp_labels = [s + [0] * (max_length - len(s)) for s in sp_labels]
        # sp_labels = torch.tensor(padded_sp_labels)
        sp_labels = [torch.tensor(s) for s in sp_labels] #here we changed differently


        if self.detailed:
            all_ratoms, all_rbonds = reactant_tracking(smiles_tmp, hard=self.reagents)
            return graph_inputs, bond_labels, sp_labels, all_ratoms, all_rbonds, smiles_tmp, labels_tmp
        else:
            return graph_inputs, bond_labels

class Candidate_DataLoader(FileLoader):
    FIXED_MAX_NUM_ATOMS = 151
    FIXED_MAX_NUM_NBS = 76

    def __init__(self, filename, batch_size=1, chunk_size=10000, cutoff=150, core_size=16, shuffle=True, testing=False):
        super().__init__(filename, batch_size, chunk_size, shuffle and not testing)
        self.cutoff = cutoff
        self.core_size = core_size
        self.testing = testing

    def pad_array(self, arr, target_shape, pad_value=0):
        arr = np.asarray(arr)
        ndim = arr.ndim

        # Expand target_shape to match ndim
        if len(target_shape) < ndim:
            target_shape = list(target_shape) + list(arr.shape[len(target_shape):])

        pad_width = []
        for i in range(ndim):
            pad_before = 0
            pad_after = max(0, target_shape[i] - arr.shape[i])
            pad_width.append((pad_before, pad_after))
        
        return np.pad(arr, pad_width, mode='constant', constant_values=pad_value)

    @time_it_and_log()
    def data_generation(self, batch):
        smiles_tmp, labels_tmp, cand_tmp = [], [], []
        for x in batch:
            parts = x.strip().split()
            smiles_tmp.append(parts[0])
            labels_tmp.append(parts[1])
            cand_tmp.append(parts[2:])

        cand_bond = []
        for cand in cand_tmp:
            bonds = []
            for i in range(0, len(cand), 2):
                x, y, t = cand[i].split('-')
                x, y = tuple(sorted([int(float(x)) - 1, int(float(y)) - 1]))
                bonds.append((x, y, float(t), float(cand[i + 1])))
            cand_bond.append(bonds)

        gold_bond = []
        for edits in labels_tmp:
            label = set()
            for bond in edits.split(';'):
                x, y, t = bond.split('-')
                x, y = tuple(sorted([int(float(x)) - 1, int(float(y)) - 1]))
                label.add((x, y, float(t)))
            gold_bond.append(label)

        batch_graphs = []
        batch_confs = [] if self.testing else None

        # Load all graphs (no max calc, just use fixed sizes)
        for i, smi in enumerate(smiles_tmp):
            r, _, p = smi.split('>')
            if not self.testing:
                graph_inputs, _ = s2g_edit(r, p, cand_bond[i], gold_bond[i], cutoff=self.cutoff, core_size=self.core_size)
            else:
                graph_inputs, conf = s2g_edit(r, None, cand_bond[i], None, cutoff=self.cutoff, core_size=self.core_size,
                                              idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1, testing=self.testing)
                batch_confs.append(conf)
            batch_graphs.append(graph_inputs)

        batch_inputs = []
        for idx in range(6):
            padded_tensors = []
            for g in batch_graphs:
                arr = g[idx]
                shape = list(arr.shape)

                if idx == 0:  # input_atom: [num_atoms, 89]
                    target_shape = (self.FIXED_MAX_NUM_ATOMS, shape[1])
                elif idx in [2, 3]:  # atom_graph, bond_graph: [num_atoms, max_nb, 2]
                    target_shape = (self.FIXED_MAX_NUM_ATOMS, self.FIXED_MAX_NUM_NBS, 2)
                elif idx == 1:  # input_bond: [num_atoms, max_nb, 5]
                    target_shape = (self.FIXED_MAX_NUM_ATOMS, self.FIXED_MAX_NUM_NBS, shape[2])
                elif idx == 4:  # num_nbs: [num_atoms]
                    target_shape = (self.FIXED_MAX_NUM_ATOMS,)
                else:  # idx == 5 (or others)
                    target_shape = shape
                    target_shape[0] = self.FIXED_MAX_NUM_ATOMS

                padded_arr = self.pad_array(arr, target_shape, pad_value=0)
                padded_tensors.append(padded_arr)

            batch_tensor = torch.from_numpy(np.stack(padded_tensors, axis=0)).float()
            batch_inputs.append(batch_tensor)

        # Now batch_inputs all have fixed batch size and fixed tensor dims:
        # batch_inputs[0] shape: (B, 151, 89) or with neighbors (B, 151, 76, ...)
        # Make sure dims for neighbor axis too if needed (sometimes input_atom needs neighbor dim)

        # Example: If input_atom has shape (B, 151, N_nb, F), fix that too by padding:
        input_atom = batch_inputs[0]
        input_bond = batch_inputs[1]
        atom_graph = batch_inputs[2]
        bond_graph = batch_inputs[3]
        num_nbs = batch_inputs[4]
        input_5 = batch_inputs[5]

        # Extra padding if input_atom has neighbors dim < fixed max neighbors
        if input_atom.ndim == 4:
            B, N, neighbor_count, F = input_atom.shape
            if neighbor_count < self.FIXED_MAX_NUM_NBS:
                pad_size = self.FIXED_MAX_NUM_NBS - neighbor_count
                pad_tensor = torch.zeros((B, N, pad_size, F), dtype=input_atom.dtype, device=input_atom.device)
                input_atom = torch.cat([input_atom, pad_tensor], dim=2)
        else:
            # If no neighbor dim, add one with zeros
            input_atom = input_atom.unsqueeze(2)  # add neighbors dim
            pad_tensor = torch.zeros((input_atom.shape[0], input_atom.shape[1], self.FIXED_MAX_NUM_NBS - 1, input_atom.shape[-1]),
                                     dtype=input_atom.dtype, device=input_atom.device)
            input_atom = torch.cat([input_atom, pad_tensor], dim=2)

        # Similarly pad input_bond if needed (should be shape (B, 151, 76, 5))
        if input_bond.shape[2] < self.FIXED_MAX_NUM_NBS:
            pad_size = self.FIXED_MAX_NUM_NBS - input_bond.shape[2]
            pad_tensor = torch.zeros((input_bond.shape[0], input_bond.shape[1], pad_size, input_bond.shape[3]),
                                     dtype=input_bond.dtype, device=input_bond.device)
            input_bond = torch.cat([input_bond, pad_tensor], dim=2)

        # Return padded tensors replacing originals
        batch_inputs[0] = input_atom
        batch_inputs[1] = input_bond

        if not self.testing:
            labels = [g[6] for g in batch_graphs]
            max_label_len = max(len(l) for l in labels)
            padded_labels = []
            for l in labels:
                pad_len = max_label_len - len(l)
                padded_labels.append(np.pad(l, (0, pad_len), mode='constant'))
            batch_labels = torch.from_numpy(np.stack(padded_labels, axis=0)).float()

            return tuple(batch_inputs), batch_labels
        else:
            return tuple(batch_inputs), batch_confs, smiles_tmp
