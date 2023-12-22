import math

import networkx as nx
import numpy as np
import pandas as pd
import os
import argparse
from multiprocessing import Pool

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
from rdkit.Chem.SaltRemover import SaltRemover
from Dataset import MolNet


remover = SaltRemover()
smile_graph = {}
meta = ['W', 'U', 'Zr', 'He', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Rb', 'Sr', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Gd', 'Tb', 'Ho', 'W', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac']


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                               ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                                'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                                'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                                'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) +
                    [atom.GetIsAromatic()] + get_ring_info(atom))


def get_ring_info(atom):
    ring_info_feature = []
    for i in range(3, 9):
        if atom.IsInRingSize(i):
            ring_info_feature.append(1)
        else:
            ring_info_feature.append(0)
    return ring_info_feature

def order_gnn_features(bond):
    weight = [1, 2, 3, 1.5]
    bt = bond.GetBondType()
    bond_feats = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC]

    for i, m in enumerate(bond_feats):
        if m == True and i != 0:
            b = weight[i]
        elif m == True and i == 0:
            if bond.GetIsConjugated() == True:
                b = 1.4
            else:
                b = 1
        else:pass
    return b


def order_tf_features(bond):
    weight = [1, 2, 3, 1.5]
    bt = bond.GetBondType()
    bond_feats = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC]
    for i, m in enumerate(bond_feats):
        if m == True:
            b = weight[i]
    return b        


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smiletopyg(smi):
    g = nx.Graph()
    mol = Chem.MolFromSmiles(smi)
    c_size = mol.GetNumAtoms()

    features = []
    for i, atom in enumerate(mol.GetAtoms()):
        g.add_node(i)
        feature = atom_features(atom)
        features.append((feature / sum(feature)).tolist()) 

    c = []
    adj_order_matrix = np.eye(c_size)
    dis_order_matrix = np.zeros((c_size,c_size))
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bfeat = order_gnn_features(bond)
        g.add_edge(a1, a2, weight=bfeat)

        tfft = order_tf_features(bond)
        adj_order_matrix[a1, a2] = tfft
        adj_order_matrix[a2, a1] = tfft
        if bond.GetIsConjugated():
            c = list(set(c).union(set([a1, a2])))

    g = g.to_directed()
    edge_index = np.array(g.edges).tolist()

    edge_attr = []
    for w in list(g.edges.data('weight')):
        edge_attr.append(w[2])  # ？？？

    for i in range(c_size):
        for j in range(i,c_size):
            if adj_order_matrix[i, j] == 0 and i != j:
                conj = False

                try:
                    paths = list(nx.node_disjoint_paths(g, i, j))
                    if len(paths) > 1:
                        paths = sorted(paths, key=lambda p: len(p), reverse=False)
                    for path in paths:
                        if set(path) < set(c):
                            conj = True
                            break
                except nx.NetworkXNoPath:
                    # 处理没有路径的情况，将值设置为正无穷大
                    print(f"No path found for SMILES: {smi}")
                    continue  # 跳过当前 SMILES 的处理

                if conj:
                    adj_order_matrix[i, j] = 1.2
                    adj_order_matrix[j, i] = 1.2
                else:
                    path = paths[0]
                    dis_order_matrix[i, j] = len(path) - 1
                    dis_order_matrix[j, i] = len(path) - 1

    g = [c_size, features, edge_index, edge_attr, adj_order_matrix, dis_order_matrix]
    return [smi, g]


def write(res):
    smi, g = res
    smile_graph[smi] = g


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransFoxMol')
    parser.add_argument('--moldata',default='test4', type=str, help='dataset name to process')
    parser.add_argument('--task',default='clas', type=str, choices=['clas', 'reg'], help='classification or regression')
    parser.add_argument('--ncpu', type=int, default=4, help='number of cpus to use (default: 4)')
    args = parser.parse_args()
    moldata = args.moldata

    if moldata in ['BS', 'SHM', 'FHM',  'RT1','Smiles','test4']:
        task = 'clas'
        if moldata == 'RT1':
            numtasks = 1
            labell = ['labels']
        elif moldata == 'BS_origin':
            numtasks = 1
            labell = ['labels']
        elif moldata == 'BS':
            numtasks = 1
            labell = ['labels']
        elif moldata == 'FHM':
            numtasks = 1
            labell = ['labels']
        elif moldata == 'SHM':
            numtasks = 1
            labell = ['labels']
        elif moldata == 'Smiles':
            numtasks = 1
            labell = ['labels']
        elif moldata == 'test4':
            numtasks = 1
            labell = ['labels']

        else:
            numtasks = 1
            labell = ['labels']

    processed_data_file = 'dataset/processed/' + moldata + '_pyg.pt'
    if not os.path.isfile(processed_data_file):
        try:
            df = pd.read_csv('./dataset/raw/'+moldata+'.csv')
        except:
            print('Raw data not found! Put the right raw csvfile in **/dataset/raw/')
        compound_iso_smiles = np.array(df['smiles']) 
        ic50s = np.array(df[labell])

        pool = Pool(args.ncpu)
        smis = []
        y = []
        result = []

        for smi, label in zip(compound_iso_smiles, ic50s):
            smis.append(smi) 
            y.append(label)             
            result.append(pool.apply_async(smiletopyg, (smi,)))
        pool.close()
        pool.join()

        for res in result:
            smi, g = res.get()
            mol = Chem.MolFromSmiles(smi)
            morgan_fp = torch.tensor(AllChem.GetMorganFingerprintAsBitVect(mol, 2)).view(1, -1)
            maccs_fp = torch.tensor(MACCSkeys.GenMACCSKeys(mol)).view(1, -1)
            rdit_fp = torch.tensor(Chem.RDKFingerprint(mol)).view(1, -1)
            smile_graph[smi] = {
                'graph_data': g,
                'morgan_fp': morgan_fp,
                'maccs_fp': maccs_fp,
                'rdit_fp': rdit_fp,
            }

        MolNet(root='./dataset', dataset=moldata, xd=smis, y=y, smile_graph=smile_graph)
    
