import numpy as np
import torch
import torch.utils.data as data
from torch.nn import init
from torch import nn
from rdkit import Chem
from rdkit import DataStructs

from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Descriptors import MolWt
from molvs import Standardizer

from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.SaltRemover import SaltRemover

from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

import matplotlib.pyplot as plt 
from os import path
import pandas as pd
import pickle

ATOM_SYMBOLS = ['H', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']
def smile_standard(smiles:str):
    mol = Chem.MolFromSmiles(smiles)
    s = Standardizer()
    if MolWt(mol) > 2000:
        return 0
    BadAtom = 0
    count = 0
    for atom in mol.GetAtoms():
        count+=1
        if (atom.GetSymbol() in ATOM_SYMBOLS) is not True:
            BadAtom = 1
            break
    if BadAtom==1 or count==1:
           return 0
    mol = s.standardize(mol)
    mol = s.fragment_parent(mol)
    remover = SaltRemover()
    mol = remover.StripMol(mol, dontRemoveEverything=True)
    SMILES = Chem.MolToSmiles(mol)
    return SMILES

with open('COCONUT_smiles.pkl','rb') as c:
    COCONUT = pickle.load(c)

data = pd.read_csv('chembl_raw data.csv')  #pd.dataframe格式
chembl_smiles = data.iloc[:,1]
clean_id = data.iloc[:,0]

COCONUT_clean = []
chembl_clean = []
clean_dict = {}
for i in COCONUT:
    try:
        smile = smile_standard(i)
    except:
        pass
    if smile != 0:
        COCONUT_clean.append(smile)

for i in range(len(chembl_smiles)):
    try:
        smile = smile_standard(chembl_smiles.iloc[i])
        clean_dict.setdefault(smile,[]).append(clean_id.iloc[i])
    except:
        pass
    if smile != 0:
        chembl_clean.append(smile)
print(clean_dict[0])
del clean_dict[0]
with open('COCONUT_clean.pkl','wb') as c:
    pickle.dump(COCONUT_clean,c,protocol = 4)
with open('chembl_clean.pkl','wb') as chem:
    pickle.dump(chembl_clean,chem,protocol = 4)
with open('clean_dict.pkl','wb') as d:
    pickle.dump(clean_dict,d,protocol = 4)