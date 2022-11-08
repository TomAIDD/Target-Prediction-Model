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

with open('COCONUT_clean.pkl','rb') as c:
    COCONUT = pickle.load(c)
with open('chembl_clean.pkl','rb') as chem:
    CHEMBL = pickle.load(chem)
intersection = []
for i in COCONUT:
    if i in CHEMBL:
        intersection.append(i)
print(len(intersection))
with open('Intersection.pkl','wb') as I:
    pickle.dump(intersection,I,protocol = 4) 