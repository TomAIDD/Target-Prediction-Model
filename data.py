import numpy as np
import torch
import torch.utils.data as data
from torch.nn import init
from torch import nn
from rdkit import Chem
from rdkit import DataStructs

from rdkit.Chem import AllChem
from rdkit.Chem import Draw

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
from tqdm import tqdm,trange

with open('clean_dict.pkl','rb') as c:
    cd = pickle.load(c)
with open('filtered_targets.pkl','rb') as f:
    targets = pickle.load(f)
with open('Intersection.pkl','rb') as i:
    intersection = pickle.load(i)
data = pd.read_csv('chembl_raw data.csv')
data['target_chembl_id'] = data['target_chembl_id'].str.replace('CHEMBL','')
data['target_chembl_id'] = data['target_chembl_id'].astype(int)

def GetFPFromSmiles(smiles):
    if smiles is not None:
        return AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(smiles), 2, 2048) #
chem_test = 0
chem_train = 0
ecfp4_training = []
ecfp4_test = []
training_tensor = torch.full([534763,238],-1,dtype = int)
test_tensor = torch.full([3724,238],-1,dtype = int)
for i in tqdm(cd):
    if i in intersection:
        fp = GetFPFromSmiles(i)
        ecfp4_test.append(fp)
        for j in cd[i]:
            t = data.iloc[j,3]
            if t in targets:
                location = targets.index(t)
                activity = data.iloc[j,2]
                test_tensor[chem_test,location] = activity
        chem_test += 1
    else:
        fp = GetFPFromSmiles(i)
        ecfp4_training.append(fp)
        for k in cd[i]:
            t = data.iloc[k,3]
            if t in targets:
                location = targets.index(t)
                activity = data.iloc[k,2]
                training_tensor[chem_train,location] = activity
            else:
                pass
        chem_train += 1
print(training_tensor.size())
print(test_tensor.size())
print(len(ecfp4_training))
print(len(ecfp4_test))

def loss_weight(tensor):
    dim0,dim1 = tensor.shape
    weight = torch.empty(dim0,dim1)
    Ni,Na = 0,0
    for i in trange(dim0):
        for j in range(dim1):
            if tensor[i][j] == -1:
                pass
            elif tensor[i][j] == 0:
                Ni +=1
            elif tensor[i][j] == 1:
                Na +=1
        if Na > 0:
            for j in range(dim1):
                if tensor[i][j] == 1:
                    weight[i][j] = Ni/Na
                elif tensor[i][j] == 0:
                    weight[i][j] = 1
                elif tensor[i][j] == -1:
                    weight[i][j] = 0
        elif Na == 0:
            for j in range(dim1):
                if tensor[i][j] == 0:
                    weight[i][j] = 1
                elif tensor[i][j] == -1:
                    weight[i][j] = 0
        Ni,Na = 0,0
    return weight

train_w = loss_weight(training_tensor)
test_w = loss_weight(test_tensor)

with open('tensor.pkl','wb') as t:
    pickle.dump(training_tensor,t,protocol = 4)
with open('test_tensor.pkl','wb') as t1:
    pickle.dump(test_tensor,t1,protocol = 4)

with open('ecfp4.pkl','wb') as e:
    pickle.dump(ecfp4_training,e,protocol = 4)
with open('ecfp4_test.pkl','wb') as e1:
    pickle.dump(ecfp4_test,e1,protocol = 4)

with open('loss_weight.pkl','wb') as l:
    pickle.dump(train_w,l,protocol = 4)
with open('loss_weight_test.pkl','wb') as l1:
    pickle.dump(test_w,l1,protocol = 4)
    