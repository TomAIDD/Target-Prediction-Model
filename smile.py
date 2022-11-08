#coding:utf-8
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

data = pd.read_csv('chembl_raw data.csv')
smiles = data.sort_values(by='canonical_smiles')
smiles_1 = smiles.iloc[:,3]
smiles_1 = smiles_1.str.replace('CHEMBL','')
smiles_1 = smiles_1.astype(int)
#以靶点为基础排序
s_target = data.iloc[:,3]
s_target = s_target.str.replace('CHEMBL','')
s_target = s_target.astype(int)
s_target = s_target.sort_values()
#将靶点用字典编号
i = 0
j = 0
targets = {}
while i < len(s_target):
    while i!=len(s_target)-1 and s_target.iloc[i]==s_target.iloc[i+1]:
        i+=1
    m = s_target.iloc[i]
    targets[m] = j
    i+=1
    j+=1                #用相似代码计数，输出靶点数为5406 输出分子数为548775

#填入每个分子的靶点
i = 0
mol = 0
while i < len(smiles):
    j = i
    while j!=len(smiles)-1 and smiles.iloc[j,1]==smiles.iloc[j+1,1]:
        j+=1
    for k in range(i,j+1):
        t = targets[smiles_1.iloc[k]] 
        train_data[mol,t] = smiles.iloc[k,2]
    mol +=1
    i = j+1

def target_filter(target):
    dim0,dim1 = target.shape
    points = 0
    out_targets = []
    for i in range(dim1):
        for j in range(dim0):
            if target[j][i] == 1:
                points += 1
        if points < 10:
            out_targets.append(i)
        points = 0
    target = target.numpy()
    target = np.delete(target,out_targets,axis = 1)  #需要存储到一个变量里，如a = np.delete（...）
    target = torch.from_numpy(target)
    print(target.shape)
    return target


train_data_1 = target_filter(train_data)


import pickle
with open('tensor.pkl','wb') as t:
    pickle.dump(train_data_1,t,protocol = 4)


