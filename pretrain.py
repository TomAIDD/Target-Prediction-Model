import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit.Chem import AllChem
from sklearn import metrics   #auroc库
import pickle
from tqdm import tqdm, trange
import sys

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

#准备ecfp4指纹(features)和label_data(labels)
with open('ecfp4.pkl','rb') as e:
    ecfp4 = pickle.load(e)
    ecfp4 = torch.tensor(ecfp4)
with open('tensor.pkl','rb') as t:
    label_data = pickle.load(t)
with open('loss_weight.pkl','rb') as l:
    loss_weight = pickle.load(l)

class netdata(Dataset):
    def __init__(self, ecfp4, label_data, loss_weight, label_num=238):          
        self.ecfp4 = ecfp4
        self.label_data = label_data
        self.loss_weight = loss_weight           #存的loss weight
        if self.label_data.shape[0] != self.ecfp4.shape[0]:
            print(self.label_data.shape[0])
            print(self.ecfp4.shape[0])
            raise  ValueError('incompatible shape of label and fingerprint')
    
    def __len__(self):
        return self.ecfp4.shape[0]
    
    def __getitem__(self, idx):
        input_tensor = self.ecfp4[idx]
        label_tensor = self.label_data[idx]
        weight_tensor = torch.tensor(self.loss_weight[idx])
        return input_tensor, label_tensor,weight_tensor


class MLP(nn.Module):
    def __init__(self, h_list,out_features=238):
        super(MLP, self).__init__()
        h_list1, h_list2 = h_list.copy(), h_list.copy()
        h_list2.append(out_features)
        h_list1.insert(0, 2048)
        self.linears = nn.ModuleList([nn.Linear(x, y) for x, y in zip(h_list1 , h_list2)])
        self.BNs = nn.ModuleList([nn.BatchNorm1d(m) for m in h_list])
        self.h_list = h_list
    
    def forward(self, x):
        x = x.float()
        for i in range(len(self.linears) - 1):
            x = self.linears[i](x)
            x = self.BNs[i](x)
            x = F.leaky_relu(x)
        x = torch.sigmoid(self.linears[-1](x))
        return x

def get_k_fold_data(k, i, data):
    assert k > 1
    fold_size = data.shape[0] // k
    data_train = None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        data_part = data[idx, :]
        if j == i: #第i折作valid
            data_valid = data_part
        elif data_train is None:
            data_train = data_part
        else:
            data_train = torch.cat((data_train, data_part), dim=0) #dim=0增加行数，竖着连接  
    return data_train, data_valid
#5折交叉验证
for i in range(5):    
    ecfp4_train,ecfp4_valid = get_k_fold_data(5,i,ecfp4)
    label_data_train,label_data_valid = get_k_fold_data(5,i,label_data)
    loss_weight_train,loss_weight_valid = get_k_fold_data(5,i,loss_weight)
    
    train_data = netdata(ecfp4_train, label_data_train, loss_weight_train, label_num=238)
    test_data = netdata(ecfp4_valid, label_data_valid, loss_weight_valid, label_num=238)
    
    net = MLP(h_list=[2048, 2048, 1024], out_features=238)
    net.to(device)
    net = nn.DataParallel(net, device_ids = [2, 0, 1])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0000005, weight_decay=0.001, betas=(0.9, 0.999))
    train_data_loader = DataLoader(train_data, shuffle=True, batch_size=128)
    test_data_loader = DataLoader(test_data, shuffle=True, batch_size=128)
    #训练
    au_roc = []
    au_roc_train = []
    au_roc_final = []
    for epoch in trange(100):
        if epoch % 5 == 0:
            with torch.no_grad():
                net.eval()
                for cat_c, test_data_batch in enumerate(test_data_loader):
                    batch = test_data_batch[0].to(device).float()
                    test_output_batch = net(batch)
                
                    test_label_batch = test_data_batch[1]
                    if cat_c == 0:
                        test_output = test_output_batch.cpu()
                        test_label = test_label_batch.cpu()
                    else:
                        test_output = torch.cat((test_output, test_output_batch.cpu()), dim=0)
                        test_label = torch.cat((test_label, test_label_batch.cpu()), dim=0)
            #先用tensor.cpu()转到cpu上再转成numpy形式
                test_label,test_output = test_label.numpy(),test_output.numpy()
                index_x,index_y = np.where(test_label != -1)
                test_label = test_label[index_x,index_y]
                test_output = test_output[index_x,index_y]
                
                test_label = torch.tensor(test_label)
                test_output = torch.tensor(test_output)
                fpr, tpr, _ = metrics.roc_curve(test_label.view(-1).detach().numpy(), test_output.view(-1).detach().numpy(), pos_label=1 )
                au_roc.append(metrics.auc(fpr, tpr))
                print(' \r  training at epoch %d now have auroc %f '%( epoch, au_roc[-1]))
                sys.stdout.flush()
    
        #后面写训练函数
        for batch in train_data_loader:
            net.train()
            input_batch, label_batch = batch[0].to(device).float(), batch[1].to(device).float()
            optimizer.zero_grad()
            outputs = net(input_batch)
        
            loss = F.binary_cross_entropy(outputs, label_batch, weight=batch[2].to(device).float())
            loss.backward()
            optimizer.step()

        

    
        with torch.no_grad():
            net.eval()
            
            outputs_ = outputs.clone().detach().cpu()
            label_batch_ = label_batch.clone().detach().cpu()
            
            outputs_,label_batch_ = outputs_.numpy(),label_batch_.numpy()
            index_x,index_y = np.where(label_batch_ != -1)
            outputs_ = outputs_[index_x,index_y]
            label_batch_ = label_batch_[index_x,index_y]
            
            outputs_ = torch.tensor(outputs_)
            label_batch_ = torch.tensor(label_batch_)
            
            fpr, tpr, _ = metrics.roc_curve(label_batch_.view(-1).detach().numpy(), outputs_.view(-1).detach().numpy(), pos_label=1 )
            au_roc_train.append(metrics.auc(fpr, tpr))
            print(' \r  ------training now have training set auroc %f------ '%(au_roc_train[-1]))
            sys.stdout.flush()
            print(f'\rtask epoch {epoch} running', end=' ')


