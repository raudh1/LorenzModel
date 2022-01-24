import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from random import randint
import os
import sys
#from torch import Dataset, Dataloader
torch.set_default_tensor_type('torch.DoubleTensor')

dirName = './RESULTS'
if not os.path.exists(dirName):
    os.makedirs(dirName)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else: 
    device = torch.device('cpu')

print(device)

class Sequence(nn.Module):
    def __init__(self, hidden, layer, features, dropout):
        super(Sequence, self).__init__()
        self.hidden   = hidden
        self.layer    = layer
        self.features = features

        self.lstm1  = nn.LSTM(self.features, self.hidden, self.layer, dropout=dropout)
        self.linear = nn.Linear(self.hidden, self.features)
 
    def forward(self, input, h_t, c_t):
        self.lstm1.flatten_parameters()
        out, (h_t, c_t) = self.lstm1(input, (h_t, c_t))
        output = out.view(input.size(0)*input.size(1),self.hidden)
        output = self.linear(output)
        output = output.view(input.size(0),input.size(1),self.features)
        return output 


if __name__ == '__main__':
    np.random.seed(0)
    # load data and make training set
    hidden   = 50
    layer    = 2
    features = 3
    dropout  = 0
    # build the model
    case = 'FULL'
        
    ini = 25


    for k in [0,1] :
        name = './DATA/'+case+'_'+str(k)+'.npy'
        data = np.load(name)
        data = np.expand_dims(data.T, axis=1)
        data = data[:100] # reduce the number of time steps (db)
        print("Data shape: ", data.shape)
        input  = torch.from_numpy(data[:-1,:,:]).double().to(device)
        target = torch.from_numpy(data[1:,:,:]).double().to(device)

        seq = Sequence(hidden,layer,features,dropout).double().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(seq.parameters(), lr =0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=50, min_lr = 5e-5)
        dirName = './RESULTS/'+case+'_'+str(k)
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        h_0 = torch.normal(mean=0.0, std=torch.ones(layer,input.size(1), hidden, dtype=torch.double)).to(device)
        c_0 = torch.normal(mean=0.0, std=torch.ones(layer,input.size(1), hidden, dtype=torch.double)).to(device)

        if k==0:
            torch.save(seq.state_dict(),'./init_model.pt')
        else:
            state_dict = torch.load('../0/init_model.pt', map_location=device)
            seq.load_state_dict(state_dict)

        err = 10
        loss1 = 1
        i = 0
            for k in range(1):
        print(k)
        name = './data/'+case+'_'+str(k)+'.npy'
data = np.load(name)
#data = np.expand_dims(data.T, axis=1)
data=np.moveaxis(data,-1,0)
    data =data[:3000]
    print(data.shape)
    input  = torch.from_numpy(data[:-1,:,:]).double().to(device)
    target = torch.from_numpy(data[1:,:,:]).double().to(device)
    
    seq = Sequence(hidden,layer,features,dropout).double().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(seq.parameters(), lr =0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=50, min_lr = 5e-5)
    dirName = './RESULTS/'+case+'_'+str(k)
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    h_0 = torch.normal(mean=0.0, std=torch.ones(layer,input.size(1), hidden, dtype=torch.double)).to(device)
    c_0 = torch.normal(mean=0.0, std=torch.ones(layer,input.size(1), hidden, dtype=torch.double)).to(device)

if k==0:
    torch.save(seq.state_dict(),'./init_model.pt')
    else:
        state_dict = torch.load('../0/init_model.pt', map_location=device)
        seq.load_state_dict(state_dict)

err = 10
    loss1 = 1
    i = 0
    epoch=1000
    fish = np.zeros(epoch)
    
    train_loss= []
    #while loss1 > 1e-3 and i<2000: old step counter
    while i<epoch:
        optimizer.zero_grad()
        out = seq(input,h_0,c_0)
        loss = criterion(out, target.to(device))
        fish_=grad(loss, seq.parameters(),retain_graph=True)
        fish_norm = 0
        for partial_deriv in fish_ :
            fish_norm += torch.norm(partial_deriv)**2
        fish[i] = torch.sqrt(fish_norm)
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step(loss)
        
        torch.save(seq.state_dict(),dirName+'/mytraining.pt')
        np.savetxt(dirName+'/loss.out',np.array([loss1]))
        
        if i%(100)==0:
            loss2 = loss
            err = np.abs(loss2.item()-loss1)/100
            loss1 = loss2.item()
            print('FILE: ', k,'STEP TEST: ', i, 'test loss:', loss2.item(), 'lr: ', optimizer.param_groups[0]['lr'], 'err: ', err)
        i += 1
    sys.stdout.flush()
    print('END: '+str(k))
    
