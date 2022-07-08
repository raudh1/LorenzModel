import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from random import randint
import os
import sys
import string
#from torch import Dataset, Dataloader
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt



#################################################


#Load DATA FUNCTION



#################################################
def load_data(case,sampling=10,start=0):
  """
  This Function load data, from different dataset. 
  Fixed Point traj = MANI
  Random  	   = PART
  Long Trajectory  = FULL
  """


  if (case=='MANI'or case=='PART' ):

    name=mypath+'/'+case+'_'+str(k)+'.npy'
    data = np.load(name).astype('float64')
    data=np.moveaxis(data,-1,0)
    data=data[::sampling,:1,:]                    # one of 9 trajectories and sampling
    N=(len(data)*3)//4
    V=len(data)//8
    T=len(data)//8
    train_data = torch.from_numpy(data[start:start+N,:,:]).double().to(device)
    valid_data = torch.from_numpy(data[start+N:start+N+V,:,:]).double().to(device)
    test_data = torch.from_numpy(data[start+N+V:start+N+V+T,:,:]).double().to(device)

    return train_data, valid_data, test_data

  elif (case=='other'):
    name= mypath+'/other_traj.npy'
    data = np.load(name).astype('float64')
    data=np.moveaxis(data,-1,0)
    data=data[::sampling,:1,:]

    N=(len(data)*3)//4
    V=len(data)//8
    T=len(data)//8

    train_data = torch.from_numpy(data[start:start+N,:,:]).double().to(device)
    valid_data = torch.from_numpy(data[start+N:start+N+V,:,:]).double().to(device)
    test_data = torch.from_numpy(data[start+N+V:start+N+V+T,:,:]).double().to(device)

    return train_data, valid_data, test_data
     
  elif (case=='zero'):
    name=mypath+'/zeros_traj.npy'
    data = np.load(name).astype('float64')

    data=np.moveaxis(data,-1,0)
    data=data[::sampling,:1,:]

    N=(len(data)*3)//4
    V=len(data)//8
    T=len(data)//8

    train_data = torch.from_numpy(data[start:start+N,:,:]).double().to(device)
    valid_data = torch.from_numpy(data[start+N:start+N+V,:,:]).double().to(device)
    test_data = torch.from_numpy(data[start+N+V:start+N+V+T,:,:]).double().to(device)

    return train_data, valid_data, test_data

  elif (case=='FULL'):
    name=mypath+'/'+case+'_'+str(k)+'.npy'
    data = np.load(name).astype('float64')
    data = np.expand_dims(data.T, axis=1)
    data=data[::sampling,:1,:]

    N=(len(data)*3)//4
    V=len(data)//8
    T=len(data)//8

    train_data = torch.from_numpy(data[start:start+N,:,:]).double().to(device)
    valid_data = torch.from_numpy(data[start+N:start+N+V,:,:]).double().to(device)
    test_data = torch.from_numpy(data[start+N+V:start+N+V+T,:,:]).double().to(device)

    return train_data, valid_data, test_data
    
    
    
    
    
##############################################

#TRAINING 

##############################################

def train(model,train_data,valid_data,lr=0.05,epoch=300):

      print('input shape=',train_data.shape," target shape =",valid_data.shape)
      criterion = nn.MSELoss()
      optimizer = optim.Adam(model.parameters(), lr) #learning rate fixed#
  
  
      err = 10
      loss1 = 1
      i = 0
      valid_loss=[]
      loss_grad=[]
      train_loss= []
      i=0
      
      min_perf = 90000
      while i<epoch: 
          model.train()
          optimizer.zero_grad()
          out,_ =model(train_data[:-1]) #INPUT
          loss = criterion(out, train_data[1:]) # OUTPUT
          loss.backward()

          train_loss.append(loss.item())

          optimizer.step()
          # validation
          with torch.no_grad():
            model.eval()
            pred,_ = model(valid_data[:-1])
            validloss = criterion(pred, valid_data[1:])
            valid_loss.append(validloss.item())
     
          if i%(50)==0:
              loss2 = loss
              err = np.abs(loss2.item()-loss1)/10
              loss1 = loss2.item()
              print(i, "train loss", train_loss[-1], "valid loss",
                  valid_loss[-1])
          i += 1
          sys.stdout.flush()
          if validloss.item() < min_perf :
            min_perf = validloss.item()
            torch.save(model.state_dict(), './param')
      
      print("last epoch: ", i)
      

      return train_loss, valid_loss
#################################


#Generate Trajectory

#################################
def generate_traj(model,inp,h,f): # h is history and f is future inp is input that can be train or test or validation set
  traj_1step ,_=  model(inp[:h+f])  # 1 step
  traj = model.predict(inp[:h],N=f) #generated

  return traj_1step.detach().numpy(), traj.detach().numpy()

##################################

#PLOTS 

##################################

def plot_traj(traj_1step,traj,h,coordinate='x',data=False,error=False): # h is history traj is generated traj
  """
  Generates Trajectory of x,y,z 
  """
  if coordinate=='x':
    i=0
  if coordinate=='y':
    i=1
  if coordinate=='z':
    i=2

  plt.figure(figsize=(14.4,8.8))
  plt.plot(traj_1step[:,0,i],label='1step model')
  plt.plot(traj[:,0,i],label='traj_generate')
  if data is True:
    plt.plot(input[:,0,i],label='real data')
  if error is True:
    print(np.mean(np.power(traj_1step[h:,0,i]-traj[h:,0,i],2)))
  plt.scatter(h,traj_1step[h,0,i],color='red')

  plt.xlim(h-100,len(traj))
  plt.ylabel(rf'${coordinate}(t)$',fontsize=20)
  plt.xlabel(r'$t$',fontsize=20)
  plt.legend()
  plt.show()
  print(traj_1step.shape, traj.shape)
  print(np.linalg.norm(traj_1step-traj))
#########################

#PLOT Lorenz System in 3D

#########################
def plot_traj_3D(traj,traj_1step,inp,h=1000,f=10,data=False):
  fig = plt.figure(figsize=(20,15))
  ax = fig.gca(projection="3d")
  start=h

  ax.plot(traj_1step[start:start+f,0,0],traj_1step[start:start+f,0,1],traj_1step[start:start+f,0,2],label='1 step traj',linestyle='--')
  ax.plot(traj[start:start+f,0,0],traj[start:start+f,0,1],traj[start:start+f,0,2],label='generated traj')
  
  if data is True:
    ax.plot(inp[start:h+f,0,0],inp[start:h+f,0,1],inp[start:h+f,0,2],label='real traj')

  
  ax.scatter3D(traj_1step[h,0,0],traj_1step[h,0,1],traj_1step[h,0,2],c='green',label='starting point')
  ax.set_xlabel(r'$x$',fontsize=30)
  ax.set_ylabel(r'$y$',fontsize=30)
  ax.set_zlabel(r'$z$',fontsize=30)

  ax.legend(fontsize=15)
















