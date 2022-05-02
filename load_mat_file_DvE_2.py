from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from pyentrp import entropy as ent
import torch
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

bins = 20
sizeMatrix=6


#data = []
entropyF = np.zeros(100)



#for i in range(100):
#   data = torch.load('./TRAIN/FULL_'+str(i)+'.pt')
#   for j in range(9):
#       entropyF[i] += ent.shannon_entropy(data[3000*j:3000*(j+1),0,0])/9
#print('average entropy in FULL ',str(np.mean(entropyF)),'order = ',str(sizeMatrix))

#data = []
#entropyP = np.zeros(100)
#for i in range(100):
#   data = torch.load('./TRAIN/PART_'+str(i)+'.pt')
#   for j in range(9):
#       entropyP[i] += ent.shannon_entropy(data[:,j,0])/9
#print('average entropy in PART '+str(np.mean(entropyP)),'order = ',str(sizeMatrix))

#data = []
#entropyM = np.zeros(100)
#for i in range(100):
#   data = torch.load('./TRAIN/MANI_'+str(i)+'.pt')
#   for j in range(9):
#       entropyM[i] += ent.shannon_entropy(data[:,j,0])/9
#print('average entropy in MANIFOLD '+str(np.mean(entropyM)),'order = ',str(sizeMatrix))

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


#fig = plt.figure(figsize=(7,5))
#ax = fig.add_subplot(111)
#ax.hist(entropyF,bins//2-3,label='Ergodic',density=False)
#ax.hist(entropyP,bins//2-3,label='Random',density=False)
#ax.hist(entropyM,bins,label='Structured',density=False)
#ax.set_xlabel('entropy_shannon')
#ax.set_ylabel('PDF')
#ax.legend(loc='upper center', shadow=True, handlelength=1.5)
#plt.savefig("entropy_shannon_long_traj.png",bbox_inches='tight')


fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
data = []
entropyM = np.zeros((5,100))
leng = [0, 1000, 2000,5000]
for k in range(5):
    for i in range(100):
        data = torch.load('./TRAIN/FULL_'+str(i)+'.pt')
        for j in range(9):
            for l in range(9):
                entropyM[k,i]+=(ent.shannon_entropy(data[3000*l*leng[k]:3000*(l+1),0,0])/9)
ax.hist(entropyM[k,:],bins,label=str(27000-leng[k]),density=False)
#print(np.mean(entropyM[k,:]))

ax.set_xlabel('entropy_shannon(FULL)')
ax.set_ylabel('PDF')
ax.legend(loc='upper right', shadow=True, handlelength=1.5)
plt.savefig("long_short_shannon_FULL_traj.png",bbox_inches='tight')

plt.show()
