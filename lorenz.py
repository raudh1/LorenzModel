#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:03:19 2018

@author: Alessandro
"""

import numpy as np
import os
import random
from scipy.integrate import solve_ivp

dirName = './DATA'
if not os.path.exists(dirName):
    os.mkdir(dirName)
# Lorenz paramters and initial conditions
u0, v0, w0 = 0, 1, 1.05

# Maximum time point and total number of time points
sigma, beta, rho = 10, 2.667, 28

def lorenz(t, X):
    """The Lorenz equations."""
    F = np.zeros(3)
    sigma, beta, rho = 10, 2.667, 28
    u, v, w = X[0], X[1], X[2]
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    F[0], F[1], F[2] = up, vp, wp
    return F

# Integrate the Lorenz equations on the time grid t
X = np.array([u0, v0, w0])

u1 = (beta*(rho-1))**(1/2)
v1 = (beta*(rho-1))**(1/2)
w1 = (rho-1)

u2 = -(beta*(rho-1))**(1/2)
v2 = -(beta*(rho-1))**(1/2)
w2 = (rho-1)

X0 = np.array([0, 0, 0])
X1 = np.array([u1, v1, w1])
X2 = np.array([u2, v2, w2])

XE = np.array([[X0], [X1], [X2]])

tmax, n = 30, 3000
t = np.linspace(0, tmax*9, n*9)
X = np.array([u0, v0, w0])
f = solve_ivp(lorenz, [0,tmax*9], X, method='RK45',t_eval=t)
F = f.y

# k was 100 
for k in range(100):
    print('ITER: '+str(k))
    ''' full dynamic'''
    seed = np.random.randint(F.shape[1],size=10)
    tmax, n = 30, 3000
    t = np.linspace(0, tmax*9, n*9)
    X = F[:,seed[9]]+np.random.rand(3)*1e-4
    
    f = solve_ivp(lorenz, [0,tmax*9], X, method='RK45',t_eval=t)
    F = f.y
    np.save('./DATA/FULL_'+str(k),F)
    print('    Full data: DONE')
    
    ''' partial dynamics'''
    t = np.linspace(0, tmax, n)
    sol = []
    for i in range(9):
        f = solve_ivp(lorenz, [0,tmax], F[:,seed[i]]+np.random.rand(3)*1e-4, method='RK45',t_eval=t)
        sol.append(f.y)
    np.save('./DATA/PART_'+str(k),np.asarray(sol))
    print('    Partial data: DONE')
    
    ''' manifold dynamics'''
    
    eps = np.random.rand(3,3)
    Q,R = np.linalg.qr(eps)
    Q=Q*1
    t = np.linspace(0, tmax, n)
    sol = []
    for j in range(3):
        for i in range(3):
            f = solve_ivp(lorenz, [0,tmax], np.squeeze(XE[j,:],axis=0)+Q[:,i], method='RK45',t_eval=t)
            sol.append(f.y)
    np.save('./DATA/MANI_'+str(k),np.asarray(sol))
    print('    Manifold data: DONE')
