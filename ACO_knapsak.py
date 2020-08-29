# -*- coding: utf-8 -*-
import math
import numpy.matlib
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class create_model():
    def __init__(self,v_l=200,v_h=500,
                      w_l=20,w_h=100,
                      m_l=1, m_h=5, 
                      max_w = 1500,
                      size=20):
        self.v=np.random.randint(low=v_l, high=v_h, size=size)
        self.w=np.random.randint(low=w_l, high=w_h, size=size)
        self.m=np.random.randint(low=m_l, high=m_h, size=size)
        self.n=len(self.v)
        self.max_w = max_w        

class empty_sol:
    def __init__(self, V1=0, W1=0, V0=0, W0=0, 
                 Violation=0, z=0, IsFeasible=0):
        self.V1=V1
        self.W1=W1
        self.V0=V0
        self.W0=W0
        self.Violation=Violation
        self.z=z
        self.IsFeasible=IsFeasible
        
# Empty Ant
class empty_ant():
    def __init__(self, tour=[],
                 cost=0, sol=[]):
        self.tour=tour
        self.cost=cost
        self.sol=sol
                        
def cost(tour,model):
    m=model.m
    v=model.v
    w=model.w
    max_w=model.max_w
    
    V1=sum(v*tour);
    W1=sum(w*tour);
    V0=sum(v*(m-tour));
    W0=sum(w*(m-tour));
    
    Violation=max(W1/max_w-1,0);
    
    alpha=10000;
    z=V0+alpha*Violation;
    
    sol = empty_sol
    sol.V1=V1
    sol.W1=W1
    sol.V0=V0
    sol.W0=W0
    sol.Violation=Violation
    sol.z=z
    sol.IsFeasible=(Violation==0)
    
    return z, sol
  
def RouletteWheelSelection(P):
    r=random.random()
    C=np.cumsum(P)
    j=np.where(C>=r)[0][0]  #find(r<=C,1,'first');
   
    return j    

## Problem Definition

model=create_model()
nVar=model.n
m=model.m

## ACO Parameters

MaxIt=300       # Maximum Number of Iterations
nAnt=200         # Number of Ants (Population Size)
Q=1
tau0= 1 #10*Q/(nVar*np.mean(model.D)) # Initial Phromone

alpha=1         # Phromone Exponential Weight
beta=1          # Heuristic Exponential Weight

rho=0.5         # Evaporation Rate

## Initialization

tau= []         # Heuristic Information Matrix
for l in range(nVar):
    tau.append(tau0*np.ones((model.m[l]), dtype=int))

BestCost=np.zeros((MaxIt,2))     # Array to Hold Best Cost Values

# Empty Ant
ants = []

for i in range(nAnt):
    ants.append(empty_ant(tour=[], cost=0, sol=[]))

# Best Ant
BestAnt = empty_ant(tour=[], cost=np.inf, sol=[])

## ACO Main Loop
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(1,1,1)

for it in range(MaxIt):    
    # Move Ants
    for k in range(nAnt):
        ants[k].tour = []        
        for l in range(nVar): 
            t = ants[k]

            P=tau[l]**alpha
            
            P=P/np.sum(P)
            
            j=RouletteWheelSelection(P)
            
            ants[k].tour.append(j)
        
        ants[k].cost, ants[k].sol =cost(ants[k].tour, model)
        
        if ants[k].cost<=BestAnt.cost:
            BestAnt=ants[k]
    
    # Update Phromones
    for k in range(nAnt):
        
        tour=ants[k].tour
        
        for l in range(nVar):                                             
            tau[l][tour[l]]+=Q/ants[k].cost
        
    tour=BestAnt.tour        
    for l in range(nVar):                                             
        tau[l][tour[l]]+=Q/ants[k].cost
    
    # Evaporation
    for i in range(len(tau)):
        tau[i]=(1-rho)*tau[i]
    
    # Store Best Cost
    BestCost[it]=[it,BestAnt.cost]
    
    # Show Iteration Information
    print(f'Iteration {it} : Best Cost = {BestCost[it,1]} ')
   
## Results

labels = np.array(range(len(BestCost)), dtype = int)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(1,1,1)
plt.plot(BestCost[:,1]);
plt.xlabel('Iteration');
plt.ylabel('Best Cost');

plt.show()



































