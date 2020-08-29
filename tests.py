# -*- coding: utf-8 -*-
import math
import numpy.matlib
import pandas as pd
import numpy as np
import random

class model():
    def __init__(self,l=20,h=100,size=20):
        self.x=np.random.randint(low=l, high=h, size=size)
        self.y=np.random.randint(low=l, high=h, size=size)
        self.n=len(self.x);
        self.D=np.zeros((self.n,self.n));
        for i in range(self.n):
            for j in range(i+1,self.n):
                self.D[i,j]=math.sqrt((self.x[i]-self.x[j])**2+(self.y[i]-self.y[j])**2);
                self.D[j,i]=self.D[i,j]
                
def cost(tour,model):
    n=len(tour)
    tour.append(tour[0])
    l=0
    for i in range(n):
        l+=model.D[tour[i],tour[i+1]];
    
    return l       

# Empty Ant
class empty_ant():
    def __init__(self, tour=[], cost=[]):
        self.tour=tour
        self.cost=cost

def RouletteWheelSelection(P):
    r=random.random()
    C=np.cumsum(P)
    
    j=np.where(C>=r)[0][0]  #find(r<=C,1,'first');

    return j     

## Problem Definition

model=model(l=20,h=100,size=20)
nVar=model.n

## ACO Parameters

MaxIt=300       # Maximum Number of Iterations
nAnt=40         # Number of Ants (Population Size)
Q=1
tau0=1          # 10*Q/(nVar*np.mean(model.D));	% Initial Phromone

alpha=1         # Phromone Exponential Weight
beta=1          # Heuristic Exponential Weight

rho=0.05        # Evaporation Rate


## Initialization

eta=1/model.D                   # Heuristic Information Matrix

tau=tau0*np.ones((nVar,nVar))    # Phromone Matrix

BestCost=np.zeros((MaxIt,1))    # Array to Hold Best Cost Values

# Ant Colony Matrix
ant=np.empty((nAnt,), dtype=object)
ant[:] = empty_ant(tour=[], cost=[])

# Best Ant
BestAnt = empty_ant(tour=[], cost=np.inf)

## ACO Main Loop
BestCost = []
for it in range(MaxIt):    
    # Move Ants
    for k in range(1,nAnt):
        ant[k].tour.append(np.random.randint(low=0, high=nVar, size=1)[0])
        
        for l in range(1,nVar): 
            t = ant[k]
            i=ant[k].tour[-1]
                 
            P=tau[i,:]**alpha*eta[i,:]**beta
            
            P[ant[k].tour]=0
            
            P=P/np.sum(P)
            
            j=RouletteWheelSelection(P)
            
            ant[k].tour.append(j)
        
        ant[k].cost=cost(ant[k].tour, model)
        
        if ant[k].cost<BestAnt.cost:
            BestAnt=ant[k]
    
    # Update Phromones
    for k in range(nAnt):
        
        tour=ant[k].tour
        
        ### tour.append(tour(1))
        
        for l in range(nVar-1):            
            i=tour[l]
            j=tour[l+1]
            
            tau[i,j]+=Q/ant[k].cost
    
    # Evaporation
    tau=(1-rho)*tau
    
    # Store Best Cost
    BestCost.append([it,BestAnt.cost])
    
    # Show Iteration Information
    print(f'Iteration {it} : Best Cost = {BestCost[it]} ')
    
    # Plot Solution
    # figure(1);
    # PlotSolution(BestAnt.Tour,model);
    
## Results

# figure;
# plot(BestCost,'LineWidth',2);
# xlabel('Iteration');
# ylabel('Best Cost');

























    
