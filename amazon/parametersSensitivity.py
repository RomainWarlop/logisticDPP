import pandas as pd
import numpy as np
from logisticDPP.logisticMultiTaskDPP import logisticMultiTaskDPP
from copy import deepcopy 

# check computation time of new gradient

category = ["diaper"] #["diaper","apparel","feeding"]
if len(category)==1:
    data = pd.read_csv("/home/romain/Documents/PhD/logisticDPP/amazon/"+category[0]+"_WithNegSampling.csv")
else:
    data = pd.read_csv("/home/romain/Documents/PhD/logisticDPP/amazon/"+"_".join(category)+"_WithNegSampling.csv")

data['itemSet'] = list(map(lambda x: list(map(int,x.split('-'))),data['itemSet']))

setName       = 'itemSet'
taskName      = 'target'
rewardName    = 'conversion'
numItems      = len(set([item for sublist in data['itemSet'].tolist() for item in sublist]))
numTasks      = numItems
numTraits     = 30
lbdas         = [1.0,0.1,0.01,0.001] 
alpha         = 0.01
eps           = 0.001
betaMomentum  = 0.0#95
numIterFixed  = 1800
minibatchSize = 5000
maxIter       = 2000
gradient_cap  = 10000.0
random_state  = 0
threshold     = 0.7

nRuns = 1

Ks = [5,10,20]

P = dict.fromkeys(lbdas)
MPR = dict.fromkeys(lbdas)

for lbda in lbdas:
    print('lambda=',lbda)
    MPR[lbda] = []
    P[lbda] = dict.fromkeys(Ks)
    for K in Ks:
        P[lbda][K] = []
    

    for run in range(nRuns):
        print("run number",run+1,"-",numTraits)
        np.random.seed(123*run)
        data['cv'] = np.random.random(len(data))
        
        trainingData = data.loc[data['cv']<threshold,]
        testData = data.loc[data['cv']>=threshold,]
        
        multitask = True
        if not(multitask):
            numTasks = 1
            for ind in trainingData.index:
                a = deepcopy(trainingData.loc[ind,'itemSet'])
                a = a+[trainingData.loc[ind,'target']]
                trainingData.at[ind,'itemSet'] = a
    
        mod = logisticMultiTaskDPP(setName=setName, taskName=taskName, 
                                   rewardName=rewardName, numItems=numItems, 
                                   numTasks=numTasks, numTraits=numTraits, 
                                   lbda=lbda, alpha=alpha, eps=eps, 
                                   betaMomentum=betaMomentum, numIterFixed=numIterFixed, 
                                   minibatchSize=minibatchSize, maxIter=maxIter, 
                                   gradient_cap=gradient_cap, random_state=random_state, 
                                   verbose=False)
        
        mod.fit(trainingData)
        MPR_, P_ = mod.meanPercentileRank_Precision(testData,Ks)
        
        MPR[lbda].append(MPR_)
        for K in Ks:
            P[lbda][K].append(P_[K])

    print("Mean Percentile Rank=",100*np.mean(MPR[lbda]))
    for K in Ks:
        print("Precision @"+str(K)+"=",100*np.mean(P[lbda][K]))
