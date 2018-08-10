import pandas as pd
import numpy as np
from logisticDPP.logisticMultiTaskDPP import logisticMultiTaskDPP
from copy import deepcopy
from ast import literal_eval
np.random.seed(0)

data = pd.read_csv("/home/romain/Documents/PhD/logisticDPP/Instacart/data/baskets.csv")
data['itemSet'] = list(map(literal_eval,data['itemSet']))
nItem = 10531

setName       = 'itemSet'
taskName      = 'target'
rewardName    = 'conversion'
numItems      = nItem
numTasks      = numItems
numTraits     = 80
lbda          = 0.01 # 0.1 -> plafond à 6 
alpha         = 0.1 # 0.1 mieux ? 
eps           = 0.001 # 0.01 -> NA
betaMomentum  = 0.0#1 passe à 0 après 150 itérations
numIterFixed  = 1800
minibatchSize = 1000 # check 10000
maxIter       = 2000 #250
gradient_cap  = 1000.0
random_state  = 0
threshold     = 0.7

nRuns = 1
MPR = []
Ks = [5,10,20]
P = dict.fromkeys(Ks)
for K in Ks:
    P[K] = []

for run in range(nRuns):
    print("run number",run+1,"-",numTraits)
    np.random.seed(123*run)
    data['cv'] = np.random.random(len(data))
    
    trainingData = data.loc[data['cv']<threshold,]
    testData = data.loc[data['cv']>=threshold,]
    #testData_sample = testData.loc[testData.index[:500]]
    
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
    
    #D0 = np.zeros(numItems)
    #D0 = np.random.normal(loc=0.1,scale=0.01,size=numItems)
    mod.fit(trainingData,testData)#,D0=D0) 
    
    MPR_, P_ = mod.meanPercentileRank_Precision(testData,Ks)
    MPR.append(MPR_)
    for K in Ks:
        P[K].append(P_[K])

print("Mean Percentile Rank=",100*np.mean(MPR))
for K in Ks:
    print("Precision @"+str(K)+"=",100*np.mean(P[K]))
