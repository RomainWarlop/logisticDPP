import pandas as pd
import numpy as np
from logisticDPP.logisticMultiTaskDPP import logisticMultiTaskDPP
from copy import deepcopy 
import time
from ast import literal_eval

t0 = time.time()
# check computation time of new gradient

category = ["diaper"] #["diaper","apparel","feeding"]
if len(category)==1:
    #data = pd.read_csv("/home/romain/Documents/PhD/logisticDPP/amazon/"+category[0]+"_WithNegSampling.csv")
    data = pd.read_csv("/home/romain/Documents/PhD/logisticDPP/amazon/"+category[0]+"_withALSnegative2.csv")
else:
    data = pd.read_csv("/home/romain/Documents/PhD/logisticDPP/amazon/"+"_".join(category)+"_WithNegSampling.csv")

#data['itemSet'] = list(map(lambda x: list(map(int,x.split('-'))),data['itemSet']))
data['itemSet'] = list(map(literal_eval,data['itemSet']))

setName       = 'itemSet'
taskName      = 'target'
rewardName    = 'conversion'
numItems      = len(set([item for sublist in data['itemSet'].tolist() for item in sublist]))
numTasks      = numItems
numTraits     = 30
lbda          = 0.1
alpha         = 0.01
eps           = 0.001
betaMomentum  = 0.0#95
numIterFixed  = 300
minibatchSize = 5000
maxIter       = 700
gradient_cap  = 10000.0
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

    nCat = len(category)
    numTraits_cat = int(numTraits/nCat)
    nItem_cat = int(numItems/nCat)
    V0 = np.random.normal(scale=0.001,size=(numItems,numTraits))
    R0 = {}
    for i in range(len(category)):
        V0[nItem_cat*i:(nItem_cat*(i+1)),numTraits_cat*i:(numTraits_cat*(i+1))] *= 10
        for task in range(i*nItem_cat,(i+1)*nItem_cat):
            R0[task] = np.random.normal(loc=0.0,scale=0.01,size=numTraits)
            R0[task][numTraits_cat*i:(numTraits_cat*(i+1))] += 1.0
    
    mod.fit(trainingData,V0=V0,R0=R0,D0=np.zeros(numItems))
    
    if multitask:
        MPR_, P_ = mod.multitask_meanPercentileRank_Precision(testData,Ks)
    else:
        MPR_, P_ = mod.singletask_meanPercentileRank_Precision(testData,Ks)
    
    MPR.append(MPR_)
    for K in Ks:
        P[K].append(P_[K])

print("Mean Percentile Rank=",100*np.mean(MPR))
for K in Ks:
    print("Precision @"+str(K)+"=",100*np.mean(P[K]))

t1 = time.time()
print('total script time:',t1-t0)