import pandas as pd
import numpy as np
from logisticDPP.logisticMultiTaskDPP import logisticMultiTaskDPP
from ast import literal_eval
from copy import deepcopy

path = "/home/romain/Documents/PhD/logisticDPP/BelgiumRetail/"

# =============================================================================
# Load training/test data
# =============================================================================
trainingDataNegSampling = pd.read_csv(path+"trainingDataNegSampling.csv")
testDataWithTarget = pd.read_csv(path+"testDataWithTarget.csv")
trainingDataNegSampling['itemSet'] = list(map(literal_eval,trainingDataNegSampling['itemSet']))
testDataWithTarget['itemSet'] = list(map(literal_eval,testDataWithTarget['itemSet']))
testData = testDataWithTarget.loc[(testDataWithTarget['setSize']>1) & (testDataWithTarget['conversion']==1),]

testData['target2'] = -1
for ind in testData.index:
    testData.loc[ind,'target2'] = testData.loc[ind,'itemSet'][-1]
    testData.at[ind,'itemSet'] = testData.loc[ind,'itemSet'][:-1]

nItem = np.max([np.max(list(map(lambda x: np.max(x),trainingDataNegSampling['itemSet']))),
               np.max(list(map(lambda x: np.max(x),testDataWithTarget['itemSet'])))])+1

setName       = 'itemSet'
taskName      = 'target'
rewardName    = 'conversion'
numItems      = nItem
numTasks      = nItem
numTraits     = 75
lbda          = 0.1
alpha         = 0.0
eps           = 0.1
betaMomentum  = 0.1
numIterFixed  = 450
minibatchSize = 10000
maxIter       = 500
gradient_cap  = 1.0
random_state  = 0

mod = logisticMultiTaskDPP(setName, taskName, rewardName, numItems, numTasks,
                           numTraits, lbda, alpha, eps, betaMomentum, 
                           numIterFixed, minibatchSize, maxIter, gradient_cap, 
                           random_state)

mod.fit(trainingDataNegSampling)

np.savetxt(path+"V.txt",mod.V)
np.savetxt(path+"D.txt",mod.D)

R = np.zeros((numTasks,numTraits))
for task in range(numTasks):
    R[task,:] = mod.R[task]
np.savetxt(path+"R.txt",R)

print("End learning")
print("numTraits:",numTraits)
print("minibatchSize:",minibatchSize)
print("maxIter:",maxIter)

testData2 = deepcopy(testData)
del testData2['target']
testData2 = testData2.rename(columns={'target2':'target'})

Ks = [5,10,20]
percentileRank = []
P = dict.fromkeys(Ks,0)

for ind in testData.index:
    true_target = testData.loc[ind,mod.taskName]
    itemSet = testData.loc[ind,mod.setName]
    subV = mod.V[itemSet,:]
    subD = mod.D[itemSet]
    scores = list(map(lambda t: mod.targetPrediction(subV,subD,t),range(mod.numItems)))
    y0 = scores[true_target]
    rank = np.sum(scores<y0)
    percentileRank.append(rank/mod.numItems)
    
    new_item = np.argmax(scores)
    itemSet.append(new_item)
    testData2.at[ind,'itemSet'] = itemSet
    
    for K in Ks:
        topKTarget = np.argsort(scores)[-K:]
        if true_target in topKTarget:
            P[K] += 1

for K in Ks:
    P[K] /= len(testData)
MPR = np.mean(percentileRank)

MPR2, P2 = mod.meanPercentileRank_Precision(testData2,Ks)

print("Mean Percentile Rank=",100*np.mean(MPR))
for K in Ks:
    print("Precision @"+str(K)+"=",100*np.mean(P[K]))

print("Mean Percentile Rank=",100*np.mean(MPR2))
for K in Ks:
    print("Precision @"+str(K)+"=",100*np.mean(P2[K]))