import pandas as pd
import numpy as np
from logisticDPP.logisticMultiTaskDPP import logisticMultiTaskDPP
from ast import literal_eval
from copy import deepcopy

path = "/home/romain/PhD/logisticDPP/BelgiumRetail/"

# =============================================================================
# Load training/test data
# =============================================================================
trainingDataNegSampling = pd.read_csv(path+"trainingDataNegSampling.csv")
testDataWithTarget = pd.read_csv(path+"testDataWithTarget.csv")
trainingDataNegSampling['itemSet'] = list(map(literal_eval,trainingDataNegSampling['itemSet']))
testDataWithTarget['itemSet'] = list(map(literal_eval,testDataWithTarget['itemSet']))

testData_sample = testDataWithTarget.loc[testDataWithTarget.index[:1000]]

nItem = np.max([np.max(list(map(lambda x: np.max(x),trainingDataNegSampling['itemSet']))),
               np.max(list(map(lambda x: np.max(x),testDataWithTarget['itemSet'])))])+1

setName       = 'itemSet'
taskName      = 'target'
rewardName    = 'conversion'
numItems      = nItem
numTasks      = nItem
numTraits     = 75
lbda          = 0.01
alpha         = 0.0
eps           = 0.001
betaMomentum  = 0.1
numIterFixed  = 450
minibatchSize = 10000
maxIter       = 500
gradient_cap  = 1.0
random_state  = 0

print("lbda=",lbda)
print("alpha=",alpha)
print("eps=",eps)
print("betaMomentum=",betaMomentum)
print("numIterFixed=",numIterFixed)
print("maxIter=",maxIter)
print("minibatchSize=",minibatchSize)

multitask = False
print("multitask=",multitask)
if not(multitask):
    numTasks = 1
    for ind in trainingDataNegSampling.index:
        a = deepcopy(trainingDataNegSampling.loc[ind,'itemSet'])
        a = a+[trainingDataNegSampling.loc[ind,'target']]
        trainingDataNegSampling.at[ind,'itemSet'] = a

mod = logisticMultiTaskDPP(setName, taskName, rewardName, numItems, numTasks,
                           numTraits, lbda, alpha, eps, betaMomentum, 
                           numIterFixed, minibatchSize, maxIter, gradient_cap, 
                           random_state)

mod.fit(trainingDataNegSampling,testData_sample)

#np.savetxt(path+"V.txt",mod.V)
#np.savetxt(path+"D.txt",mod.D)
#
#R = np.zeros((numTasks,numTraits))
#for task in range(numTasks):
#    R[task,:] = mod.R[task]
#np.savetxt(path+"R.txt",R)

print("End learning")
print("numTraits:",numTraits)
print("minibatchSize:",minibatchSize)
print("maxIter:",maxIter)
Ks = [5,10,20]
MPR, P = mod.meanPercentileRank_Precision(testDataWithTarget,Ks)

print("Mean Percentile Rank=",100*np.mean(MPR))
for K in Ks:
    print("Precision @"+str(K)+"=",100*np.mean(P[K]))

#np.savetxt(path+"V_"+str(numTraits)+".txt",mod.V)
#np.savetxt(path+"D_"+str(numTraits)+".txt",mod.D)
#
#R = np.zeros((numTasks,numTraits))
#for task in range(numTasks):
#    R[task,:] = mod.R[task]
#
#np.savetxt(path+"R_"+str(numTraits)+".txt",R)