import pandas as pd
import numpy as np
from copy import deepcopy
from logisticDPP.logisticMultiTaskDPP import logisticMultiTaskDPP

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
numTraits     = 50
lbda          = 0.1
alpha         = 0.1
eps           = 0.1
betaMomentum  = 0.0#1
numIterFixed  = 150
minibatchSize = 10000
maxIter       = 200
gradient_cap  = 1.0
random_state  = 0
threshold     = 0.7

nRuns = 1
MPR, MPR2 = [], []
Ks = [5,10,20]
P = dict.fromkeys(Ks)
P2 = dict.fromkeys(Ks)
for K in Ks:
    P[K] = []
    P2[K] = []

for run in range(nRuns):
    print("run number",run+1,"-",numTraits)
    np.random.seed(123*run)
    data['cv'] = np.random.random(len(data))
    
    trainingData = data.loc[data['cv']<threshold,]
    testData = data.loc[data['cv']>=threshold,]
    testData = testData.loc[(testData['setSize']>1) & (testData['conversion']==1),]
    
    testData['target2'] = -1
    for ind in testData.index:
        testData.loc[ind,'target2'] = testData.loc[ind,'itemSet'][-1]
        testData.at[ind,'itemSet'] = testData.loc[ind,'itemSet'][:-1]
    
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
    
    mod.fit(trainingData,V0=V0,R0=R0)
    
    mod.multitask_twoItemsCompletionPerformance(testData,Ks)
    
    MPR_, P_ = mod.meanPercentileRank_Precision(testData,Ks)
    MPR.append(MPR_)
    for K in Ks:
        P[K].append(P_[K])
    
    # add first prediction to the itemSet
    testData2 = deepcopy(testData)
    del testData2['target']
    testData2 = testData2.rename(columns={'target2':'target'})
    for ind in testData2.index:
        itemSet = testData2.loc[ind,mod.setName]
        subV = mod.V[itemSet,:]
        subD = mod.D[itemSet]
        scores = list(map(lambda t: mod.multitask_targetPrediction(subV,subD,t),range(mod.numItems)))
        new_item = np.argmax(scores)
        itemSet.append(new_item)
        testData2.at[ind,'itemSet'] = itemSet
    
    MPR_, P_ = mod.meanPercentileRank_Precision(testData2,Ks)
    MPR2.append(MPR_)
    for K in Ks:
        P2[K].append(P_[K])

print("Mean Percentile Rank=",100*np.mean(MPR))
for K in Ks:
    print("Precision @"+str(K)+"=",100*np.mean(P[K]))

print("2nd item - Mean Percentile Rank=",100*np.mean(MPR2))
for K in Ks:
    print("2nd item - Precision @"+str(K)+"=",100*np.mean(P2[K]))