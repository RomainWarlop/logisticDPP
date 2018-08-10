import pandas as pd
import numpy as np
from logisticDPP.logisticMultiTaskDPP import logisticMultiTaskDPP
from copy import deepcopy

np.random.seed(0)

data = pd.read_csv("/home/romain/Documents/PhD/logisticDPP/UKRetailer/data.csv",encoding="latin1")

nItem = len(set(data['StockCode']))
pivotTable = pd.DataFrame({'StockCode':list(set(data['StockCode'])),
                           'itemid':list(range(nItem))})

data = pd.merge(data,pivotTable)
data = data.groupby(['CustomerID','InvoiceDate'])['itemid'].apply(list).reset_index()
del data['CustomerID']
del data['InvoiceDate']
data = data.rename(columns={'itemid':'itemSet'})

data['itemSet'] = list(map(lambda x: list(set(x)),data['itemSet']))
data['setSize'] = list(map(len,data['itemSet']))

data = data.loc[data['setSize']<100,]

dataPos = data.loc[data['setSize']>1]

def deleteOne(D):
    target = np.random.choice(D['itemSet'],1)[0]
    itemSet = list(set(D['itemSet'])-set([target]))
    return {'itemSet':itemSet,'target':target}

def addOne(D,K):
    out = []
    targets = np.random.choice(list(set(range(nItem))-set(D['itemSet'])),K,replace=False)
    for target in targets:
        out.append({'itemSet':D['itemSet'],'target':target})
    return out

dataPos = pd.DataFrame(list(map(lambda x: deleteOne(dataPos.loc[x].to_dict()),dataPos.index)))
dataPos['conversion'] = 1

negativeSampling = 5
dataNeg = list(map(lambda x: addOne(data.loc[x].to_dict(),negativeSampling),data.index))
dataNeg = pd.DataFrame(list([item for sublist in dataNeg for item in sublist]))
dataNeg['conversion'] = 0

data = pd.concat([dataPos,dataNeg],ignore_index=True)

setName       = 'itemSet'
taskName      = 'target'
rewardName    = 'conversion'
numItems      = nItem
numTasks      = numItems
numTraits     = 100
lbda          = 0.01 # 0.1 -> plafond à 6 
alpha         = 0.1 # 0.1 mieux ? 
eps           = 0.001 # 0.01 -> NA
betaMomentum  = 0.0#1 passe à 0 après 150 itérations
numIterFixed  = 1800
minibatchSize = 5000 # check 10000
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
    testData_sample = testData.loc[testData.index[:500]]
    
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
    mod.fit(trainingData,testData_sample)#,D0=D0) 
    
    MPR_, P_ = mod.meanPercentileRank_Precision(testData,Ks)
    MPR.append(MPR_)
    for K in Ks:
        P[K].append(P_[K])

print("Mean Percentile Rank=",100*np.mean(MPR))
for K in Ks:
    print("Precision @"+str(K)+"=",100*np.mean(P[K]))