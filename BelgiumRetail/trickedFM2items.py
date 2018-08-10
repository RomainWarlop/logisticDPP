import pandas as pd
import numpy as np
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from ast import literal_eval
from copy import deepcopy

path = "/home/romain/Documents/PhD/logisticDPP/BelgiumRetail/"
trainingDataNegSampling = pd.read_csv(path+"trainingDataNegSampling.csv")
testDataWithTarget = pd.read_csv(path+"testDataWithTarget.csv")
trainingDataNegSampling['itemSet'] = list(map(literal_eval,trainingDataNegSampling['itemSet']))
testDataWithTarget['itemSet'] = list(map(literal_eval,testDataWithTarget['itemSet']))

nItem = np.max([np.max(list(map(lambda x: np.max(x),trainingDataNegSampling['itemSet']))),
               np.max(list(map(lambda x: np.max(x),testDataWithTarget['itemSet'])))])+1

y_train = trainingDataNegSampling['conversion']
y_test = testDataWithTarget['conversion']
del trainingDataNegSampling['conversion']
del testDataWithTarget['conversion']

def getNames(tab,ind):
    items = tab.loc[ind,'itemSet']
    items = ['item_'+str(item) for item in items]
    
    target = "target_"+str(tab.loc[ind,'target'])
    items.append(target)
    return items

dictTrain = list(map(lambda ind: dict.fromkeys(getNames(trainingDataNegSampling,ind),1),trainingDataNegSampling.index))
dictTest = list(map(lambda ind: dict.fromkeys(getNames(testDataWithTarget,ind),1),testDataWithTarget.index))

v = DictVectorizer()
X_train = v.fit_transform(dictTrain)
X_test = v.transform(dictTest)

# =============================================================================
# Factorization Machine
# =============================================================================
fm = pylibfm.FM(num_factors=20,num_iter=50,task="classification")
fm.fit(X_train,y_train)

# =========================================================================
# Compute MPR
# =========================================================================
data_test = pd.concat([testDataWithTarget,y_test],axis=1)
data_test = data_test.loc[(data_test['setSize']>1) & (data_test['conversion']==1),]

data_test['target2'] = -1
for ind in data_test.index:
    data_test.loc[ind,'target2'] = data_test.loc[ind,'itemSet'][-1]
    data_test.at[ind,'itemSet'] = data_test.loc[ind,'itemSet'][:-1]

testData2 = deepcopy(data_test)
del testData2['target']

percentileRank, percentileRank2 = [], []
precisionAt5    = 0
precisionAt10   = 0
precisionAt20   = 0
precisionAt5_2  = 0
precisionAt10_2 = 0
precisionAt20_2 = 0
for ind in data_test.index:
    subdata = data_test.loc[ind,]
    true_target = subdata['target']
    itemSet = subdata['itemSet']
    subdata = []
    for i in range(nItem):
        itemsName = ['item_'+str(item) for item in itemSet]
        target = "target_"+str(i)
        itemsName.append(target)
        subdata.append(itemsName)
    subdict = list(map(lambda x: dict.fromkeys(x,1),subdata))
    subX = v.transform(subdict)
    subY = fm.predict(subX)
    y0 = subY[true_target]
    rank = np.sum(subY>y0)
    percentileRank.append(1-rank/nItem)
    top5Target = np.argsort(subY)[-5:]
    top10Target = np.argsort(subY)[-10:]
    top20Target = np.argsort(subY)[-20:]
    if true_target in top5Target:
        precisionAt5 += 1
    if true_target in top10Target:
        precisionAt10 += 1
    if true_target in top20Target:
        precisionAt20 += 1
    
    new_item = np.argmax(subY)
    itemSet.append(new_item)
    testData2.at[ind,'itemSet'] = itemSet
    true_target2 = int(testData2.loc[ind,'target2'])
    
    for i in range(nItem):
        itemsName = ['item_'+str(item) for item in itemSet]
        target = "target_"+str(i)
        itemsName.append(target)
        subdata.append(itemsName)
    subdict = list(map(lambda x: dict.fromkeys(x,1),subdata))
    subX2 = v.transform(subdict)
    subY2 = fm.predict(subX2)
    subY2 = np.array(subY2)
    y02 = subY2[true_target2]
    rank2 = np.sum(subY2>y02)
    percentileRank2.append(1-rank2/nItem)
    top5Target = np.argsort(subY2)[-5:]
    top10Target = np.argsort(subY2)[-10:]
    top20Target = np.argsort(subY2)[-20:]
    if true_target2 in top5Target:
        precisionAt5_2 += 1
    if true_target2 in top10Target:
        precisionAt10_2 += 1
    if true_target2 in top20Target:
        precisionAt20_2 += 1

Ks = [5,10,20]
P  = dict.fromkeys(Ks)
P2 = dict.fromkeys(Ks)

MPR    = 100.*np.mean(percentileRank)
P[5]   = 100.*precisionAt5/len(data_test)
P[10]  = 100.*precisionAt10/len(data_test)
P[20]  = 100.*precisionAt20/len(data_test)

MPR2   = 100.*np.mean(percentileRank2)
P2[5]  = 100.*precisionAt5_2/len(data_test)
P2[10] = 100.*precisionAt10_2/len(data_test)
P2[20] = 100.*precisionAt20_2/len(data_test)

print("MPR=",np.mean(MPR))
for K in Ks:
    print("Precision @"+str(K)+"=",np.mean(P[K]))

print("2nd item - MPR=",np.mean(MPR2))
for K in Ks:
    print("2nd item - Precision @"+str(K)+"=",np.mean(P2[K]))
