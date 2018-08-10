import pandas as pd
import numpy as np
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from ast import literal_eval

path = "/home/romain/PhD/logisticDPP/BelgiumRetail/"
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

# =============================================================================
# Compute MPR
# =============================================================================
data_test = pd.concat([testDataWithTarget,y_test],axis=1)
data_test = data_test.loc[data_test['conversion']==1,]

percentileRank = []
precisionAt5 = 0
precisionAt10 = 0
precisionAt20 = 0
for ind in data_test.index:
    subdata = data_test.loc[ind,]
    true_target = subdata['target']
    itemSet = subdata['itemSet']
    #itemSet = itemSet.split('-')
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

print("MPR=",100*np.mean(percentileRank))
print("Precision @5=",100*precisionAt5/len(data_test))
print("Precision @10=",100*precisionAt10/len(data_test))
print("Precision @20=",100*precisionAt20/len(data_test))














