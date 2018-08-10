import pandas as pd
import numpy as np
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
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

def getNames(tab,ind):
    items = ['item_'+str(item) for item in tab.loc[ind,'itemSet']]
    
    target = "target_"+str(tab.loc[ind,'target'])
    items.append(target)
    return items

threshold   = 0.7
numTraits   = 50
nRuns       = 1
MPR         = []
Ks          = [5,10,20]
P           = dict.fromkeys(Ks)
for K in Ks:
    P[K] = []

run = 1
np.random.seed(123*run)
data['cv'] = np.random.random(len(data))

train = data.loc[data['cv']<threshold,['target','itemSet']]
test = data.loc[data['cv']>=threshold,['target','itemSet']]

y_train = data.loc[data['cv']<threshold,'conversion']
y_test = data.loc[data['cv']>=threshold,'conversion']

dictTrain = list(map(lambda ind: dict.fromkeys(getNames(train,ind),1),train.index))
dictTest = list(map(lambda ind: dict.fromkeys(getNames(test,ind),1),test.index))

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
data_test = pd.concat([test,y_test],axis=1)
data_test['setSize'] = list(map(len,data_test['itemSet']))
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
    
    subdata = []
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
