import pandas as pd
import numpy as np
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from copy import deepcopy

category = ["diaper","apparel","feeding"]
if len(category)==1:
    data = pd.read_csv("/home/romain/Documents/PhD/logisticDPP/amazon/"+category[0]+"_WithNegSampling.csv")
else:
    data = pd.read_csv("/home/romain/Documents/PhD/logisticDPP/amazon/"+"_".join(category)+"_WithNegSampling.csv")

#data = data.loc[data['setSize']>1,]
#roc_auc_score(data['conversion'],data['setSize'])

def getNames(tab,ind):
    items = tab.loc[ind,'itemSet'].split('-')
    items = ['item_'+item for item in items]
    
    target = "target_"+str(tab.loc[ind,'target'])
    items.append(target)
    return items

threshold   = 0.7
numTraits   = 50
nRuns       = 3
MPR, MPR2   = [], []
Ks          = [5,10,20]
P           = dict.fromkeys(Ks)
P2          = dict.fromkeys(Ks)

for K in Ks:
    P[K] = []
    P2[K] = []

for run in range(nRuns):
    print("run number",run)
    np.random.seed(123*run)
    data['cv'] = np.random.random(len(data))

    train = data.loc[data['cv']<threshold,['target','itemSet']]
    test = data.loc[data['cv']>=threshold,['target','itemSet','setSize']]
    
    y_train = data.loc[data['cv']<threshold,'conversion']
    y_test = data.loc[data['cv']>=threshold,'conversion']
    
    dictTrain = list(map(lambda ind: dict.fromkeys(getNames(train,ind),1),train.index))
    dictTest = list(map(lambda ind: dict.fromkeys(getNames(test,ind),1),test.index))
    
    v = DictVectorizer()
    X_train = v.fit_transform(dictTrain)
    X_test = v.transform(dictTest)
    
    # =========================================================================
    # Factorization Machine
    # =========================================================================
    fm = pylibfm.FM(num_factors=numTraits,num_iter=100,task="classification")
    fm.fit(X_train,y_train)
    
    # =========================================================================
    # Compute MPR
    # =========================================================================
    data_test = pd.concat([test,y_test],axis=1)
    data_test = data_test.loc[(data_test['setSize']>1) & (data_test['conversion']==1),]
    data_test['itemSet'] = list(map(lambda x: list(map(int,x.split('-'))),data_test['itemSet']))

    data_test['target2'] = -1
    for ind in data_test.index:
        data_test.loc[ind,'target2'] = data_test.loc[ind,'itemSet'][-1]
        data_test.at[ind,'itemSet'] = data_test.loc[ind,'itemSet'][:-1]
    
    testData2 = deepcopy(data_test)
    del testData2['target']
    
    nItem = 100*len(category)
    percentileRank, percentileRank2 = [], []
    precisionAt5 = 0
    precisionAt10 = 0
    precisionAt20 = 0
    precisionAt5_2 = 0
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
        
    MPR.append(100*np.mean(percentileRank))
    P[5].append(100*precisionAt5/len(data_test))
    P[10].append(100*precisionAt10/len(data_test))
    P[20].append(100*precisionAt20/len(data_test))

    MPR2.append(100*np.mean(percentileRank2))
    P2[5].append(100*precisionAt5_2/len(data_test))
    P2[10].append(100*precisionAt10_2/len(data_test))
    P2[20].append(100*precisionAt20_2/len(data_test))

print("MPR=",np.mean(MPR))
for K in Ks:
    print("Precision @"+str(K)+"=",np.mean(P[K]))

print("2nd item - MPR=",np.mean(MPR2))
for K in Ks:
    print("2nd item - Precision @"+str(K)+"=",np.mean(P2[K]))
