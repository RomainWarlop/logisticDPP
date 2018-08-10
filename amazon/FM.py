import pandas as pd
import numpy as np
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer

category = ["diaper"]#["diaper","apparel","feeding"]

data = {}
nItemByCat = {}
deltaByCat = {}
nItem = 0
for cat in category:
    data[cat] = pd.read_csv("/home/romain/Documents/PhD/logisticDPP/amazon/1_100_100_100_"+cat+"_regs.csv",
                   sep=";",header=None,names=['itemSet'])

    data[cat]['setSize'] = list(map(lambda x: len(x.split(',')),data[cat]['itemSet']))
    items = list(map(lambda x: list(map(int,x.split(','))),data[cat]['itemSet']))
    items = list(set([item for sublist in items for item in sublist]))
    nItemByCat[cat] = len(items)
    deltaByCat[cat] = nItem
    nItem += len(items)

sets = []
for cat in category:
    tmp = list(map(lambda x: x.split(','),data[cat]['itemSet']))
    tmp = list(map(lambda x: list(map(lambda y: int(y)+deltaByCat[cat],x)),tmp))
    sets.extend(tmp)
size = list(map(len,sets))
nUsers = len(sets)
users = np.repeat(range(nUsers),size)

flatsets = [str(int(item)-1) for sublist in sets for item in sublist]
df = pd.DataFrame({'user':users,'item':flatsets})
df.index = df['user']
df['rating'] = 1.0

itemPop = df.groupby(['item'])['user'].count().sort_values(ascending=False)[:50].reset_index()
itemPop = itemPop.rename(columns={'user':'freq'})
itemPop['key'] = 1

def random_mask(x):
    result = np.zeros_like(x)
    if len(x)>1:
        result[np.random.choice(len(x))] = 1
    return result

negative    = True
threshold   = 0.7
numTraits   = 5
nRuns       = 3
MPR         = []
Ks          = [5,10,20]
P           = dict.fromkeys(Ks)
for K in Ks:
    P[K] = []
    
for run in range(nRuns):
    print("run number",run+1,"-",numTraits)
    np.random.seed(123*run)
    
    testUsers = list(np.random.choice(range(nUsers),size=int((1-threshold)*nUsers),replace=False))
    trainingUsers = list(set(range(nUsers))-set(testUsers))
    trainingData = df.loc[trainingUsers]
    trainingData.index = range(len(trainingData))
    
    testData = df.loc[testUsers]
    testData.index = range(len(testData))
    
    mask = testData.groupby(['user'])['user'].transform(random_mask).astype(bool)
    not_mask = list(map(lambda x: not(x),mask))
    
    trainingData = pd.concat([trainingData,testData.loc[not_mask]])
    testData = testData.loc[mask]
    
    if negative:
        allUsers = pd.DataFrame({'user':list(range(nUsers)),'key':1})
        negativeSampling = pd.merge(allUsers,itemPop)
        negativeSampling = pd.merge(negativeSampling,trainingData,
                                    on=['user','item'],how='left')
        negativeSampling = negativeSampling.fillna(0)
        negativeSampling = negativeSampling.loc[negativeSampling['rating']==0]
        negativeSampling = negativeSampling.groupby(['user']).head(n=2).reset_index()
        negativeSampling = negativeSampling[['user','item','rating']]
        trainingData = pd.concat([trainingData,negativeSampling])
        
    trainingData['user'] = list(map(lambda x: str(int(x)),trainingData['user']))
    testData['user'] = list(map(lambda x: str(int(x)),testData['user']))
    
    v = DictVectorizer()
    X_train = v.fit_transform(trainingData[['user','item']].to_dict(orient='records'))
    y_train = trainingData['rating']
    
    fm = pylibfm.FM(num_factors=numTraits,num_iter=50,task="classification")
    fm.fit(X_train,y_train)

    percentileRank = []
    precisionAt5 = 0
    precisionAt10 = 0
    precisionAt20 = 0
    for ind in testData.index:
        user = testData.loc[ind,'user']
        true_target = int(testData.loc[ind,'item'])
        subdata = []
        for i in range(nItem):
            subdata.append({'user':user,'item':str(int(i))})
        subX = v.transform(subdata)
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

    MPR.append(100*np.mean(percentileRank))
    P[5].append(100*precisionAt5/len(testData))
    P[10].append(100*precisionAt10/len(testData))
    P[20].append(100*precisionAt20/len(testData))

print("MPR=",np.mean(MPR))
for K in Ks:
    print("Precision @"+str(K)+"=",np.mean(P[K]))


