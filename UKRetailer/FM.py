import pandas as pd
import numpy as np
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer

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
data = data.loc[data['setSize']>1]

sets = list(map(lambda x: list(map(lambda y: int(y),x)),data['itemSet']))
size = list(map(len,sets))
nUsers = len(sets)
users = np.repeat(range(nUsers),size)

flatsets = [str(int(item)) for sublist in sets for item in sublist]
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

negative    = False
threshold   = 0.7
numTraits   = 50
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
        trainingUsers = pd.DataFrame({'user':trainingUsers,'key':1})
        negativeSampling = pd.merge(trainingUsers,itemPop)
        negativeSampling = pd.merge(negativeSampling,trainingData,
                                    on=['user','item'],how='left')
        negativeSampling = negativeSampling.fillna(0)
        negativeSampling = negativeSampling.loc[negativeSampling['rating']==0]
        negativeSampling = negativeSampling.groupby(['user']).head(n=5).reset_index()
        negativeSampling = negativeSampling[['user','item','rating']]
        trainingData = pd.concat([trainingData,negativeSampling])
    
    trainingData['user'] = list(map(lambda x: str(int(x)),trainingData['user']))
    testData['user'] = list(map(lambda x: str(int(x)),testData['user']))
    
    v = DictVectorizer()
    X_train = v.fit_transform(trainingData.to_dict(orient='records'))
    y_train = np.repeat(1.0,X_train.shape[0])
    
    fm = pylibfm.FM(num_factors=numTraits,num_iter=50,task="classification")
    fm.fit(X_train,y_train)

    percentileRank = []
    precisionAt5 = 0
    precisionAt10 = 0
    precisionAt20 = 0
    
    def userPerf(ind):
        prec = dict.fromkeys(Ks,0)
        user = testData.loc[ind,'user']
        true_target = int(testData.loc[ind,'item'])
        subdata = []
        for item in range(nItem):
            subdata.append({'user':user,'item':str(int(item))})
        subX = v.transform(subdata)
        subY = fm.predict(subX)
        y0 = subY[true_target]
        rank = np.sum(subY>y0)
        PR = 1-rank/nItem
        for K in Ks:
            topTarget = np.argsort(subY)[-K:]
            if true_target in topTarget:
                prec[K] = 1
        return PR, prec
    
    perfs = list(map(userPerf,testData.index))
    precision = dict.fromkeys(Ks,0)
    for i in range(len(perfs)):
        percentileRank.append(perfs[i][0])
        for K in Ks:
            precision[K] += perfs[i][1][K]
    
    MPR.append(100*np.mean(percentileRank))
    for K in Ks:
        P[K].append(100*precision[K]/len(testData))

print("MPR=",np.mean(MPR))
for K in Ks:
    print("Precision @"+str(K)+"=",np.mean(P[K]))