import pandas as pd
import numpy as np
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer

data = pd.read_csv("/home/romain/Documents/PhD/logisticDPP/BelgiumRetail/data",
                   header=None,names=['itemSet'])

sets = list(map(lambda x: x.split(' ')[:-1],data['itemSet']))
sets = list(map(lambda x: list(map(lambda y: int(y),x)),sets))
size = list(map(len,sets))
nUsers = len(sets)
users = np.repeat(range(nUsers),size)

flatsets = [str(int(item)) for sublist in sets for item in sublist]
df = pd.DataFrame({'user':users,'item':flatsets})
df.index = df['user']
nItem = len(set(df['item']))

def random_mask(x):
    result = np.zeros_like(x)
    if len(x)>1:
        result[np.random.choice(len(x))] = 1
    return result

threshold   = 0.7
numTraits   = 30
nRuns       = 1
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
    
    trainingData['user'] = list(map(lambda x: str(int(x)),trainingData['user']))
    testData['user'] = list(map(lambda x: str(int(x)),testData['user']))
    
    v = DictVectorizer()
    X_train = v.fit_transform(trainingData.to_dict(orient='records'))
    y_train = np.repeat(1.0,X_train.shape[0])
    
    fm = pylibfm.FM(num_factors=numTraits,num_iter=150,task="classification")
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
        for i in range(nItem):
            subdata.append({'user':user,'item':str(int(i))})
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
    
#numTraits = 30
#MPR= 63.00228915125711
#Precision @5= 17.644291091593477
#Precision @10= 18.502979924717692
#Precision @20= 19.200909661229613