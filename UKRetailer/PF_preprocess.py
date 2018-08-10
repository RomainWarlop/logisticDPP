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
df['rating'] = 1

def random_mask(x):
    result = np.zeros_like(x)
    if len(x)>1:
        result[np.random.choice(len(x))] = 1
    return result

threshold = 0.7
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

rnd = np.random.random(len(trainingData))
validationData = trainingData.loc[rnd>0.99]
trainingData = trainingData.loc[rnd<0.99]

path = '/home/romain/Documents/PhD/logisticDPP/UKRetailer/'
trainingData[['user','item','rating']].to_csv(path+'training.csv',sep='\t',header=False,index=False)
testData[['user','item','rating']].to_csv(path+'test.csv',sep='\t',header=False,index=False)
validationData[['user','item','rating']].to_csv(path+'validation.csv',sep='\t',header=False,index=False)
