import pandas as pd
import numpy as np
from copy import deepcopy

path = "/home/romain/Documents/PhD/logisticDPP/Instacart/data/"

data = pd.read_csv(path+"order_products__train.csv")
#prior = pd.read_csv(path+"order_products__prior.csv")
#data = pd.concat([train,prior])
#del train
#del prior
data = data[['order_id','product_id','add_to_cart_order']]
data = data.groupby(['order_id','product_id']).first().reset_index()
sizes = data.groupby('order_id')['product_id'].count().reset_index()
sizes = sizes.rename(columns={'product_id':'size'})
data = pd.merge(data,sizes)
data = data.loc[data['size']>2]

itemFreq = data.groupby('product_id')['order_id'].count().reset_index()
itemFreq = itemFreq.rename(columns={'order_id':'freq'})
data = pd.merge(data,itemFreq)
data = data.loc[data['freq']>15]

nItem = len(set(data['product_id']))
itemPivot = pd.DataFrame({'product_id':list(set(data['product_id'])),
                          'product_num':range(nItem)})

data = pd.merge(data,itemPivot)
data = data[['order_id','add_to_cart_order','product_num']]
data = data.sort_values(by=['order_id','add_to_cart_order'])

# =============================================================================
# RNN
# =============================================================================
data_rnn = deepcopy(data)
data_rnn = data_rnn[['order_id','product_num','add_to_cart_order']]
data_rnn = data_rnn.rename(columns={'order_id':'SessionId',
                                    'product_num':'ItemId',
                                    'add_to_cart_order':'Timestamps'})
data_rnn['cv'] = np.random.random(len(data))
train_rnn = data_rnn.loc[data_rnn['cv']<0.7,['SessionId','ItemId','Timestamps']]
test_rnn = data_rnn.loc[data_rnn['cv']>=0.7,['SessionId','ItemId','Timestamps']]

train_rnn.to_csv("/home/romain/Documents/PhD/RNN/data/instacart_train.csv",
                index=False)
test_rnn.to_csv("/home/romain/Documents/PhD/RNN/data/instacart_test.csv",
                index=False)

# =============================================================================
# Low Rank DPP
# =============================================================================
sets = deepcopy(data)
sets['product_num'] += 1
sets = sets.groupby('order_id')['product_num'].apply(lambda x: list(set(x))).reset_index()

sets = list(map(lambda x: ",".join(list(map(str,x))),sets['product_num']))
sets = pd.DataFrame({'sets':sets})

sets.to_csv("/home/romain/Documents/PhD/LowRankDPP/Instacart/data/data.csv",
            index=False)
# =============================================================================
# Log DPP
# =============================================================================

data = data.groupby(['order_id'])['product_num'].apply(list).reset_index()
data = data.rename(columns={'product_num':'itemSet'})

data['itemSet'] = list(map(lambda x: list(set(x)),data['itemSet']))
data['setSize'] = list(map(len,data['itemSet']))

dataPos = data.loc[data['setSize']>1]

def deleteOne(D):
    target = D['itemSet'][-1]
    itemSet = D['itemSet'][:-1]
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

data.to_csv(path+"baskets.csv",index=False)
