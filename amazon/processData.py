import pandas as pd
import numpy as np

category = ["apparel"] #["diaper","apparel","feeding"]

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

## export for LowRankDPP
#concat = []
#for cat in category:
#    for ind in data[cat].index:
#        itemSet = list(map(lambda x: int(x)+deltaByCat[cat],data[cat].loc[ind,'itemSet'].split(',')))
#        itemSet = ','.join(list(map(str,itemSet)))
#        concat.append(itemSet)
#concat = pd.DataFrame({'itemSet':concat})
#concat.to_csv("/home/romain/Documents/PhD/LowRankDPP/Amazon/data/cross_cat.csv",index=False)

dataWithNegSampling = pd.DataFrame(columns=['itemSet','setSize','target','conversion'])
negSampling = 5

for cat in category:
    print(cat)
    for ind in data[cat].index:
        itemSet = list(map(lambda x: int(x)-1+deltaByCat[cat],data[cat].loc[ind,'itemSet'].split(',')))
        setSize = data[cat].loc[ind,'setSize']
        neg = np.random.choice(list(set(range(nItem))-set(itemSet)),negSampling,replace=False)
        for i in range(negSampling):
            dataWithNegSampling.loc[(negSampling+1)*ind+i] =  ['-'.join(list(map(str,itemSet))),setSize,neg[i],0]
        if len(itemSet)>1:
            pos = np.random.choice(list(itemSet),1,replace=False)[0]
            itemSet = list(set(itemSet)-set([pos]))
            dataWithNegSampling.loc[(negSampling+1)*ind+negSampling] =  ['-'.join(list(map(str,itemSet))),setSize-1,pos,1]

if len(category)==1:
    dataWithNegSampling.to_csv("/home/romain/Documents/PhD/logisticDPP/amazon/"+category[0]+"_WithNegSampling.csv",index=False)
else:
    dataWithNegSampling.to_csv("/home/romain/Documents/PhD/logisticDPP/amazon/"+"_".join(category)+"_WithNegSampling.csv",index=False)








