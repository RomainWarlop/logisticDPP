import pandas as pd
path = '/home/romain/Documents/PhD/logisticDPP/UKRetailer/'

data = pd.read_csv(path+'PF_rankings_final10.tsv',sep='\t')
testData = pd.read_csv(path+'test.csv',sep='\t',header=None,
                       names=['user','item','rating'])

data = data[['user.id','item.id','pred','rank']]
data = pd.merge(data,testData,left_on='user.id',right_on='user')

nItem = len(set(data['item.id']))
MPR = 1-data.loc[data['item.id']==data['item'],'rank'].mean()/nItem

Ks = [5,10,20]
P = dict.fromkeys(Ks)
nUser = len(set(data['user.id']))
for K in Ks:
    subdata = data.loc[data['rank']<=K]
    subdata = subdata.loc[subdata['item.id']==subdata['item']]
    P[K] = len(subdata)/nUser*100