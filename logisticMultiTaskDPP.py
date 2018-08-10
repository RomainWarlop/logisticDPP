# =============================================================================
# TO DO
# - regularization 
# - weighted regularization
# =============================================================================

import numpy as np
from random import shuffle
from collections import Counter

def sigma(x):
    if x<1e-5:
        return x
    else:
        return 1-np.exp(-x)

class logisticMultiTaskDPP(object):
    
    def __init__(self, setName, taskName, rewardName, numItems, numTasks,
                 numTraits=10, lbda=0.1, alpha=0.1, eps=0.1, betaMomentum=0, 
                 numIterFixed=50, minibatchSize=100, maxIter=500, gradient_cap=None, 
                 random_state=None, verbose=False):
        self.setName = setName
        self.taskName = taskName
        self.rewardName = rewardName
        self.numItems = numItems
        self.numTasks = numTasks
        self.numTraits = numTraits
        self.lbda = lbda
        self.alpha = alpha
        self.eps = eps
        self.eps0 = eps
        self.betaMomentum = betaMomentum
        self.numIterFixed = numIterFixed
        self.minibatchSize = minibatchSize
        self.maxIter = maxIter
        if gradient_cap is not None:
            self.gradient_cap = gradient_cap
        else:
            self.gradient_cap = 1.0
        self.seed = None
        self.it = 0
        if random_state is not None:
            self.seed = random_state
        self.verbose = verbose
        
        if (self.taskName is None) & (self.numTasks>1):
            print("ERROR: please give a task name for multi task learning")
        else:
            self.multitask = False
            if self.numTasks>1:
                self.multitask = True

# =============================================================================
#     fitting
# =============================================================================
    def fit_singletask(self,trainingData,testData=None,V0=None,D0=None):
        if V0 is not None:
            self.V = V0
        else:
            self.V = np.random.normal(scale=0.01,size=(self.numItems,self.numTraits))
        if D0 is not None:
            self.D = D0
        else:
            self.D = np.random.normal(loc=1.0,scale=0.01,size=self.numItems)
        
        index = list(trainingData.index)
        shuffle(index)
        
        if self.betaMomentum>0:
            V_momentum = np.zeros((self.numItems,self.numTraits))
            D_momentum = np.zeros(self.numItems)
        
        miniBatchStartIndex = 0
        while self.it<self.maxIter:
            if self.it%10==0:
                print("\n start iter",self.it)
                
                if self.it%50==0:
                    if testData is not None:
                        MPR, P = self.singletask_meanPercentileRank_Precision(testData,[5,10,20])
                        print("="*30)
                        print("Mean Percentile Rank:",int(10000*MPR)/100)
                        print("Precision @5:",int(10000*P[5])/100)
                        print("Precision @10:",int(10000*P[10])/100)
                        print("Precision @20:",int(10000.0*P[20])/100.0)
                        print("="*30)
                           
            self.it+=1
            if miniBatchStartIndex+self.minibatchSize>len(index):
                miniBatchIndex = index[miniBatchStartIndex:]
                miniBatchStartIndex = 0
                shuffle(index)
            else:
                miniBatchIndex = index[miniBatchStartIndex:miniBatchStartIndex+self.minibatchSize]
                miniBatchStartIndex += self.minibatchSize
                
            miniBatchData = trainingData.loc[miniBatchIndex]
            
            if self.betaMomentum>0:
                gradients = self.computeGradient_singletask(self.V+self.betaMomentum*V_momentum, 
                                                           self.D+self.betaMomentum*D_momentum, 
                                                           miniBatchData)
                
                V_gradient, D_gradient = gradients
                V_gradient = np.clip(V_gradient,-self.gradient_cap,self.gradient_cap)
                D_gradient = np.clip(D_gradient,-self.gradient_cap,self.gradient_cap)
                
                V_momentum *= self.betaMomentum
                V_momentum += (1-self.betaMomentum)*self.eps*V_gradient
                self.V += V_momentum
                D_momentum *= self.betaMomentum
                D_momentum += (1-self.betaMomentum)*self.eps*D_gradient
                self.D += D_momentum
            else:
                gradients = self.computeGradient_singletask(self.V, self.D, miniBatchData)
            
            V_gradient, D_gradient = gradients
            
            if self.verbose:
                print("max V gradient =",abs(V_gradient).max())
                print("max D gradient =",abs(D_gradient).max())
            
            V_gradient = np.clip(V_gradient,-self.gradient_cap,self.gradient_cap)
            D_gradient = np.clip(D_gradient,-self.gradient_cap,self.gradient_cap)
            
            
            self.V += self.eps*V_gradient
            self.D += self.eps*D_gradient
            
            if self.it >= self.numIterFixed:
                self.eps = self.eps0 / (1 + self.it/ self.numIterFixed)
                print("Reduced eps:",self.eps)
        
        if testData is not None:
            MPR, P = self.singletask_meanPercentileRank_Precision(testData,[5,10,20])
            print("="*30)
            print("Mean Percentile Rank:",int(10000*MPR)/100)
            print("Precision @ 5:",int(10000*P[5])/100)
            print("Precision @ 10:",int(10000*P[10])/100)
            print("Precision @ 20:",int(10000.0*P[20])/100.0)
            print("="*30)

    def computeGradient_singletask(self,V,D,data):
        V_gradient = np.zeros((self.numItems,self.numTraits))
        D_gradient = np.zeros(self.numItems)
        
        itemsInData = list(set([item for sublist in data[self.setName] for item in sublist]))
        for index in data.index:
            y = data.loc[index,self.rewardName]
            itemSet = data.loc[index,self.setName]
            
            subV = self.V[itemSet,:]
            subD = self.D[itemSet]
            subK = subV.dot(subV.T)+np.diag(subD**2)
            subK_inv = np.linalg.inv(subK)
            det_m = np.linalg.det(subK)
            sigma_m = sigma(self.lbda*det_m)
            
            if y==0:
                delta_y_sigma = -1
            else:
                delta_y_sigma = (y-sigma_m)/(sigma_m)
            
            for item in itemSet:
                # compute gradient on D[item]
                i0 = itemSet.index(item)
                subK_inv_itemitem = subK_inv[i0,i0]
                out = subK_inv_itemitem*self.D[item]*delta_y_sigma*det_m
                D_gradient[item] += 2*self.lbda*out
                
                for k in range(self.numTraits):
                    # compute gradient on V[item,k]
                    subK_inv_item = subK_inv[i0,:]
                    out = subK_inv_item.dot(subV[:,k])*delta_y_sigma*det_m
                    V_gradient[item,k] += 2*self.lbda*out
            
        for item in itemsInData:
            V_gradient[item,:] -= self.alpha*(1/self.itemsWeight[item])*V[item,:]
            D_gradient[item] -= self.alpha*(1/self.itemsWeight[item])*D[item]
            
        return V_gradient, D_gradient

    def fit_multitask(self,trainingData,testData=None,V0=None,D0=None,R0=None):
        if V0 is not None:
            self.V = V0
        else:
            self.V = np.random.normal(scale=0.01,size=(self.numItems,self.numTraits))
        if R0 is not None:
            self.R = R0
        else:
            self.R = {}
            for task in range(self.numTasks):
                self.R[task] = np.random.normal(loc=1.0,scale=0.01,size=self.numTraits)
        if D0 is not None:
            self.D = D0
        else:
            self.D = np.random.normal(loc=1.0,scale=0.01,size=self.numItems)
        
        index = list(trainingData.index)
        shuffle(index)
        
        if self.betaMomentum>0:
            V_momentum = np.zeros((self.numItems,self.numTraits))
            D_momentum = np.zeros(self.numItems)
            R_momentum = {}
            for task in range(self.numTasks):
                R_momentum[task] = np.zeros(self.numTraits)
        
        miniBatchStartIndex = 0
        while self.it<self.maxIter:

            if self.it%1==0:
                print("\n start iter",self.it)
                
                if (self.it%10==0) & (self.it>0):
                    if testData is not None:
                        MPR, P = self.multitask_meanPercentileRank_Precision(testData,[5,10,20])
                        print("="*30)
                        print("Mean Percentile Rank:",int(10000.*MPR)/100.)
                        print("Precision @ 5:",int(10000*P[5])/100.)
                        print("Precision @ 10:",int(10000*P[10])/100)
                        print("Precision @ 20:",int(10000.0*P[20])/100.0)
                        print("="*30)
            
            self.it+=1
            if miniBatchStartIndex+self.minibatchSize>len(index):
                miniBatchIndex = index[miniBatchStartIndex:]
                miniBatchStartIndex = 0
                shuffle(index)
            else:
                miniBatchIndex = index[miniBatchStartIndex:miniBatchStartIndex+self.minibatchSize]
                miniBatchStartIndex += self.minibatchSize
                
            miniBatchData = trainingData.loc[miniBatchIndex]
            
            if self.betaMomentum>0:
                R_wMomentum = {}
                for task in range(self.numTasks):
                    R_wMomentum[task] = self.R[task]+self.betaMomentum*R_momentum[task]
                gradients = self.computeGradient_multitask(self.V+self.betaMomentum*V_momentum, 
                                                           self.D+self.betaMomentum*D_momentum, 
                                                           R_wMomentum, miniBatchData)
                V_gradient, D_gradient, R_gradient = gradients
                V_gradient = np.clip(V_gradient,-self.gradient_cap,self.gradient_cap)
                D_gradient = np.clip(D_gradient,-self.gradient_cap,self.gradient_cap)
                
                V_momentum *= self.betaMomentum
                V_momentum += (1-self.betaMomentum)*self.eps*V_gradient
                self.V += V_momentum
                D_momentum *= self.betaMomentum
                D_momentum += (1-self.betaMomentum)*self.eps*D_gradient
                self.D += D_momentum
                
                for task in range(self.numTasks):
                    R_gradient[task] = np.clip(R_gradient[task],-self.gradient_cap,self.gradient_cap)
                    R_momentum[task] *= self.betaMomentum
                    R_momentum[task] += (1-self.betaMomentum)*self.eps*R_gradient[task]
                    self.R[task] += R_momentum[task]
            else:
                gradients = self.computeGradient_multitask(self.V, self.D, self.R, miniBatchData)
            
                V_gradient, D_gradient, R_gradient = gradients
                
                if self.verbose:
                    print("max V gradient =",abs(V_gradient).max())
                    print("max D gradient =",abs(D_gradient).max())
                    print("max R gradient =",np.max(list(map(lambda x: abs(R_gradient[x]),range(self.numTasks)))))
                    
                V_gradient = np.clip(V_gradient,-self.gradient_cap,self.gradient_cap)
                D_gradient = np.clip(D_gradient,-self.gradient_cap,self.gradient_cap)
                
                self.V += self.eps*V_gradient
                self.D += self.eps*D_gradient
                
                for task in range(self.numTasks):
                    R_gradient[task] = np.clip(R_gradient[task],-self.gradient_cap,self.gradient_cap)
                    self.R[task] += self.eps*R_gradient[task]
                        
            if self.it >= self.numIterFixed:
                self.eps = self.eps0 / (1 + self.it/ self.numIterFixed)
                print("Reduced eps:",self.eps)
        
        if testData is not None:
            MPR, P = self.multitask_meanPercentileRank_Precision(testData,[5,10,20])
            print("")
            print("="*16,"FINAL PERF","="*16)
            print("Mean Percentile Rank:",int(10000.0*MPR)/100.0)
            print("Precision @ 5:",int(10000.0*P[5])/100.0)
            print("Precision @ 10:",int(10000.0*P[10])/100.0)
            print("Precision @ 20:",int(10000.0*P[20])/100.0)
            print("="*44)
    
    def computeGradient_multitask(self,V,D,R,data):
        V_gradient = np.zeros((self.numItems,self.numTraits))
        D_gradient = np.zeros(self.numItems)
        R_gradient = {}
        for task in range(self.numTasks):
            R_gradient[task] = np.zeros(self.numTraits)
        
        itemsInData = list(set([item for sublist in data[self.setName] for item in sublist]))
        taskInData = list(set(data[self.taskName]))
        
        for index in data.index:
            y = data.loc[index,self.rewardName]
            itemSet = data.loc[index,self.setName]
            task = data.loc[index,self.taskName]
            
            subV = self.V[itemSet,:]
            subD = self.D[itemSet]
            subK = subV.dot(np.diag(self.R[task]**2)).dot(subV.T)+np.diag(subD**2)
            try:
                subK_inv = np.linalg.inv(subK)
            except:
                print(itemSet,'-',task)
            det_m = np.linalg.det(subK)
            sigma_m = sigma(self.lbda*det_m)
            
            if y==0:
                delta_y_sigma = -1
            else:
                delta_y_sigma = (y-sigma_m)/(sigma_m)
            
            for item in itemSet:
                # compute gradient on D[item]
                i0 = itemSet.index(item)
                subK_inv_itemitem = subK_inv[i0,i0]
                out = subK_inv_itemitem*self.D[item]*delta_y_sigma*det_m
                D_gradient[item] += 2*self.lbda*out
                
                subK_inv_item = subK_inv[i0,:]
                for k in range(self.numTraits):
                    # compute gradient on V[item,k]
                    out = self.R[task][k]**2*subK_inv_item.dot(subV[:,k])*delta_y_sigma*det_m
                    V_gradient[item,k] += 2*self.lbda*out
                
                    # compute gradient on R[task][k]
                    tr = self.R[task][k]*subK_inv.dot(subV[:,k]).dot(subV[:,k])
                    out = tr*delta_y_sigma*det_m
                    R_gradient[task][k] += 2*self.lbda*out
        
        for item in itemsInData:
            V_gradient[item,:] -= self.alpha*(1/self.itemsWeight[item])*V[item,:]
            D_gradient[item] -= self.alpha*(1/self.itemsWeight[item])*D[item]
            
        for task in taskInData:
            R_gradient[task] -= self.alpha*(1/self.taskWeight[task])*R[task]
        
        return V_gradient, D_gradient, R_gradient
    
    def fit(self,trainingData,testData=None,V0=None,D0=None,R0=None):
        # compute regularization weights for items
        items = [item for sublist in trainingData[self.setName] for item in sublist]
        self.itemsWeight = Counter(items)
        
        if self.multitask:
            # compute regularization weights for tasks
            self.taskWeight = trainingData[self.taskName].value_counts().to_dict()
            self.fit_multitask(trainingData,testData=testData,V0=V0,D0=D0,R0=R0)
        else:
            self.fit_singletask(trainingData,testData=testData,V0=V0,D0=D0)
    
# =============================================================================
#     prediction
# =============================================================================
    def multitask_targetPrediction(self,subV,subD,target):
        return np.linalg.det(subV.dot(np.diag(self.R[target]**2)).dot(subV.T)+np.diag(subD**2))

    def singletask_targetPrediction(self,itemSet,target):
        itemSetTarget = itemSet+[target]
        subV = self.V[itemSetTarget,:]
        subD = self.D[itemSetTarget]
        return np.linalg.det(subV.dot(subV.T)+np.diag(subD**2))
    
    def multitask_meanPercentileRank_Precision(self,data,Ks):
        conversionData = data.loc[data[self.rewardName]==1,]
        percentileRank = []
        precision = dict.fromkeys(Ks,0)
        
        for ind in conversionData.index:
            true_target = conversionData.loc[ind,self.taskName]
            itemSet = conversionData.loc[ind,self.setName]
            subV = self.V[itemSet,:]
            subD = self.D[itemSet]
            scores = list(map(lambda t: self.multitask_targetPrediction(subV,subD,t),range(self.numItems)))
            y0 = scores[true_target]
            rank = np.sum(scores<y0)
            percentileRank.append(rank/(self.numItems-len(itemSet)))
            
            for K in Ks:
                topKTarget = np.argsort(scores)[-K:]
                if true_target in topKTarget:
                    precision[K] += 1
        
        for K in Ks:
            precision[K] /= len(conversionData)
            
        return np.mean(percentileRank), precision
    
    def multitask_meanPercentileRank_Precision_multipleCompletion(self,data,Ks,nProduct):
        conversionData = data.loc[data[self.rewardName]==1,]
        percentileRank, precision = {}, {}
        for n in range(nProduct):
            percentileRank[n] = []
            precision[n] = dict.fromkeys(Ks,0)
        
        for ind in conversionData.index:
            itemSet = conversionData.loc[ind,self.setName]
            for n in range(nProduct):
                true_target = conversionData.loc[ind,self.taskName+str(n+1)]
                subV = self.V[itemSet,:]
                subD = self.D[itemSet]
                scores = list(map(lambda t: self.multitask_targetPrediction(subV,subD,t),range(self.numItems)))
                y0 = scores[true_target]
                rank = np.sum(scores<y0)
                percentileRank[n].append(rank/self.numItems)
                
                new_item = np.argmax(scores)
                itemSet.append(new_item)
                for K in Ks:
                    topKTarget = np.argsort(scores)[-K:]
                    if true_target in topKTarget:
                        precision[n][K] += 1
        
        for n in range(nProduct):
            for K in Ks:
                precision[n][K] /= len(conversionData)
        
        MPR = {}
        for n in range(nProduct):
            MPR[n] = np.mean(percentileRank[n])
        
        return MPR, precision
    
    def singletask_meanPercentileRank_Precision(self,data,Ks):
        conversionData = data.loc[data[self.rewardName]==1,]
        percentileRank = []
        precision = dict.fromkeys(Ks,0)
        
        for ind in conversionData.index:
            true_target = conversionData.loc[ind,self.taskName]
            itemSet = conversionData.loc[ind,self.setName]
            scores = list(map(lambda t: self.singletask_targetPrediction(itemSet,t),range(self.numItems)))
            scores = np.array(scores)
            y0 = scores[true_target]
            rank = np.sum(scores<y0)
            percentileRank.append(rank/self.numItems)
            
            for K in Ks:
                topKTarget = np.argsort(scores)[-K:]
                if true_target in topKTarget:
                    precision[K] += 1
        
        for K in Ks:
            precision[K] /= len(conversionData)
            
        return np.mean(percentileRank), precision
    
    def meanPercentileRank_Precision(self,data,Ks):
        if self.multitask:
            return self.multitask_meanPercentileRank_Precision(data,Ks)
        else:
            return self.singletask_meanPercentileRank_Precision(data,Ks)
    
#    def roc_auc(self,data):
#        y_hat = []
#        for ind in data.index:
#            itemSet = data.loc[ind,self.setName]
#            subV = self.V[itemSet,:]
#            subD = self.D[itemSet]
#            subK = subV.dot(subV.T)+np.diag(subD**2)
#            y_hat.append(np.linalg.det(subK))
#
#        out = roc_auc_score(data[self.rewardName],y_hat)
#        return out

