# =============================================================================
# TO DO
# - regularization 
# - weighted regularization
# =============================================================================

# =============================================================================
# Problem definition 
# Each player can play at different positif. Let i be the player, p the position
# tilde_V_ip = V_i diag(R_p)
# tilde_V_ip latent factors for this player at this position
# V_i latent factors of this player
# R_i latent factors of this position
# =============================================================================

import pandas as pd
import numpy as np
from random import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score
from copy import deepcopy

def sigma(x):
    if x<1e-5:
        return x
    else:
        return 1-np.exp(-x)

class logisticAdversarialMultiTaskDPP(object):
    
    def __init__(self, teamA, taskA, teamB, taskB, rewardName, numPlayers, numTasks,
                 numTraits=10, lbda=0.1, alpha=0.1, eps=0.1, betaMomentum=0, 
                 reverse=False, numIterFixed=50, minibatchSize=100, maxIter=500,
                 random_state=None, verbose=False):
        self.teamA = teamA
        self.taskA = taskA
        self.teamB = teamB
        self.taskB = taskB
        self.rewardName = rewardName
        self.numPlayers = numPlayers
        self.numTasks = numTasks
        self.numTraits = numTraits
        self.lbda = lbda
        self.alpha = alpha
        self.eps = eps
        self.eps0 = eps
        self.betaMomentum = betaMomentum
        self.reverse = reverse
        self.numIterFixed = numIterFixed
        self.minibatchSize = minibatchSize
        self.maxIter = maxIter
        self.seed = None
        self.it = 0
        if random_state is not None:
            self.seed = random_state
        self.verbose = verbose
        
        if (self.taskA is None) & (self.numTasks>1):
            print("ERROR: please give a task name for multi task learning")
    
    def fit_singletask(self,trainingData,testData=None,V0=None,D0=None):
        if V0 is not None:
            self.V = V0
        else:
            self.V = np.random.normal(scale=0.01,size=(self.numItems,self.numTraits))
        if V0 is not None:
            self.D = D0
        else:
            self.D = np.random.normal(loc=1.0,scale=0.01,size=self.numItems)
        
        index = list(trainingData.index)
        shuffle(index)
        
        miniBatchStartIndex = 0
        while self.it<self.maxIter:
            if self.it%10==0:
                print("\n start iter",self.it)
                
                if testData is not None:
                    AUC = self.roc_auc(testData)
                    print("="*30)
                    print("ROC AUC:",int(100*AUC)/100)
                    print("="*30)
                           
            self.it+=1
            if miniBatchStartIndex+self.minibatchSize>len(index):
                miniBatchIndex = index[miniBatchStartIndex:]
                miniBatchStartIndex = 0
                shuffle(index)
            else:
                miniBatchIndex = index[miniBatchStartIndex:miniBatchStartIndex+self.minibatchSize]
                miniBatchStartIndex += self.minibatchSize
                
            miniBacthData = trainingData.loc[miniBatchIndex]
            
            gradients = self.computeGradient_singletask(self.V, self.D, miniBacthData)
            
            V_gradient, D_gradient = gradients
            V_gradient = np.clip(V_gradient,-1,1)
            D_gradient = np.clip(D_gradient,-1,1)
            
            if self.verbose:
                print("max V gradient =",abs(V_gradient).max())
                print("max D gradient =",abs(D_gradient).max())
            
            self.V += self.eps*V_gradient
            self.D += self.eps*D_gradient
            
            if self.it >= self.numIterFixed:
                self.eps = self.eps0 / (1 + self.it/ self.numIterFixed)
                print("Reduced eps:",self.eps)
        
        if testData is not None:
            AUC = self.roc_auc(testData)
            print("")
            print("="*16,"FINAL PERF","="*16)
            print("ROC AUC:",int(1000*AUC)/1000)
            print("="*44)

    def computeGradient_singletask(self,V,D,data):
        V_gradient = np.zeros((self.numItems,self.numTraits))
        D_gradient = np.zeros(self.numItems)
                
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

        return V_gradient, D_gradient

    def fit_multitask(self,trainingData,testData=None,V0=None,D0=None,R0=None):
        if V0 is not None:
            self.V = V0
        else:
            self.V = np.random.normal(scale=0.01,size=(self.numPlayers,self.numTraits))
        if R0 is not None:
            self.R = R0
        else:
            self.R = {}
            for task in range(self.numTasks):
                self.R[task] = np.random.normal(loc=1.0,scale=0.01,size=self.numTraits)
        if V0 is not None:
            self.D = D0
        else:
            self.D = np.random.normal(loc=1.0,scale=0.01,size=self.numPlayers)
        
        index = list(trainingData.index)
        shuffle(index)
        
        miniBatchStartIndex = 0
        while self.it<self.maxIter:
            if self.it%10==0:
                print("\n start iter",self.it)

                if testData is not None:
                    pred = self.winner(testData)
                    pred['diff'] = pred['score_'+self.teamA]-pred['score_'+self.teamB]
                    AUC = roc_auc_score(testData[self.rewardName],pred['diff'])
                    print("="*30)
                    print("AUC:",int(100*AUC)/100)
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
            
            gradients = self.computeGradient_multitask(self.V, self.D, self.R, miniBatchData)
            
            V_gradient, D_gradient, R_gradient = gradients
            V_gradient = np.clip(V_gradient,-1,1)
            D_gradient = np.clip(D_gradient,-1,1)
            for task in range(self.numTasks):
                R_gradient[task] = np.clip(R_gradient[task],-1,1)
            
            if self.verbose:
                print("max V gradient =",abs(V_gradient).max())
                print("max D gradient =",abs(D_gradient).max())
                print("max R gradient =",np.max(list(map(lambda x: abs(R_gradient[x]),range(self.numTasks)))))
            
            self.V += self.eps*V_gradient
            self.D += self.eps*D_gradient
            for task in range(self.numTasks):
                self.R[task] += self.eps*R_gradient[task]
            
            if self.it >= self.numIterFixed:
                self.eps = self.eps0 / (1 + self.it/ self.numIterFixed)
                print("Reduced eps:",self.eps)
        
        if testData is not None:
            pred = self.winner(testData)
            pred['diff'] = pred['score_'+self.teamA]-pred['score_'+self.teamB]
            AUC = roc_auc_score(testData[self.rewardName],pred['diff'])
            pred['win_hat'] = list(map(lambda x: 1 if x>0 else 0,pred['diff']))
            print("")
            print("="*16,"FINAL PERF","="*16)
            print("AUC:",int(100*AUC)/100)
            print("Accuracy:",accuracy_score(testData[self.rewardName],pred['win_hat']))
            print("="*44)
    
    def computeGradient_multitask(self,V,D,R,data):
        V_gradient = np.zeros((self.numPlayers,self.numTraits))
        D_gradient = np.zeros(self.numPlayers)
        R_gradient = {}
        for task in range(self.numTasks):
            R_gradient[task] = np.zeros(self.numTraits)
        
        for index in data.index:
            y = data.loc[index,self.rewardName]
            teamA = data.loc[index,self.teamA]
            taskA = data.loc[index,self.taskA]
            teamB = data.loc[index,self.teamB]
            taskB = data.loc[index,self.taskB]
            
            subV_A = self.V[teamA,:]
            subVtilde_A = deepcopy(subV_A)
            for player in range(len(teamA)):
                subVtilde_A[player,:] = subV_A[player,:].dot(np.diag(self.R[taskA[player]]))
            subD_A = self.D[teamA]
            
            subK_A = subVtilde_A.dot(subVtilde_A.T)+np.diag(subD_A**2)
            subK_A_inv = np.linalg.inv(subK_A)
            detA_m = np.linalg.det(subK_A)
            
            subV_B = self.V[teamB,:]
            subVtilde_B = deepcopy(subV_B)
            for player in range(len(teamB)):
                subVtilde_A[player,:] = subV_B[player,:].dot(np.diag(self.R[taskB[player]]))
            subD_B = self.D[teamB]
            
            subK_B = subVtilde_B.dot(subVtilde_B.T)+np.diag(subD_B**2)
            subK_B_inv = np.linalg.inv(subK_B)
            detB_m = np.linalg.det(subK_B)
            
            sigma_m = sigma(self.lbda*(detA_m-detB_m))
            
            if y==0:
                delta_y_sigma = -1
            else:
                delta_y_sigma = (y-sigma_m)/(sigma_m)
            
            for player in teamA:
                # compute gradient on D[player]
                i0 = teamA.index(player)
                subKA_inv_itemitem = subK_A_inv[i0,i0]
                out = subKA_inv_itemitem*self.D[player]*delta_y_sigma*detA_m
                D_gradient[player] += 2*self.lbda*out
                
                subKA_inv_item = subK_A_inv[i0,:]
                for k in range(self.numTraits):
                    # compute gradient on V[player,k]
                    out = self.R[taskA[i0]][k]*subKA_inv_item.dot(subVtilde_A[:,k])*delta_y_sigma*detA_m
                    V_gradient[player,k] += 2*self.lbda*out
                
                    # compute gradient on R[user][k]
                    tr = subK_A_inv.dot(subV_A[:,k]).dot(subVtilde_A[:,k])
                    out = tr*delta_y_sigma*detA_m
                    R_gradient[task][k] += 2*self.lbda*out
            
            for player in teamB:
                # compute gradient on D[player]
                i0 = teamB.index(player)
                subKB_inv_itemitem = subK_B_inv[i0,i0]
                out = subKB_inv_itemitem*self.D[player]*delta_y_sigma*detB_m
                D_gradient[player] += 2*self.lbda*out
                
                subKB_inv_item = subK_B_inv[i0,:]
                for k in range(self.numTraits):
                    # compute gradient on V[player,k]
                    out = self.R[taskB[i0]][k]*subKB_inv_item.dot(subVtilde_B[:,k])*delta_y_sigma*detB_m
                    V_gradient[player,k] += 2*self.lbda*out
                
                    # compute gradient on R[user][k]
                    tr = subK_B_inv.dot(subV_B[:,k]).dot(subVtilde_B[:,k])
                    out = tr*delta_y_sigma*detB_m
                    R_gradient[task][k] += 2*self.lbda*out
        
        return V_gradient, D_gradient, R_gradient
    
    def fit(self,trainingData,testData=None,V0=None,D0=None,R0=None):
        if self.reverse:
            if self.verbose:
                print("Append to the training data the interchanged of team A and B")
            reverseTraining = trainingData.copy()
            reverseTraining[self.teamA] = trainingData[self.teamB]
            reverseTraining[self.teamB] = trainingData[self.teamA]
            reverseTraining[self.taskA] = trainingData[self.taskB]
            reverseTraining[self.taskB] = trainingData[self.taskA]
            reverseTraining[self.rewardName] = 1-reverseTraining[self.rewardName]
            trainingData = pd.concat([trainingData,reverseTraining],ignore_index=True)
        
        if self.numTasks>1:
            self.fit_multitask(trainingData,testData=testData,V0=V0,D0=D0,R0=R0)
        else:
            self.fit_singletask(trainingData,testData=testData,V0=V0,D0=D0)
    
    def winner(self,data):
        out = pd.DataFrame(columns=['score_'+self.teamA,'score_'+self.teamB])
        for index in data.index:
            teamA = data.loc[index,self.teamA]
            taskA = data.loc[index,self.taskA]
            teamB = data.loc[index,self.teamB]
            taskB = data.loc[index,self.taskB]
            
            subV_A = self.V[teamA,:]
            subVtilde_A = deepcopy(subV_A)
            for player in range(len(teamA)):
                subVtilde_A[player,:] = subV_A[player,:].dot(np.diag(self.R[taskA[player]]))
            subD_A = self.D[teamA]
            
            subK_A = subVtilde_A.dot(subVtilde_A.T)+np.diag(subD_A**2)
            detA = np.linalg.det(subK_A)
            
            subV_B = self.V[teamB,:]
            subVtilde_B = deepcopy(subV_B)
            for player in range(len(teamB)):
                subVtilde_B[player,:] = subV_B[player,:].dot(np.diag(self.R[taskB[player]]))
            subD_B = self.D[teamB]
            
            subK_B = subVtilde_B.dot(subVtilde_B.T)+np.diag(subD_B**2)
            detB = np.linalg.det(subK_B)
            
            out.loc[index,'score_'+self.teamA] = detA
            out.loc[index,'score_'+self.teamB] = detB
        
        return out


