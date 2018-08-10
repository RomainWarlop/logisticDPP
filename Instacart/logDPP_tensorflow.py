import pandas as pd
import numpy as np
from copy import deepcopy 
import tensorflow as tf 
from random import shuffle
import time
from ast import literal_eval

t0 = time.time()

path = "/home/romain/Documents/PhD/logisticDPP/Instacart/"
data = pd.read_csv("/home/romain/Documents/PhD/logisticDPP/Instacart/data/baskets.csv")
data['itemSet'] = list(map(literal_eval,data['itemSet']))
data['setSize'] = list(map(len,data['itemSet']))

nItem = np.max([np.max(list(map(lambda x: np.max(x),data['itemSet']))),
               np.max(list(map(lambda x: np.max(x),data['itemSet'])))])+1

# Parameters
setName         = 'itemSet'
taskName        = 'target'
rewardName      = 'conversion'
numItems        = nItem
numTasks        = nItem
numTraits       = 80
lbda            = 0.01
alpha           = 0.0#1
init_learning_rate   = 0.0001
minibatchSize   = 10000
training_epochs = 700
random_state    = 0
threshold       = 0.7

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
              init_learning_rate,                # Base learning rate.
              global_step,  # Current index into the dataset.
              100,          # Decay step.
              0.95,                # Decay rate.
              staircase=True)

nRuns = 1
MPR = np.zeros(nRuns)
Ks = [5,10,20]
P = dict.fromkeys(Ks)
for K in Ks:
    P[K] = np.zeros(nRuns)

#===============================================================================
# CREATE MODEL
#===============================================================================
def unnormalized_predict(itemSet,t):
    subV = tf.gather(weights['V'],itemSet)
    subD = tf.matrix_diag(tf.square(tf.gather(weights['D'],itemSet)))
    subR = tf.matrix_diag(tf.square(tf.gather(weights['R'],t)))
    K1 = tf.matmul(subV,subR)
    K = tf.matmul(K1,tf.transpose(subV,perm=[0,2,1]))
    K = tf.add(K,subD)
    eps = tf.eye(tf.shape(K)[1],tf.shape(K)[1],[tf.shape(K)[0]])*1e-3
    K = tf.add(K,eps)
    res = tf.matrix_determinant(K)
    return res

def logsigma(itemSet,t):
    res = 1-tf.exp(-lbda*unnormalized_predict(itemSet,t))
    return tf.stack([res,1-res],axis=1)

def regularization(itemSet,t):
    itemsInBatch, _ = tf.unique(tf.reshape(itemSet,[-1]))
    targetsInBatch, _ = tf.unique(tf.reshape(t,[-1]))
    subV = tf.gather(weights['V'],itemsInBatch)
    subD = tf.gather(weights['D'],itemsInBatch)
    subR = tf.gather(weights['R'],targetsInBatch)
    subV_norm = tf.reduce_sum(tf.norm(subV,axis=1))
    subD_norm = tf.norm(subD)
    subR_norm = tf.reduce_sum(tf.norm(subR,axis=1))
    return subV_norm+subD_norm+subR_norm

def predict(itemSet,true_target):
    precision = []
    itemSets = tf.reshape(tf.tile(itemSet,[numItems]),[numItems,tf.size(itemSet)])
    #itemSets = [itemSet for i in range(numItems)]
    targets = list(range(numItems))
    scores = unnormalized_predict(itemSets,targets)
    _, indices = tf.nn.top_k(scores, k=numItems, sorted=True)
    percentile_rank = tf.where(tf.equal(indices,true_target))
    for K in Ks:
        _, indices = tf.nn.top_k(scores, k=K, sorted=True)
        out, idx = tf.setdiff1d(indices,true_target)
        precision.append(K-tf.shape(out))
    return precision, percentile_rank

#===============================================================================
# TF GRAPH INPUT
#===============================================================================
X = tf.placeholder(tf.int32, [None,None])
target = tf.placeholder(tf.int32, [None])
Y = tf.placeholder(tf.float32, [None,2])
itemSet_ = tf.placeholder(tf.int32, [None])
true_target = tf.placeholder(tf.int32, [1])

intermediatePerf = True
for run in range(nRuns):
    print("run number",run+1,"/",nRuns)
    np.random.seed(123*run)
    data['cv'] = np.random.random(len(data))
    data[rewardName] = list(map(float,data[rewardName]))

    trainingData = data.loc[data['cv']<threshold,]
    testData = data.loc[data['cv']>=threshold,]
    testData = testData.loc[testData[rewardName]==1,]
    train_size = len(trainingData)

    setSizes = list(set(trainingData['setSize']))
    X_train = dict.fromkeys(setSizes)
    target_train = dict.fromkeys(setSizes)
    Y_train = dict.fromkeys(setSizes)
    train_size = dict.fromkeys(setSizes)

    X_test = dict.fromkeys(setSizes)
    target_test = dict.fromkeys(setSizes)
    Y_test = dict.fromkeys(setSizes)
    test_size = dict.fromkeys(setSizes)

    for setSize in setSizes:
        train_size[setSize] = (trainingData['setSize']==setSize).sum()
        X_train[setSize] = np.array(trainingData.loc[trainingData['setSize']==setSize,'itemSet'].tolist())
        target_train[setSize] = np.array(trainingData.loc[trainingData['setSize']==setSize,taskName].tolist())
        Y_train[setSize] = np.array(trainingData.loc[trainingData['setSize']==setSize,rewardName])
        Y_train[setSize] = np.concatenate((Y_train[setSize].reshape(-1,1),1-Y_train[setSize].reshape(-1,1)),axis=1)

        test_size[setSize] = (testData['setSize']==setSize).sum()
        X_test[setSize] = np.array(testData.loc[testData['setSize']==setSize,'itemSet'].tolist())
        target_test[setSize] = np.array(testData.loc[testData['setSize']==setSize,taskName].tolist())
        Y_test[setSize] = np.array(testData.loc[testData['setSize']==setSize,rewardName])
        Y_test[setSize] = np.concatenate((Y_test[setSize].reshape(-1,1),1-Y_test[setSize].reshape(-1,1)),axis=1)

    # tf Graph input
    X = tf.placeholder(tf.int32, [None,None])
    target = tf.placeholder(tf.int32, [None])
    Y = tf.placeholder(tf.float32, [None,2])
    itemSet_ = tf.placeholder(tf.int32, [None])
    true_target = tf.placeholder(tf.int32, [1])

    # Store layers weight & bias
    weights = {
        'V': tf.Variable(tf.random_normal([numItems, numTraits],mean=0.0,stddev=0.01)),
        'R': tf.Variable(tf.random_normal([numItems, numTraits],mean=1.0,stddev=0.01)),
        'D': tf.Variable(tf.random_normal([numItems],mean=1.0,stddev=0.01))
    }

    # Construct model
    logits = logsigma(X,target)
    perf = predict(itemSet_,true_target)

    # Define loss and optimizer
    loss_op = tf.losses.log_loss(labels=Y,predictions=logits)#+alpha*regularization(X,target)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.01,beta2=0.01) # beta plus petit ?
    #optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate) 
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Initializing the variables
    init = tf.global_variables_initializer()

    print("Start learning")
    with tf.Session() as sess:
        sess.run(init)
        
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            print("epoch",epoch+1,"/",training_epochs,end="\r")
            shuffle(setSizes)
            for setSize in setSizes:
                miniBatchStartIndex = 0
                index = list(range(train_size[setSize]))
                shuffle(index)
                total_batch = int(train_size[setSize]/minibatchSize)+1
                # Loop over all batches
                while miniBatchStartIndex<train_size[setSize]:
                    if miniBatchStartIndex+minibatchSize>len(index):
                        miniBatchIndex = index[miniBatchStartIndex:]
                        miniBatchStartIndex = train_size[setSize]
                    else:
                        miniBatchIndex = index[miniBatchStartIndex:miniBatchStartIndex+minibatchSize]
                        miniBatchStartIndex += minibatchSize
                    batch_x, batch_target, batch_y = X_train[setSize][miniBatchIndex,:], target_train[setSize][miniBatchIndex], Y_train[setSize][miniBatchIndex,:]

                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, target: batch_target, Y: batch_y})
                    avg_cost += c / total_batch

            #if epoch%20==0:
            #    M = tf.reduce_max(tf.abs(weights['D']))
            #    print('max D:',M.eval())

            if (intermediatePerf & ((epoch+1)%50==0)) | ((epoch+1)==training_epochs):
                print("Epoch:", '%02d' % (epoch+1), "cost={:.9f}".format(avg_cost))
                # Test model
                # performance on target
                precisions = np.zeros(len(Ks))
                MPR_ = []
                total_testSize = 0
                print("Start prediction at epoch",epoch+1)
                setSizes.sort()
                for setSize in setSizes:
                    print("setSize=",setSize,"/",len(setSizes),end="\r")
                    total_testSize += test_size[setSize]
                    for i in range(test_size[setSize]):
                        tmp, PR = sess.run(perf,feed_dict={itemSet_: X_test[setSize][i], true_target: [target_test[setSize][i]]})
                        MPR_.append(PR)
                        tmp = np.squeeze(np.array(tmp))
                        precisions += tmp

                MPR[run] = (1-np.mean(MPR_)/numItems)*100
                for K in Ks:
                    P[K][run] = precisions[Ks.index(K)]/total_testSize*100
                
                print("Precisions:", precisions/total_testSize*100)
                print("MPR:",MPR[run])
                print("\n")

print("Mean Percentile Rank=",np.mean(MPR))
for K in Ks:
    print("Precision @"+str(K)+"=",np.mean(P[K]))

t1 = time.time()
print('total script time:',t1-t0)