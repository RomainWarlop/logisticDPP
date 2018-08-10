push!(LOAD_PATH,"/home/romain/Documents/PhD/logisticDPP/src")

using LogisticLowRankDPP

nCore = nprocs()
println("Executing with $nCore cores")

dataDir = "/home/romain/Documents/PhD/logisticDPP/amazon"

# params
lambda = 0.01
alpha = 0.0#2
eps = 0.1e-1
betaMomentum = 0.0#95
numIterFixed = 500
minibatchSize = 100000
numTraits = 20

dataFileName = "/home/romain/Documents/PhD/logisticDPP/amazon/dataWithNegSampling.csv"
trainingDataFileName = "trainingData.jld"
testDataFileName = "testData.jld"
trainingSetSizePercent = 0.8

userName = :target
itemSetName = :itemSet
rewardName = :conversion
setLengthName = :setSize

# convert data to the right format
convertDataToItemSet(dataDir,dataFileName,trainingDataFileName, testDataFileName, trainingSetSizePercent, userName, itemSetName, rewardName, setLengthName)

# learn model
trainingItemSetsDictFileName = "$dataDir/$trainingDataFileName"
trainingItemSetsDictObjectName = "trainingDataDict"
testItemSetsDictFileName = "$dataDir/$testDataFileName"
testItemSetsDictObjectName = "testDataDict"

learnedModelOutputDirName = "$dataDir/tensorLogisticModel"

doUserDPPLearningSparseVectorData(trainingItemSetsDictFileName,
    trainingItemSetsDictObjectName, testItemSetsDictFileName,
    testItemSetsDictObjectName, learnedModelOutputDirName, numTraits, lambda=lambda,
    alpha=alpha, eps=eps, betaMomentum=betaMomentum, numIterFixed=numIterFixed, minibatchSize=minibatchSize, maxIter=500)
