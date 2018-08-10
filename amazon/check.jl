using JLD

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

trainingItemSetsDictFileName = "$dataDir/$trainingDataFileName"
trainingItemSetsDictObjectName = "trainingDataDict"
testItemSetsDictFileName = "$dataDir/$testDataFileName"
testItemSetsDictObjectName = "testDataDict"

learnedModelOutputDirName = "$dataDir/tensorLogisticModel"

itemTraitsMatrix = load("$learnedModelOutputDirName/learnedUserDPPItemTraitsMatrix-k$numTraits-lambdaPop$alpha.jld", "itemTraitsMatrix");
itemsWeight = load("$learnedModelOutputDirName/learnedUserDPPItemsWeight-k$numTraits-lambdaPop$alpha.jld", "itemsWeight");
usersWeight = load("$learnedModelOutputDirName/learnedUserDPPUsersWeight-k$numTraits-lambdaPop$alpha.jld","usersWeight");
