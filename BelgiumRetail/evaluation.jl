using DataFrames
using DataStructures

path = "/home/romain/Documents/PhD/logisticDPP/BelgiumRetail"
dataFileName = "$path/testDataWithTarget.csv"
dataTest = readtable(dataFileName)

V = readtable("$path/V.txt",separator=' ',header=false);
V = Array(V);

D = readtable("$path/D.txt",separator=' ',header=false);
D = Array(D);

R = readtable("$path/R.txt",separator=' ',header=false);
R = Array(R);

numTask = size(R)[1]
numItems = size(V)[1]
taskName = :target
setName = :itemSet
Ks = [5,10,15]

function literal_eval(stringSet)
	itemSet = split(stringSet,", ")
	itemSet[1] = split(itemSet[1],"[")[2]
	itemSet[end] = split(itemSet[end],"]")[1]
	itemSet = map(x->parse(Int64,x)+1,itemSet)
	return itemSet
end

function predictBasket(ind)
	scores = []
	precision5 = 0
	precision10 = 0
	precision20 = 0
	true_target = dataTest[ind,taskName]+1
    itemSet = literal_eval(dataTest[ind,setName])
    subV = V[itemSet,:]
    subD = D[itemSet]
    for task in 1:numTask
    	@inbounds subK = subV * diagm(R[task,:]) * subV' + diagm(subD.^2)
	    @inbounds det_ = det(subK)
	    @inbounds push!(scores,det_)
	end
    y0 = scores[true_target]
    rank = sum(scores.<y0)/numItems

    top20 = sortperm(scores)[end-19:end]
    top10 = top20[end-9:end]
    top5 = top10[end-4:end]

    if true_target in top5
    	precision5 = 1
    	precision10 = 1
    	precision20 = 1
    elseif true_target in top10
    	precision10 = 1
    	precision20 = 1
    elseif true_target in top20
    	precision20 = 1
    end

    return rank, precision5, precision10, precision20
end

percentileRank = []
precisions5 = 0
precisions10 = 0
precisions20 = 0

println("Start prediction")

n = size(dataTest)[1]
for ind in 1:n
	if ind%100==0
		println("iter $ind / $n")
	end
	rank, precision5, precision10, precision20 = predictBasket(ind)
	push!(percentileRank,rank)
	precisions5 += precision5
	precisions10 += precision10
	precisions20 += precision20
end

MPR = 100*mean(percentileRank)
P5 = 100*precisions5/n
P10 = 100*precisions10/n
P20 = 100*precisions20/n

println("MPR: $MPR")
println("P@5: $P5")
println("P@10: $P10")
println("P@20: $P20")