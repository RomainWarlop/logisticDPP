#!/bin/bash

datadir=$(readlink -f $1)
outdir=$(readlink -f $2)
K=$3
iter=100

echo "creating directory structure"
if [ -d $outdir ]; then
    rm -rf $outdir
fi
mkdir $outdir

mkdir $outdir/spf
mkdir $outdir/pf
mkdir $outdir/sf
mkdir $outdir/pop
mkdir $outdir/rand

seed=948237247

cd ../src

echo " * initializing study of main model (this will launch multiple processes"
echo "   that will continue living after this bash script has completed)"

convf=100
savef=1000
mini=100
maxi=1000

(time (./spf --data $datadir --out $outdir/spf --svi --K $K --seed $seed --save_freq $savef --conv_freq $conf --min_iter $mini --max_iter $maxi --final_pass > $outdir/spf.out 2> $outdir/spf.err) > $outdir/spf.time.out 2> $outdir/spf.time.err &)
(time (./spf --data $datadir --out $outdir/pf --svi --K $K --seed $seed --save_freq $savef --conv_freq $conf --factor_only --min_iter $mini --max_iter $maxi --final_pass > $outdir/pf.out 2> $outdir/pf.err) > $outdir/pf.time.out 2> $outdir/pf.time.err &)
(time (./spf --data $datadir --out $outdir/sf --svi --K $K --seed $seed --save_freq $savef --conv_freq $conf --social_only --min_iter $mini --max_iter $maxi --final_pass > $outdir/sf.out 2> $outdir/sf.err) > $outdir/sf.time.out 2> $outdir/sf.time.err &)

(time (./pop --data $datadir --out $outdir/pop > $outdir/pop.out 2> $outdir/pop.err) > $outdir/pop.time.out 2> $outdir/pop.time.err &)
(time (./rand --data $datadir --out $outdir/rand > $outdir/rand.out 2> $outdir/rand.err) > $outdir/rand.time.out 2> $outdir/rand.time.err &)
