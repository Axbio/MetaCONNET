#! /bin/bash

python=python
samtools=samtools

assembly=$1
model=$2
bam=$3
tag=$4
wdir=$5
thread=$6

echo "Model: $model"

$samtools faidx $assembly.fa
echo "$samtools faidx $assembly.fa"

$python -m pipeline.predict -model $model -mode correction -bam $bam -tag $tag -d $wdir -cpus $thread -parallel $assembly.fa.fai
echo "$python -m pipeline.predict -model $model -mode correction -bam $bam -tag $tag -d $wdir -cpus $thread -parallel $assembly.fa.fai"

cat $wdir/${tag}.*.phase1.fa > $wdir/polished_draft_assembly.fa
echo "$wdir/${tag}.*.phase1.fa > $wdir/polished_draft_assembly.fa"