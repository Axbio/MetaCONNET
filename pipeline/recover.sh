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

$python -m pipeline.predict -model $model -mode recovery -bam $bam -tag $tag -d $wdir -cpus $thread -parallel $assembly.fa.fai

$python  -m pipeline.apply_ins \
-parallel $assembly.fa.fai \
-fasta $assembly.fa \
-d $wdir \
-bam $bam  \
-threads $thread

cat $wdir/*.recovered.fa > $wdir/recovered_assembly.fa