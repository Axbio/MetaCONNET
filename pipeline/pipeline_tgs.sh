#!/bin/bash

assembly=$1
tgs=$2
tag=$3
wdir=$4
model1=$5
model2=$6
thread=$7
mode=$8

round=2
echo " Params: \nAssembly:$assembly \n TGS Read: $tgs \n Tag: $tag \n Working Dir: $wdir \n Model1: $model1 \n Model2: $model2 \n Thread: $thread \n Round: $round"

rm -rf $wdir/data
mkdir $wdir/data
ln -s $assembly $wdir/data/draft_assembly_r1.fa

for r in $(seq 1 2);do
    echo $r
    rm -rf $wdir/round$r
    mkdir $wdir/round$r
    mkdir $wdir/round$r/phase1
    mkdir $wdir/round$r/phase2
    
    tmp_tag=${tag}_r${r}
    tmp_assembly=$wdir/data/draft_assembly_r$r
    
    bash mapping_tgs.sh $tmp_assembly.fa $tgs $wdir/data/$tmp_tag $mode $thread
    date "+Start Correction Time:  %D %T"
    bash correct.sh $tmp_assembly $model1 $wdir/data/$tmp_tag.sort_deN.bam $tmp_tag $wdir/round$r/phase1 $thread
    bash mapping_tgs.sh $wdir/round$r/phase1/polished_draft_assembly.fa $tgs $wdir/data/${tmp_tag}_rec $mode $thread
    date "+Start Recovery Time:  %D %T"
    bash recover.sh $wdir/round$r/phase1/polished_draft_assembly $model2 $wdir/data/${tmp_tag}_rec.sort_deN.bam  ${tmp_tag}_rec $wdir/round$r/phase2 $thread
    ln -s $wdir/round$r/phase2/recovered_assembly.fa $wdir/data/draft_assembly_r$((r+1)).fa
done

ln -s $wdir/data/draft_assembly_r3.fa $wdir/${tag}_polished.fasta

date "+Completion Time:  %D %T"

