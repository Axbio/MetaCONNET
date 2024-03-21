#!/bin/bash
#usage: bash ./bwa_merge_deN.sh Illumina-SRR8073716 ONT10kb-SRR8073713

SAMTOOLS=samtools
MINIMAP2=minimap2
thread=$5

# $1 : draft assembly fasta file
# $2 : ont fa
# $3 : wdir
# $4 : ont or pacbio
# $5 : number of threads

#1 alignment

date "+Start TGS Reads Alignment Time:  %D %T"
# ont alignment
if [ $4 = 'ont' ];then
  echo "$MINIMAP2 -ax map-ont -L -t $thread $1 $2 > $3_tgs.sam"
  $MINIMAP2 -ax map-ont -L -t $thread $1 $2 > $3_tgs.sam
else
  echo "$MINIMAP2 -ax map-pb -L $1 $2 -t $thread> $3_tgs.sam"
  $MINIMAP2 -ax map-pb -L $1 $2 -t $thread> $3_tgs.sam
fi

date "+Start Alignment Post-Processing:  %D %T" 

echo "$SAMTOOLS sort -@ $thread $3_tgs.sam -o $3_tgs.sort.bam"
$SAMTOOLS sort -@ $thread $3_tgs.sam -o $3_tgs.sort.bam
echo "$SAMTOOLS index -@ $thread $3_tgs.sort.bam"
$SAMTOOLS index -@ $thread $3_tgs.sort.bam
echo "$SAMTOOLS view -@ $thread -bhF 12 $3_tgs.sort.bam -o $3_tgs.unimapped.bam"
$SAMTOOLS view -@ $thread -bhF 12 $3_tgs.sort.bam -o $3_tgs.unimapped.bam    #F12(remove unmap reads)
$SAMTOOLS index $3_tgs.unimapped.bam


date "+Start Alignment Merging:  %D %T" 
#2 merge
echo "$SAMTOOLS view -h -@ $thread $3_tgs.unimapped.bam | awk -F ' ' '{if($10!~/N/ || $0~/^@/) print }' > $3.deN.sam"
$SAMTOOLS view -h -@ $thread $3_tgs.unimapped.bam | awk -F " " '{if($10!~/N/ || $0~/^@/) print }' > $3.deN.sam      #remove sequence with "N"
echo "$SAMTOOLS sort -@ $thread $3.deN.sam -o $3.sort_deN.bam"
$SAMTOOLS sort -@ $thread $3.deN.sam -o $3.sort_deN.bam
echo "$SAMTOOLS index $3.sort_deN.bam"
$SAMTOOLS index $3.sort_deN.bam

date "+Alignment Ended:  %D %T" 
