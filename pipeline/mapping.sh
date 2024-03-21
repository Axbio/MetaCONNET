#!/bin/bash
#usage: bash ./bwa_merge_deN.sh Illumina-SRR8073716 ONT10kb-SRR8073713

SAMTOOLS=samtools
MINIMAP2=minimap2
thread=$7

# $1 : draft assembly fasta file
# $2 : illumina pair-end fa 1
# $3 : illumina pair-end fa 2
# $4 : ont fa
# $5 : tag
# $6 : ont or pacbio
# $7 : number of threads
# usage: sh mapping.sh /AxBio_share/users/sbr/model/ONT10kbmini2_round1/recovered_assembly.fa /AxBio_share/users/lwq/project/01_IMAU/01_IMAU/02_HybridMetagenomic/mock_dataset/Illumina/trimmomatic/Illumina-SRR8073716_QC_1P.fastq /AxBio_share/users/lwq/project/01_IMAU/01_IMAU/02_HybridMetagenomic/mock_dataset/Illumina/trimmomatic/Illumina-SRR8073716_QC_2P.fastq /AxBio_share/users/lwq/project/01_IMAU/01_IMAU/02_HybridMetagenomic/mock_dataset/raw_data/ONT10kb-SRR8073713.fastq /AxBio_share/users/sbr/Data/MOCK/ONT10kb_r2

#1 alignment

# $BWA index $1
# illumina alignment
date "+Start Illumina Pair-end Reads Alignment Time:  %D %T"
echo "$MINIMAP2 -ax sr -L --secondary=yes -Y -t $thread $1 $2 $3 > $5_illumina.sam"
$MINIMAP2 -ax sr -L --secondary=yes -Y -t $thread $1 $2 $3 > $5_illumina.sam


date "+Start TGS Reads Alignment Time:  %D %T"
# ont alignment
if [ $6 = 'ont' ];then
  echo "$MINIMAP2 -ax map-ont -L -t $thread $1 $4 > $5_tgs.sam"
  $MINIMAP2 -ax map-ont -L -t $thread $1 $4 > $5_tgs.sam
else
  echo "$MINIMAP2 -ax map-pb -L $1 $4 -t $thread> $5_tgs.sam"
  $MINIMAP2 -ax map-pb -L $1 $4 -t $thread> $5_tgs.sam
fi

date "+Start Alignment Post-Processing:  %D %T" 
for i in "illumina" "tgs";
	do
        echo "$SAMTOOLS sort -@ $thread -m 4G $5_${i}.sam -o $5_${i}.sort.bam"
        $SAMTOOLS sort -@ $thread -m 4G $5_${i}.sam -o $5_${i}.sort.bam
        echo "$SAMTOOLS index -@ $thread -m 4G $5_${i}.sort.bam"
	$SAMTOOLS index -@ $thread -m 4G $5_${i}.sort.bam
	echo "$SAMTOOLS view -@ $thread -m 4G -bhF 12 $5_${i}.sort.bam -o $5_${i}.unimapped.bam"
        $SAMTOOLS view -@ $thread -m 4G -bhF 12 $5_${i}.sort.bam -o $5_${i}.unimapped.bam    
        $SAMTOOLS index $5_${i}.unimapped.bam
done

date "+Start Alignment Merging:  %D %T" 

#2 merge


echo "$SAMTOOLS merge -@ $thread  --write-index -cp -O BAM $5.merge.bam $5_illumina.unimapped.bam $5_tgs.unimapped.bam"
$SAMTOOLS merge -@ $thread --write-index -cp -O BAM $5.merge.bam $5_illumina.unimapped.bam $5_tgs.unimapped.bam

echo "$SAMTOOLS sort -@ $thread -m 4G  $5.merge.bam   -o $5.sort_deN.bam"
$SAMTOOLS sort -@ $thread -m 4G $5.merge.bam -o $5.sort_deN.bam
echo "$SAMTOOLS index $5.sort_deN.bam"
$SAMTOOLS index $5.sort_deN.bam

date "+Alignment Ended:  %D %T" 