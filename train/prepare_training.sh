#! /bin/bash

echo "Prepare training files for CONNET model training"
#
SCRIPT_FILE=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT_FILE)

wdir=$1
ref=$2
bam=$3
assembly=$4
phase=$5
tag=$6


label=$wdir/${tag}_corrected_phase$phase.txt

echo "params:"
echo "Working DIR: $wdir\n Reference: $ref\n BAM: $bam \n Assembly: $assembly \n Phase: $phase \n Tag: $tag \n"

echo "prepare labels"

minimap2 -a -x asm10 -B5 -O4,16 --no-long-join -r 200 -N 50 -s 65 -z 200 --mask-level 0.9 --min-occ 200 -g 2500 --score-N 2 --cs -t 24  $assembly $ref> $wdir/$tag.sam

echo "minimap2 -a -x asm10 -B5 -O4,16 --no-long-join -r 200 -N 50 -s 65 -z 200 --mask-level 0.9 --min-occ 200 -g 2500 --score-N 2 --cs -t 24  $assembly $ref > $wdir/$tag.sam"

python3 $SCRIPT_DIR/extract_alignment.py $wdir/$tag.sam

python3 $SCRIPT_DIR/from_pos_to_fa.py -pos $wdir/$tag.alignment  -ref $ref  -contigs $assembly  -tag $tag
python3 $SCRIPT_DIR/split_training.py $wdir/$tag.region 1000000

echo "
python3 $SCRIPT_DIR/extract_alignment.py $wdir/$tag.sam

python3 $SCRIPT_DIR/from_pos_to_fa.py -pos $wdir/$tag.alignment  -ref $ref  -contigs $assembly  -tag $tag
python3 $SCRIPT_DIR/split_training.py $wdir/$tag.region 1000000
"

echo "prepare labeled tensors"
metaconnet_prepare -bam $bam  -ref $label  -parallel $wdir/$tag.parallel  -o $wdir -phase $phase

echo "metaconnet_prepare -bam $bam  -ref $label  -parallel $wdir/$tag.parallel  -o $wdir -phase $phase"
