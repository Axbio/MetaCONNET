#!/bin/bash
mkdir -p data
rm -rf data/*
metaconnet --lr longreads.fastq --c contigs.fasta --o data --n test --t 12
