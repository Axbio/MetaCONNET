# MetaCONNET
A tool for metagenomics assemblies polishing, especially for Nanopore long read assemblies

# How to install

## Dependency
- minimap2
- samtools
- python3    
**Make sure samtools, minimap2 and python are at the bin path**

## Installation
1. Clone this project 
2. At project, run `pip install .`

# How to use

## Commands

```
usage: MetaCONNET [-h] [--sr SR SR] --lr LR --c C --o O --n N [--t T] [--v]

MetaCONNET input

optional arguments:
  -h, --help            show this help message and exit
  --sr SR SR            NGS read fastq/fasta read files
  --lr LR               long read fastq/fasta file
  --c C                 contig fastq/fastq file
  --o O                 output folder
  --n N                 task name
  --t T                 thread number
  --v, --version        show program's version number and exit
```

`--lr` and `--c` are required. If you want to combine short reads, please use `--sr` with read1 and read2 file path.   
You can use `--o` to specify a working directory.    
You will need to specify a name for the task as the future prefix for the output file by `--n`.    
MetaCONNT has allowed multithreading, so please remember to include thread numbers `--t` for parallel.

## Output
These directories and files will be created during polishing at the working directory.
- data
- round1
- round2 
- <task_name>_polished.fasta

The polished fasta is in <task_name>_polished.fasta as a softlink to last fasta in round2 recovery phase

## Testing

for testing purposes, you can `cd` to the test folder and execute test.sh. The test data is also in the test folder.     

## Demo
https://github.com/Axbio/MetaCONNET/assets/164155007/39fa76ec-51ff-4761-b011-de0701c89f9f



