# MetaCONNET
A tool for metagenomics assemblies polishing, especially for Nanopore long read assemblies

# How to install

## Conda installation
1. download conda from https://docs.anaconda.com/miniconda/ ; run `conda init` and reopen shell terminal
2. clone this project
3. At project directory, run `bash install.bash`

The conda environment metaconda will be created and the metaconnet is in the conda env.

## No conda installation
### Step 1 : Install dependencies
- minimap2
- samtools
- python3.8 or python 3.9   
**Make sure samtools, minimap2 and python are at the bin path**

### Step 2: Installation
1. Clone this project 
2. At project directory, run `pip install -r requirements.txt`
3. At project directory, run `pip install .`

# How to use

## Commands

```
usage: MetaCONNET [-h] [--sr SR SR] --lr LR --c C --o O --n N [--t T] [--v] [--fc FC]

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
  --fc, --flowcell      flow cell version : R10, R9. if not given, R9 flow cell model will be used
```

`--lr` and `--c` are required. If you want to combine short reads, please use `--sr` with read1 and read2 file path.   
You can use `--o` to specify a working directory.    
You will need to specify a name for the task as the future prefix for the output file by `--n`.    
MetaCONNT has allowed multithreading, so please remember to include thread numbers `--t` for parallel.
Version 1.1 has added R10 model, specify `--fc` as `R10` to use R10 models 

## Output
These directories and files will be created during polishing at the working directory.
- data
- round1
- round2 
- <task_name>_polished.fasta

The polished fasta is in <task_name>_polished.fasta as a softlink to last fasta in round2 recovery phase

## Testing

for testing purposes, you can `cd` to the `test` folder and execute `bash test.sh`. The test data read : `longreads.fastq` and contigs: `contigs.fasta` are also in the test folder.     

## Demo
https://github.com/Axbio/MetaCONNET/assets/164155007/39fa76ec-51ff-4761-b011-de0701c89f9f




