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

For testing purposes, you can `cd` to the `test` folder and execute `bash test.sh`. The test data read : `longreads.fastq` and contigs: `contigs.fasta` are also in the test folder.     


## Model training

For who is interested in training a model of their own, first of all, the models are under `~/pipeline/training_model` .

The training codes are under `~/train`

Step 1 : Prepare training data.    

Run `prepare_training.sh`

Example :
```shell
#corretion
wdir=phase1
bam=map.bam # reads to assembly bam
assembly=assembly.fasta
ref=ref.fasta
time bash prepare_training.sh $wdir $ref  $bam $assembly 0 dataset_name
#recovery
wdir=phase2
time bash prepare_training.sh $wdir $ref  $bam $assembly 1 dataset_name
```
Step 2 : Run training.    
This requires you to have a gpu and a tensorflow-gpu ready environment

Example :
```shell
model1=meta_correction
model2=meta_recovery
#corretion
metaconnet_train -epochs 200 -parallel phase1/dataset_name.parallel -train phase1 -o $model1 -phase correction
#recovery
metaconnet_train -epochs 200 -parallel phase2/dataset_name.parallel -train phase2 -o $model2 -phase recovery
```

This will train 200 epochs of the training datasets and save the final epoch model and each epoch checkpoint. You can load the final model and find the best model checkpoint and save to keras models.




