import argparse
import subprocess
import os
from .config_parser import MODEL1, MODEL2, VERSION

def argparsers():
    parser = argparse.ArgumentParser(description='MetaCONNET input', prog="MetaCONNET")
    parser.add_argument('--sr','--shortread', 
                    help='NGS read fastq/fasta read files',nargs=2, required=False)
    parser.add_argument('--lr','--longread', 
                    help='long read fastq/fasta file', required=True)     
    parser.add_argument('--c','--contigs', 
                    help='contig fastq/fastq file', required=True)
    parser.add_argument('--o','--out', 
                    help='output folder', required=True)              
    parser.add_argument('--n','--name', 
                    help='task name', required=True)  
    parser.add_argument('--t','--threads', 
                    help='thread number', required=False)  
    parser.add_argument('--v','--version', action='version', version=f'%(prog)s {VERSION}')

    return parser

def run():
    parser = argparsers()
    args = parser.parse_args()
        
    model1 = MODEL1
    model2 = MODEL2
    thread = os.cpu_count()
    if args.t :  
        thread = int(args.t)

    if os.path.exists(args.o) and os.path.isdir(args.o) : 
        wdir = os.path.abspath(f"{args.o}")
    else :
        raise Exception (f"{args.o} is not found")

    if args.sr :
        cmd = f"bash pipeline.sh {args.c} {args.sr[0]} {args.sr[1]} {args.lr} {args.n} {wdir} {model1} {model2} {thread} ont "
    else :
        cmd = f"bash pipeline_tgs.sh {args.c} {args.lr} {args.n} {wdir} {model1} {model2} {thread} ont "
    
    print(cmd)
    result = subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    run()
