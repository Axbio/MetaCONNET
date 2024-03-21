import os
import argparse
import pandas as pd
from .config_parser import SAMTOOLS
import multiprocessing
import glob

def toint(x):
    x = x.lower()
    if x=='a': return 0
    elif x=='c': return 1
    elif x=='g': return 2
    elif x=='t': return 3
    return 4

def insert(fasta, contig, input_bam, workdir):
    print(fasta, contig, input_bam, workdir)
    REF = os.popen(f"{SAMTOOLS} faidx {fasta} {contig}").read().splitlines()[1:] 
    REF = list("".join(REF))

    nr = 0
    PILEUP = iter(os.popen(f"{SAMTOOLS} mpileup -r {contig} -B -q0 -Q0 -aa {input_bam}"))
    RESULT = []
    
    for f in glob.glob(os.path.join(workdir, f"*.{contig}.phase2.fa")):
        for row in open(f):
            pred_pos, pred_len = map(int, row.split(','))
            pred_pos = pred_pos + 1
            while nr <= pred_pos:
                pileup = next(PILEUP)
                nr += 1

            ins_patterns = []
            ins_lengths = []
            p = pileup.split()[4].split('+')[1:] # pileup cannot start with '+', can safely discard the first chunk
            for pp in p:
                ins_len = ""
                idx = 0
                while pp[idx].isdigit():
                    ins_len += pp[idx]
                    idx += 1
            
                ins_len = int(ins_len) if ins_len != "" else 0
                ins_lengths.append(ins_len)
                ins_pattern = pp[idx:idx+ins_len]
                ins_patterns.append(ins_pattern)
        
            median_ins_len = sorted(ins_lengths)[len(ins_lengths)//2] if len(ins_lengths) else 0
            if pred_len == 5 and median_ins_len > 5: 
                pred_len = median_ins_len
        
            vote = [[0,0,0,0,0] for _ in range(pred_len)]
            for pattern in ins_patterns:
                for i,ch in enumerate(pattern):
                    if i==pred_len: break
                    vote[i][toint(ch)] += 1
        
            predicted = "".join([ "ACGTN"[vote[i].index(max(vote[i]))] for i in range(len(vote))])
        
            RESULT.append((pred_pos,predicted))
                        
    for i,p in RESULT[::-1]: 
        REF[i] += p

    OUT = open(os.path.join(workdir, f"{contig}.phase2.recovered.fa"),"w")
    OUT.write(f'>{contig}\n{"".join(REF)}\n') 

if __name__ == '__main__':

    """
    Apply predicted insertions to assembly
    """

    parser = argparse.ArgumentParser(description='CONNET phase2 apply insertion')
    parser.add_argument('-parallel')
    parser.add_argument('-fasta')
    parser.add_argument('-bam')
    parser.add_argument('-d')
    parser.add_argument('-threads', type=int)

    args = parser.parse_args()
    
    parallel_table = pd.read_csv(args.parallel, sep='\s', names=["ctg", "length", "offsets", "LINEBASES", "LINEWIDTH"])
    contigs = parallel_table['ctg'].unique()
    
    #for contig in contigs:
    inputs = [(args.fasta, contig, args.bam, args.d) for contig in contigs]

    with multiprocessing.Pool(processes=args.threads) as pool:
        pool.starmap(insert, inputs)
