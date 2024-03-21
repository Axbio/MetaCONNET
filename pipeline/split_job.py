import os
import sys
import itertools
import traceback
SAMTOOLS = "samtools"
"""
Split job into smaller batches,
each batch contains one contig only.

Only supports < 26 ** 3 batches now.

param:

R: job name
BATCH: size in bp, default 8Mbp for phase1, 3.5Mbp for phase2
"""

alphabet = "abcdefghijklmnopqrstuvwxyz"
names = ["".join(x) for x in itertools.product(alphabet, alphabet, alphabet)]


def run_split_tasks(fa_file, batch_size):
    cmd = f"{SAMTOOLS} faidx {fa_file}" 
    os.system(cmd)
    faidx = open(f"{fa_file}.fai").read().splitlines()
    jobs = []
    for row in faidx:
        row = row.split()
        ctg = row[0]
        length = int(row[1])
        begin = 0

        while begin + 100 < length:
            end = min(length, begin + batch_size)
            jobs.append([ctg, begin+1, end, names[begin // batch_size]])
            begin += batch_size
    file_name = ".".join(fa_file.split(".")[:-1])
    OUT = open(f"{file_name}.parallel", "w")
    for j in jobs:
        OUT.write(f"{file_name} {j[0]} {j[3]} {j[1]} {j[2]}\n")
    OUT.close()


if __name__ == '__main__':
    fa_file = sys.argv[1]
    batch_size = int(sys.argv[2])
    # try:
    #     run_split_tasks(fa_file, batch_size)
    # except:
    #     traceback.print_exc()
    run_split_tasks(fa_file, batch_size)