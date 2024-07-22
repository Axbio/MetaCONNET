import argparse
import re
from Bio import SeqIO
import os
import logging
import sys

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
format= '%(asctime)s : %(levelname)s %(message)s',
datefmt="%Y-%m-%d %A %H:%M:%S"
)
def get_complement(s):
    if s == "A":
        return "T"
    if s == "T":
        return "A"
    if s == "C":
        return "G"
    if s == "G":
        return "C"
    return "X"

def correct_draft_assembly(contig_dict, ref_dict, pos, phase=0):
      
    corrected_assembly = {} 
    alignment = open(pos, "r")
    pos_dict = dict()
    for key in contig_dict.keys():
        if phase == 0:
            pos_dict[key] = ["-"] * len(contig_dict[key].seq) ## TODO key NA
        else:
            pos_dict[key] = ["-"] * len(contig_dict[key].seq) 
    i = 0
    deletion_length = 0
    while True: 
        match = alignment.readline()
        if not match:
            break
        l = match.strip().split("\t")
        # alignment file
        # ctg    ref contig_pos  ref_pos    strand
        contig_name = l[0]
        ref_name = l[1]
        contig_pos = int(l[2]) if l[2] != 'None' and l[2] != None else None
        ref_pos = int(l[3]) if l[3] != 'None' and l[3] != None else None
        strand = l[4]
        contig_seq = contig_dict[contig_name].seq 
        ref_seq = ref_dict[ref_name].seq
        
        if contig_pos == None:
            if phase == 0:
                continue
            else:
                ## deletion
                if ref_pos != None:
                    deletion_length = deletion_length + 1
        else:
            if phase == 1:
                if deletion_length > 0:
                    # if strand == "-":
                    #     pos_dict[contig_name][contig_pos - 1] = deletion_length
                    #     # pos_dict[contig_name][contig_pos] = 0
                    # else:
                    pos_dict[contig_name][contig_pos - 2] = deletion_length
                    pos_dict[contig_name][contig_pos - 1] = 0
                else:
                    pos_dict[contig_name][contig_pos - 1] = 0
                deletion_length = 0
            elif ref_pos == None:
                ## insertion
                pos_dict[contig_name][contig_pos - 1] = "X" # inserted sequence vs misassembled
            else:
                if strand == "-":
                    pos_dict[contig_name][contig_pos - 1] = get_complement(ref_seq[ref_pos - 1])

                else:
                    pos_dict[contig_name][contig_pos - 1] = ref_seq[ref_pos - 1]

        i = i + 1
        if  (i % 1000 == 0) and i >= 1000 : 
            logger.info("进度 %d kbp" % (i//1000))
    
    for key in contig_dict.keys():
        if phase == 0:
            corrected_assembly[key] = "".join(pos_dict[key])
        else:
            corrected_assembly[key] = ",".join([str(i) for i in pos_dict[key]])

    alignment.close()
    return corrected_assembly


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mark draft assembly using aligned reference')
    parser.add_argument('-pos')
    parser.add_argument('-ref')
    parser.add_argument('-contigs')
    parser.add_argument('-tag')
    args = parser.parse_args()

    tag = args.tag
    logger.info(f"####################################")
    logger.info("Start")
    
    logger.info(f"Start correcting {args.contigs} to reference {args.ref} using alignment {args.pos}")
    contig_dict = SeqIO.to_dict(SeqIO.parse(args.contigs, "fasta"))
    ref_dict = SeqIO.to_dict(SeqIO.parse(args.ref, "fasta"))
    # alignment = os.popen(cmd).read().splitlines()
    
    seq1_dict = correct_draft_assembly(contig_dict, ref_dict, args.pos, 0)
    seq2_dict = correct_draft_assembly(contig_dict, ref_dict, args.pos, 1)

    wdir = os.path.dirname(args.pos)
    logger.info("Path:" + os.path.join(wdir, f"{tag}_corrected_phase0.txt"))
    logger.info("Writing")
    with open(os.path.join(wdir, f"{tag}_corrected_phase0.txt"), "w") as f1:
        with open(os.path.join(wdir, f"{tag}_corrected_phase1.txt"), "w") as f3:
            for contig in seq1_dict.keys():
                # write phase1 sequence
                f1.write(f">{contig}\n")
                f1.write(f"{seq1_dict[contig]}\n")
            
                # write phase2 sequence
                f3.write(f">{contig}\n")
                f3.write(f"{seq2_dict[contig]}\n")
    logger.info("End")
    logger.info(f"####################################")

