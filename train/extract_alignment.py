import pysam
from Bio import SeqIO
import sys
import traceback
import os

def get_seq_coordinate(aln):
    read_end = aln.infer_read_length() - 1
    skip_start = aln.cigartuples[0][1] if aln.cigartuples[0][0] in [4, 5] else 0
    skip_end = aln.cigartuples[-1][1] if aln.cigartuples[-1][0] in [4, 5] else 0

    if aln.is_reverse:
        seq_start = skip_end 
        seq_end = read_end - skip_start
    else:
        seq_start = skip_start 
        seq_end = read_end - skip_end
    return seq_start, seq_end

def extract(sam_file):
    sam = pysam.AlignmentFile(sam_file, "r")
    file_name = sam_file
    wdir, fullname = os.path.split(file_name)
    fname, ext = os.path.splitext(fullname)
    alignment_file = os.path.join(wdir, fname + ".alignment")
    region_file = os.path.join(wdir, fname + ".region")
    alignment = open(alignment_file, "w")
    region = open(region_file, "w")
    contig_pos_map = {}
    contig_map = {}
    for s in sam.fetch():
        if s.is_mapped:

            if s.is_forward:    
                strand = "+"
            else:
                strand = "-"

            if s.reference_name not in contig_pos_map:
                contig_pos_map[s.reference_name] = []
                contig_map[s.reference_name] = []

            ## 
            if len(contig_pos_map[s.reference_name]) != 0:
                to_remove = []
                for i in range(len(contig_pos_map[s.reference_name])):
                    line = contig_pos_map[s.reference_name][i]
                    start = line[0]
                    end = line[1]

                    # 

                    if (s.reference_start < end + 1  and s.reference_end> start) or \
                    (start < s.reference_end  and end + 1 > s.reference_start):
                        if (end + 1 - start ) < s.reference_end - s.reference_start: #
                            to_remove.append((line, contig_map[s.reference_name][i]))
                            to_add = True
                        else:
                            to_add = False
                            break

            else:
                to_add = True
                to_remove = []

            if to_add:
                for m in to_remove:
                    contig_pos_map[s.reference_name].remove(m[0])
                    contig_map[s.reference_name].remove(m[1])
                
                alignment_start, alignment_end = get_seq_coordinate(s)

                if strand == "-":    
                    gap = s.query_alignment_start
                    align_pair = [(alignment_end + gap - i,j) if i != None else (i,j) for i, j in s.get_aligned_pairs() ]
                else:
                    gap = alignment_start - s.query_alignment_start
                    align_pair = [(gap + i,j) if i != None else (i,j) for i, j in s.get_aligned_pairs() ]

                

                contig_map[s.reference_name].append((s.query_name, align_pair))
                contig_pos_map[s.reference_name].append((s.reference_start, s.reference_end - 1, alignment_start, alignment_end, 
                            s.query_name, strand))
    
    sys.stderr.write("finish reading sam")
    for k, item in contig_map.items():
        for i in range(len(item)):
            for row in item[i][1]:
                alignment.write(f"{k}\t{item[i][0]}\t{row[1] + 1 if row[1] != None else None}\t{row[0] + 1 if row[0] != None else None}\t{contig_pos_map[k][i][5]}\n")
            region.write(f"{k}\t{contig_pos_map[k][i][4]}\t{contig_pos_map[k][i][0] + 1}\t{contig_pos_map[k][i][1] + 1}\t{contig_pos_map[k][i][2] + 1}\t{contig_pos_map[k][i][3] + 1}\t{contig_pos_map[k][i][5]}\n")
    
    sys.stderr.write(f"{alignment_file}") 
    alignment.close()
    region.close()           

if __name__ == '__main__':

    """
    useful for extracting alignment positions in sam files
    sample usage:
    python /AxBio_share/users/sbr/model/CONNET/extract_alignment.py /AxBio_share/users/lwq/project/01_IMAU/pairwise/minimap2_asm20/ONT10kb_ref12_10.sam 
    """
    try:
        extract(sys.argv[1])
    except:
        traceback.print_exc()
    
