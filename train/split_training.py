import itertools
import sys
import traceback
import os

alphabet = "abcdefghijklmnopqrstuvwxyz"
names = ["".join(x) for x in itertools.product(alphabet, alphabet, alphabet)]

def run_split_tasks(pos_region, batch_size):
    pos_line = open(f"{pos_region}").read().splitlines()
    jobs = []
    for row in pos_line:
        row = row.split()
        ctg = row[0]
        begin = int(row[2])
        ref_begin = 0
        last = int(row[3])
        length = last - begin + 1
        while ref_begin + 500 < length:
            end = min(last, begin + batch_size - 1)
            ref_end = min(length, ref_begin + batch_size)
            jobs.append([ctg, names[begin // batch_size], begin, end, ref_begin + 1, ref_end])
            begin += batch_size
            ref_begin += batch_size
    file_name = pos_region
    wdir, fullname = os.path.split(file_name)
    fname, ext = os.path.splitext(fullname)
    outfile = os.path.join(wdir, f"{fname}.parallel")
    OUT = open(outfile, "w")
    for j in jobs:
        OUT.write(f"{fname} {j[0]} {j[1]} {j[2]} {j[3]} {j[4]} {j[5]}\n")
    OUT.close()


if __name__ == '__main__':
    pos_region = sys.argv[1]
    batch_size = int(sys.argv[2])
    try:
        run_split_tasks(pos_region, batch_size)
    except:
        traceback.print_exc()
