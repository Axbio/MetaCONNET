import logging
import parse_pileup
import numpy as np
from .config_parser import SAMTOOLS
import os
import numpy as np
import argparse
import pandas as pd
import time

def split_batches(df, proportion):
    import random
    N = df.shape[0]
    check_list = []
    s = 0
    while s < int(sum(df['end'] - df['start'] + 1) * proportion):
        m = random.randint(0, N-1)
        if m not in check_list:
            s += df.loc[m, 'end'] - df.loc[m, 'start'] + 1
            check_list.append(m)
        else:
            continue
    return check_list

def convert_fa_to_dict(file_name):
    with open(file_name) as fa:
        fa_dict = {}
        for line in fa:
            # 去除末尾换行符
            line = line.replace('\n','')
            if line.startswith('>'):
                # 去除 > 号
                seq_name = line[1:]
                fa_dict[seq_name] = ''
            else:
                # 去除末尾换行符并连接多行序列
                fa_dict[seq_name] += line.replace('\n','')
    return fa_dict

class DataPreparator:

    def __init__(self, input_bam, input_ref, contig, start, end,  phase=0):
        self.ctg = str(contig)
        self.start = int(start)
        self.end = int(end)
        self.phase = int(phase)
        self.input_bam = input_bam
        self.input_ref = input_ref
        self.logger = logging.getLogger('preprocess')
        self.time_steps = 500
        self.feat_size = 250 if self.phase == 0 else 550

    def run(self):
        x_data = self.gen_x_data()
        y_data = self.gen_y_data()
        np.savez(f"training_data/{self.ctg}_{self.start}_{self.end}.npz", x=x_data, y=y_data)
        
    def gen_x_data(self):
        start_time = time.time()
        print(f"{self.ctg} {self.start} {self.end} start time {start_time}")
        # raw = gen_phase_pipe(self.ctg, self.start, self.end, self.input_bam, self.phase)
        
        if self.phase == 0:
            raw = parse_pileup.gen_phase1(*[self.ctg, self.start, self.end, self.input_bam])
        elif self.phase == 1:
            raw = parse_pileup.gen_phase2(*[self.ctg, self.start, self.end, self.input_bam])
        raw = raw.reshape(-1, self.time_steps, self.feat_size)
        batch_time = time.time()
        print(f"{self.ctg} {self.start} {self.end} batch time {batch_time - start_time}")
        return raw
    
    def gen_y_data(self):
        if self.phase == 0:
            cmd = f"{SAMTOOLS} faidx %s %s:%d-%d" % (self.input_ref, self.ctg, self.start, self.end)
            ref_sequence = "".join(os.popen(cmd).read().splitlines()[1:])

            total_length = len(ref_sequence)
            batch_num = (total_length - 2) // self.time_steps

            set_y = []

            for ref in ref_sequence[1:batch_num*self.time_steps+1]:
                s = []
                for base in ["A", "C", "G", "T", "X"]: 
                    if ref == base:
                        s.append(1)
                    else:
                        s.append(0)
                set_y.append(s)
            arr = np.array(set_y)
            
            return arr.reshape(batch_num, self.time_steps, 5)

        else:

            ref_sequence = convert_fa_to_dict(self.input_ref)[self.ctg].split(",")[self.start-1:self.end]
            total_length = len(ref_sequence)
            batch_num = (total_length - 2) // self.time_steps

            set_y = []
            # for i in range(batch_num*self.time_steps):
            for i in range(1,batch_num*self.time_steps+1):
                s = [0] * 6
                insert_size = 0
                if ref_sequence[i] in ['-']:
                    # set_y.append(s)
                    print(self.ctg, self.start, self.end)
                    continue
                else:
                    insert_size = int(ref_sequence[i])
                if insert_size < 5:
                    s[insert_size] = 1
                else:
                    s[5] = 1
                set_y.append(s)
            arr = np.array(set_y)
            return arr.reshape(batch_num, self.time_steps, 6)

def run ():
    parser = argparse.ArgumentParser(description='Split bam into pileup tensors')
    parser.add_argument('-bam')
    parser.add_argument('-ref')
    parser.add_argument('-parallel')
    parser.add_argument('-phase', default=0)
    parser.add_argument('-o')
    parser.add_argument('-p', default=0.1)

    args = parser.parse_args()
    bam = args.bam
    reference = args.ref
    parallel = args.parallel

    parallel_table = pd.read_csv(parallel, sep='\s', names=["path", "ctg", "name", "start", "end", "ref_start", "ref_end"])
    test_batches = split_batches(parallel_table, args.p)

    if not os.path.exists(args.o):
        os.mkdir(args.o)

    for index in range(parallel_table.shape[0]):
        row = parallel_table.loc[index]
        ctg = row["ctg"]
        start = row["start"]
        end = row["end"]
        pre = DataPreparator(bam, reference, ctg, start, end, phase=args.phase)
        x_data = pre.gen_x_data()
        y_data = pre.gen_y_data()
        if index not in test_batches:
            np.savez(os.path.join(args.o, f"training_{ctg}_{start}_{end}.npz"), x=x_data, y=y_data)
        else:
            np.savez(os.path.join(args.o, f"testing_{ctg}_{start}_{end}.npz"), x=x_data, y=y_data)
    
if __name__ == '__main__':
    run()

    