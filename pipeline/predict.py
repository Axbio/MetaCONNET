import os, sys
import logging
import tensorflow as tf
import numpy as np
import keras
from .preparator import DataPreparator
import argparse
import pandas as pd
from multiprocessing import Process
import multiprocessing as mp
from collections import defaultdict


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
format= '%(asctime)s : %(levelname)s %(message)s',
datefmt="%Y-%m-%d %A %H:%M:%S"
)

logging.getLogger('tensorflow').disabled = True
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
manager = mp.Manager()

class Scheduler:
    def __init__(self, cpu_number, bam,  feat_size, model, chunks, phase=0):
        self._return_list = manager.list()
        self._queue = manager.Queue()
        # self._Xs = Xs
        self._chunks = chunks
        self.__init_workers(cpu_number, len(chunks), bam, feat_size, model, phase)
        

    def __init_workers(self, cpu_number, task_number, bam,  feat_size, model, phase):
        self._workers = list()
        if cpu_number > task_number:
            max_num = task_number
        else:
            max_num = cpu_number
        for i in range(max_num):
            self._workers.append(Worker(i, self._queue, bam, feat_size, model, self._return_list, phase))

    def start(self):
        
        print(f"worker number {len(self._workers)}")

        print(f"Task: {len(self._chunks)}")
        for chunk in self._chunks:
            self._queue.put(chunk)

        #add a None into queue to indicate the end of task
        self._queue.put(None)

        for worker in self._workers:
            worker.start()

        # wait all fo workers finish
        for worker in self._workers:
            worker.join()
        
        print("all of workers have been done")
        return self._return_list

class PreprocessLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(PreprocessLayer, self).__init__()

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[-1], keepdims=True)
        x = (inputs - mean) / tf.math.sqrt(var + 1e-6)
        return x       

class Worker(Process):
    def __init__(self, wid, queue, bam, feat_size, model, return_list, phase=0):
        Process.__init__(self, name='ModelProcessor')
        self.wid = wid
        self._queue = queue
        self._feat_size = feat_size
        self._model = model
        self._return_list = return_list
        self._bam = bam
        self._model = model
        self._phase = phase

    def run(self):
        model = keras.models.load_model(self._model, custom_objects={'PreprocessLayer':PreprocessLayer()})
        while True:
            print(f"running worker {self.wid}")
            try:
                task = self._queue.get(timeout=10)
                if task is None:
                    self._queue.put(None)
                    break
                print(f"task{task}")
                ret = self.predict(task, model)
                self._return_list.append((task[0], task[1], ret))
            except Exception as error:
                print("Timeout occurred: " + str(error))

    def predict(self, chunk, model):
        preparator = DataPreparator(self._bam, None, chunk[0], chunk[1], chunk[2], self._phase)
        X = preparator.gen_x_data()
        if self._phase == 0:
            ret = run_phase1_predict(X, model, self._feat_size)
        else:
            ret = run_phase2_predict(X, model, self._feat_size)
        return ret

def run_phase1_predict(X, model, feat_size):
    y0 = model.predict_on_batch(X)
    print("finish predicting")
    X = np.reshape(X, (-1, feat_size))
    y0 = np.reshape(y0,(-1, 5))
    y0_class = y0.argmax(axis=-1)
    # keep_idx_x = np.where(np.sum(X, axis = 1) >= 4)[0] ## this is useful for assembly polishing, but not helpful in consensus where read number could be 1-2 rounds
    keep_idx_x = np.where(np.sum(X, axis = 1) >= 0)[0]
    keep_idx_y = np.where(y0_class < 4)[0]
    keep_idx = np.intersect1d(keep_idx_x, keep_idx_y, assume_unique = True)
    y0_class = y0_class[keep_idx]
    ret = "".join(map(lambda x:"ACGT"[x], y0_class))
    return ret

def run_phase2_predict(X, model, feat_size):
    y1 = model.predict_on_batch(X)
    y1 = np.reshape(y1,(-1, 6))
    y1_class = y1.argmax(axis=-1)
    keep_idx = np.where(y1_class != 0)[0]
    return [(idx, y1_class[idx]) for idx in keep_idx]
    #for idx in keep_idx: OUT.write("%d,%d\n" % (offset+idx,y1_class[idx]))
    #offset += 100000

def get_chunks(ctg, start, end, needs_tailing=True):
    tailing = False
    chunks = []
    if end - start + 1 > 100002:
        for step in zip(range(start, end - 100001, 100000), range(start + 100001, end, 100000)):
            chunks.append((ctg, step[0], step[1]))
            ter = step[1]
        if ter < end:
            chunks.append((ctg, ter - 1, end))
    else:
        chunks.append((ctg, start, end))

    if (end - start + 1) % 500 != 2 and needs_tailing:    
        end_trim = end - start - 500
        chunks.append((ctg, end_trim,  end + 2))
        tailing = True
    return chunks, tailing

class CorrectionPredictor:

    def __init__(self, bam,  model, output_name, time_steps=500, feat_size=250, n_class=5, cpu_number=48):
        self.bam = bam
        self.model = model
        self.output_name = output_name
        self.logger = logging.getLogger()
        self.time_steps = time_steps
        self.feat_size = feat_size
        self.n_class = n_class
        self.cpu_number = cpu_number
        
    def predict(self, ctg, start, end, out):       
        
        self.logger.info(f"Preparing input data...\n")
        self.logger.info(f"{self.bam}:{ctg}:{start}-{end}")  
        
        chunks, tailing = get_chunks(ctg, start, end)
        scheculer = Scheduler(self.cpu_number, self.bam, self.feat_size, self.model, chunks)
        sequence = scheculer.start()
        
        if tailing:
            tail = (end - start + 1) % 500
            print(tail)
            sequence[-1] = sequence[-1][- tail:]
        
        with open(os.path.join(out, "%s.%s.%s.%s.phase1.fa.0" % (self.output_name, ctg, start, end)), "w") as OUT:
            OUT.write(f">{ctg} {start}:{end}\n")
            OUT.write("".join(sequence))
            OUT.write("\n")
        self.logger.info("Finish correction prediction\n")

    def predict_bulk(self, bulks, out):       
        
        self.logger.info(f"Preparing input data...\n")
        # bulks = [i for i in sorted(bulks, key=lambda i: i[0]+i[1])]
        
        total_chunks = []
        tailing_list = []
 
        for bulk in bulks:
            chunks, tailing = get_chunks(bulk[0], bulk[1], bulk[2])
            total_chunks.extend(chunks)
            if tailing:
                tailing_list.append(bulk[0])
        scheculer = Scheduler(self.cpu_number, self.bam, self.feat_size, self.model, total_chunks)
        sequence_list = scheculer.start()
        sequence_dict = defaultdict(list)

        for seq in sequence_list:
            sequence_dict[seq[0]].append((seq[1], seq[2]))

        for bulk in bulks:
            sequences = [j for i, j in sorted(sequence_dict[bulk[0]], key=lambda i: i[0])]
            if bulk[0] in tailing_list:
                tail = (bulk[2] - bulk[1] + 1) % 500
                sequences[-1] = sequences[-1][502-tail:]
        
            with open(os.path.join(out, "%s.%s.phase1.fa" % (self.output_name, bulk[0])), "w") as OUT:
                OUT.write(f">{bulk[0]}\n")
                OUT.write("".join(sequences))
                OUT.write("\n")
        self.logger.info("Finish correction prediction\n")


class RecoveryPredictor:
    def __init__(self, bam, model, output_name, time_steps=500, feat_size=550, n_class=6, cpu_number=48):
        self.bam = bam
        self.model = model
        self.output_name = output_name
        self.logger = logging.getLogger()
        self.time_steps = time_steps
        self.feat_size = feat_size
        self.n_class = n_class
        self.cpu_number = cpu_number

    def predict(self, ctg, start, end, out):
        
        self.logger.info(f"Predicting file {self.bam} region: {ctg}:{start}-{end}\n")

        #  allow_soft_placement=True,
        self.logger.info(f"Preparing input data...\n")
        self.logger.info(f"{self.bam}:{ctg}:{start}-{end}")
        
        chunks, tailing = get_chunks(ctg, start, end, False)
        scheculer = Scheduler(self.cpu_number, self.bam, self.feat_size, self.model, chunks, phase=1)
        sequence = scheculer.start()
        sequences = [(i,j) for _, i, j in sorted(sequence, key=lambda i: i[1])]
        with open(os.path.join(out, "%s.%s.%s.%s.phase2.fa" % (self.output_name, ctg, start, end)), "w") as OUT:
            for seq in sequences:
                offset = seq[0]
                for idx, s in seq[1]:
                    OUT.write("%d,%d\n" % (offset+idx-1,s))
        self.logger.info("Finish recovery prediction\n")

    
    def predict_bulk(self, bulks, out): 

        self.logger.info(f"Preparing input data...\n")
        # bulks = [i for i in sorted(bulks, key=lambda i: i[0]+i[1])]
        
        total_chunks = []
        for bulk in bulks:
            chunks, tailing = get_chunks(bulk[0], bulk[1], bulk[2], False)
            total_chunks.extend(chunks)

        scheculer = Scheduler(self.cpu_number, self.bam, self.feat_size, self.model, total_chunks, phase=1)
        sequence_list = scheculer.start()
        sequence_dict = defaultdict(list)
        for seq in sequence_list:
            sequence_dict[seq[0]].append((seq[1], seq[2]))
        ### TODO: 改成loop over sequence names, and one contig save to one file, and change the code in apply ins to fix deletion in whole contigs
        for bulk in bulks:
            sequences = [(i,j) for i, j in sorted(sequence_dict[bulk[0]], key=lambda i: i[0])]
            with open(os.path.join(out, "%s.%s.phase2.fa" % (self.output_name, bulk[0])), "w") as OUT:
                for seq in sequences:
                    offset = seq[0]
                    for idx, s in seq[1]:
                        OUT.write("%d,%d\n" % (offset+idx-1,s))                    
        self.logger.info("Finish recovery prediction\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CONNET phase1 prediction')
    parser.add_argument('-model')
    parser.add_argument('-parallel', required=False)
    parser.add_argument('-mode', help="input correction or recovery")
    parser.add_argument('-bam')
    parser.add_argument('-tag')
    parser.add_argument('-d', help="output directory")
    parser.add_argument('-cpus', default=48, type=int)
    parser.add_argument('-ctg', required=False)
    parser.add_argument('-start', required=False)
    parser.add_argument('-end', required=False)
    args = parser.parse_args()

    if args.mode == "correction":
        predictor = CorrectionPredictor(args.bam,  args.model, args.tag, cpu_number=args.cpus)
    elif args.mode == "recovery":
        predictor = RecoveryPredictor(args.bam,  args.model, args.tag, cpu_number=args.cpus)
    else:
        pass

    if args.parallel:
        parallel_table = pd.read_csv(args.parallel, sep='\s', names=["ctg", "length", "offsets", "LINEBASES", "LINEWIDTH"])

        bulks = []
        for index in range(parallel_table.shape[0]):
            row = parallel_table.loc[index]
            ctg = row["ctg"]
            start = 1
            end = row["length"]
            if end < 500:
                continue
            bulks.append((ctg, start, end))
        predictor.predict_bulk(bulks, args.d)
    else:
        predictor.predict(args.ctg, int(args.start), int(args.end), args.d)