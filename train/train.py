import os
import tensorflow as tf
import numpy as np
import keras
import pandas as pd
import glob
from keras.layers import Dropout
from config_parser import connet_logger

SEED = 42
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def set_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

class PreprocessLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(PreprocessLayer, self).__init__()

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[-1], keepdims=True)
        x = (inputs - mean) / tf.math.sqrt(var + 1e-6)
        return x

def table_to_paths(df, path):
    return_list = df.apply(lambda row: f"{path}/{row['path']}_training_{row['ctg']}_{row['start']}_{row['end']}.npz", axis=1)
    return list(return_list)

def generate_tensor(file):
    with np.load(file) as data:
        x = data['x'].astype(np.int32)
        y = data['y'].astype(np.int32)
        return x, y

def check_training_file(row, path, file_list):
    ctg = row["ctg"]
    start = row["start"]
    end = row["end"]
    name = row["path"]
    data_path = f"{path}/{name}_training_{ctg}_{start}_{end}.npz"

    if data_path in file_list and ((end - start) > 100):
        return True
    else:
        return False

def check_testing_file(row, path, file_list):
    ctg = row["ctg"]
    start = row["start"]
    end = row["end"]
    name = row["path"]
    data_path = f"{path}/{name}_testing_{ctg}_{start}_{end}.npz"

    if data_path in file_list and ((end - start) > 100):
        return True
    else:
        return False

def load_test_data(path, parallel_table):
    x_arr = []
    y_arr = []
    for index, row in parallel_table.iterrows():
        ctg = row["ctg"]
        start = row["start"]
        end = row["end"]
        name = row["path"]
        print(path)
        print(f"{name}_testing_{ctg}_{start}_{end}.npz")
        data_path = os.path.join(path, f"{name}_testing_{ctg}_{start}_{end}.npz")
        
        npfile = np.load(data_path)
        tf.cast(npfile['x'], dtype="int32")
        tf.cast(npfile['y'], dtype="int32")
        x_arr.append(npfile['x'])
        y_arr.append(npfile['y'])
    X,Y = np.concatenate(x_arr, axis=0), np.concatenate(y_arr, axis=0)
    return X, Y

class BatchGenerator:
    
    def __init__(self, path, parallel_table):
        self.path = path
        self.parallel_table = parallel_table

    def func(self):
        i = 0
        N = self.parallel_table.shape[0]  

        index_list = [i for i in self.parallel_table.index]
        while True:
            row = self.parallel_table.loc[index_list[i]]
            ctg = row["ctg"]
            start = row["start"]
            end = row["end"]
            name = row["path"]
            data_path = f"{self.path}/{name}_training_{ctg}_{start}_{end}.npz"
            npfile = np.load(data_path)
            x = npfile['x'].astype(np.int32)
            y = npfile['y'].astype(np.int32)
            step = 0 
            while step < x.shape[0]:
                step = step + 1
                yield x[step-1], y[step-1]
            i = i + 1
            if i >= N:
                i = 0

class Correction:

    def __init__(self, training_data_path, training_batch, output_name, around=0, time_steps=500, feat_size=250, n_class=5):
        self.training_data_path = training_data_path
        self.training_batch = training_batch
        self.output_name = output_name
        self.time_steps = time_steps
        self.feat_size = feat_size
        self.n_class = n_class
        self.around = around
        wdir, fullname = os.path.split(self.output_name)
        fname, ext = os.path.splitext(fullname)
        self.wdir = wdir
        self.fname = fname
        
    def train(self, epochs):
        connet_logger.info(f"####################################")
        connet_logger.info(f"###########Training Begin###########")
        connet_logger.info(f"####################################")

        if tf.test.gpu_device_name():
            set_gpu()
        
        parallel_table = self.training_batch[self.training_batch['type'] == 'training']

        from sklearn.model_selection import train_test_split

        x_train,  x_validate = train_test_split(parallel_table, test_size=0.1)
        validate_file_list =  table_to_paths(x_validate, self.training_data_path)
        validateset = tf.data.Dataset.list_files(validate_file_list)
        validateset = validateset.shuffle(len(validate_file_list))
        validateset = validateset.map(lambda x: tf.numpy_function(generate_tensor, [x], (tf.int32, tf.int32)))
        validateset = validateset.cache()

        train_file_list =  table_to_paths(x_train, self.training_data_path)
        trainset = tf.data.Dataset.list_files(train_file_list)
        trainset = trainset.shuffle(len(train_file_list))
        trainset = trainset.map(lambda x: tf.numpy_function(generate_tensor, [x], (tf.int32, tf.int32)))
        trainset = trainset.cache()

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.time_steps, self.feat_size)))
        model.add(PreprocessLayer())
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), input_shape=(self.time_steps, self.feat_size))) # could increase LSTM node
        model.add(Dropout(0.5)) # could reduce dropout 
        model.add(tf.keras.layers.Dense(self.n_class, activation='softmax')) # could change softmax, or add leaning rate
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS)
        
        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=15, verbose=1)
        checkpoint_path = f"{self.wdir}/{self.fname}_checkpoint_model_epoch_{{epoch:03d}}.h5"  # Define the path to save the model
        checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_freq='epoch')
        history = model.fit(trainset.prefetch(2),  epochs=epochs, validation_data=validateset, callbacks=[checkpoint_callback, early_stopping]) 

        hist_df = pd.DataFrame(history.history) 
        hist_df.to_csv(os.path.join(self.wdir, f"{self.fname}.history.csv"), index=False, sep="\t")
        
        model.summary()
        model.save(self.output_name)

        connet_logger.info(f"####################################")
        connet_logger.info(f"############Training End############")
        connet_logger.info(f"####################################")
    
    def evaluate(self):
        model = tf.keras.models.load_model(self.output_name)
        connet_logger.info(f"####################################")
        connet_logger.info(f"##########Evaluation Begin##########")
        connet_logger.info(f"####################################")
        
        parallel_table = self.training_batch[self.training_batch['type'] == 'testing']

        x_test_set, y_test_set = load_test_data(self.training_data_path, parallel_table)
        
        results = model.evaluate(x_test_set, y_test_set, verbose=2)
          
        connet_logger.info (results)

        connet_logger.info(f"####################################")
        connet_logger.info(f"##########Evaluation End############")
        connet_logger.info(f"####################################")
    
class Recovery:

    def __init__(self, training_data_path, training_batch, output_name, around=0, time_steps=500, feat_size=550, n_class=6):
        self.training_data_path = training_data_path
        self.training_batch = training_batch
        self.output_name = output_name
        self.time_steps = time_steps
        self.feat_size = feat_size
        self.n_class = n_class
        self.around = around
        wdir, fullname = os.path.split(self.output_name)
        fname, ext = os.path.splitext(fullname)
        self.wdir = wdir
        self.fname = fname
        
    def train(self, epochs):
        connet_logger.info(f"####################################")
        connet_logger.info(f"#########Training Begin#############")
        connet_logger.info(f"####################################")

        if tf.test.gpu_device_name():
            set_gpu()

        parallel_table = self.training_batch[self.training_batch['type'] == 'training']

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.time_steps, self.feat_size)))
        model.add(PreprocessLayer())
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), input_shape=(self.time_steps, self.feat_size)))
        model.add(Dropout(0.5))
        model.add(tf.keras.layers.Dense(self.n_class, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS)
        
        from sklearn.model_selection import train_test_split

        x_train,  x_validate = train_test_split(parallel_table, test_size=0.1, random_state=SEED)
        checkpoint_path = f"{self.wdir}/{self.fname}_checkpoint_model_epoch_{{epoch:03d}}.h5"  # Define the path to save the model
        checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_freq='epoch')

        validate_file_list =  table_to_paths(x_validate, self.training_data_path)
        validateset = tf.data.Dataset.list_files(validate_file_list)
        validateset = validateset.shuffle(len(validate_file_list))
        validateset = validateset.map(lambda x: tf.numpy_function(generate_tensor, [x], (tf.int32, tf.int32)))
        validateset = validateset.cache()

        train_file_list =  table_to_paths(x_train, self.training_data_path)
        trainset = tf.data.Dataset.list_files(train_file_list)
        trainset = trainset.shuffle(len(train_file_list))
        trainset = trainset.map(lambda x: tf.numpy_function(generate_tensor, [x], (tf.int32, tf.int32)))
        trainset = trainset.cache()

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1)

        history = model.fit(trainset,  epochs=epochs,  validation_data=validateset, callbacks=[early_stopping, checkpoint_callback]) ## 只用dataset map就相当于以前用generator 
        hist_df = pd.DataFrame(history.history) 
        hist_df.to_csv(os.path.join(self.wdir, f"{self.fname}.history.csv"), index=False, sep="\t")
        
        model.summary()
        model.save(self.output_name)
        
        connet_logger.info(f"####################################")
        connet_logger.info(f"###########Training End#############")
        connet_logger.info(f"####################################")
    
    def evaluate(self):
        model = tf.keras.models.load_model(self.output_name)
        connet_logger.info(f"####################################")
        connet_logger.info(f"##########Evaluation Begin##########")
        connet_logger.info(f"####################################")
        
        parallel_table = self.training_batch[self.training_batch['type'] == 'testing']
        x_test_set, y_test_set = load_test_data(self.training_data_path, parallel_table)
        
        results = model.evaluate(x_test_set, y_test_set, verbose=2)

        connet_logger.info (results)

        connet_logger.info(f"####################################")
        connet_logger.info(f"##########Evaluation End############")
        connet_logger.info(f"####################################")

if __name__ == '__main__':
    import argparse
    """
    
    """
    parser = argparse.ArgumentParser(description='Train A Bidirectional Model')
    parser.add_argument('-parallel')
    parser.add_argument('-train')
    parser.add_argument('-o')
    parser.add_argument('-epochs', default=10)
    parser.add_argument('-mode')
    parser.add_argument('-phase', default="correction")

    args = parser.parse_args()
    train_file_list = glob.glob(f'{args.train}/*_training_*')
    test_file_list = glob.glob(f'{args.train}/*_testing_*')
    df = pd.read_csv(args.parallel, sep='\s', names=["path", "ctg", "name",  "start", "end", "ref_start", "ref_end"])
    df['type'] = ''
    df.loc[df.apply(lambda row:check_training_file(row, args.train, train_file_list), axis=1), 'type'] = 'training'
    df.loc[df.apply(lambda row:check_testing_file(row, args.train, test_file_list), axis=1), 'type'] = 'testing'
    
    if args.phase == "correction":
        model = Correction(args.train, df, args.o)
    elif args.phase == "recovery":
        model = Recovery(args.train, df, args.o)

    if args.mode == "evaluate":
        model.evaluate()
    else:
        model.train(int(args.epochs))
        model.evaluate()
