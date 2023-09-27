from __future__ import print_function, division
import numpy as np
import random
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Conv2D, Flatten, Reshape, MaxPooling1D, SimpleRNN, Bidirectional, Lambda, Permute, TimeDistributed, MaxPooling2D, LSTM, Masking, LayerNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from tensorflow.keras import regularizers
from sklearn.preprocessing import scale
import math, h5py, random, os, pickle
from math import ceil
from scipy.io import wavfile
from tensorflow import keras
import python_speech_features as psf
from tqdm import tqdm
from collections import namedtuple
from tensorflow import nn
import timeit
from tensorflow.keras.utils import to_categorical
import statistics
from scipy.signal import medfilt
import sys

my_rng = random.Random(30420)
MAX_SEED = 2**32 - 1
SEEDS = [my_rng.randint(0, MAX_SEED) for x in range(10)]

DATA_DIR = r'F:\matt\timbuck10_train'
VAL_DIR = r'F:\matt\timbuck10_val'
BATCH_SIZE = 64
EPOCHS = 50
VAL_SIZE = 0.05 # 5% of training data
MODEL_DIR = 'timbuck_trained_models_repetitions'
MODEL_BASENAME = 'crisp_3blstm_128units_bs64'

def get_sequence(yname):

    y = np.load(os.path.join(DATA_DIR, yname))
    y = np.argmax(y, 1)
    
    s = [y[0]]
    for yI in y[1:]:
        if not yI == s[-1]:
            s.append(yI)
    return s
    
def make_batch(data):

    pad_to_size = max([d.shape[0] for d in data])
    
    for i, tensor in enumerate(data):
    
        to_pad = pad_to_size - tensor.shape[0]
        
        if to_pad > 0:
        
            width = ((0, to_pad), (0, 0))
            tensor = np.pad(tensor, pad_width=width, mode='constant', constant_values=-1)
            
        data[i] = tensor
        
        
    return np.stack(data, 0)
        
class SpeechSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
    
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        
    def __len__(self):
    
        return int(np.ceil(len(self.x) / float(self.batch_size)))
        
    def __getitem__(self, idx):
    
        start = idx * self.batch_size
        last = min(start + self.batch_size, len(self.x))
        Xs = [np.load(fname, allow_pickle=True) for fname in self.x[start:last]]
        Ys = [np.load(fname, allow_pickle=True) for fname in self.y[start:last]]
        
        X_batch = make_batch(Xs)
        Y_batch = make_batch(Ys)
        
        return X_batch, Y_batch         
        
def write_res(fname, epoch_label, h):

    with open(fname, 'a') as w:
    
        train_loss = h['loss'][0]
        train_acc = h['accuracy'][0]
        
        val_loss = h['val_loss'][0]
        val_acc = h['val_accuracy'][0]
        
        vals = [train_loss, train_acc, val_loss, val_acc]
        vals = [str(x) for x in vals]
        
        w.write('{}\t{}'.format(epoch_label, '\t'.join(vals)))
        w.write('\n')
        
    return
    
if __name__ == '__main__':

    for i, s in zip(range(10), SEEDS):
    
        tf.keras.utils.set_random_seed(s)
    
        print('BEGINNING ROUND {}'.format(i+1))
    
        RESULTS_NAME = 'real_seed_crisp_3blstm_128units_bs64_rd{}_res.txt'.format(i+1)
        MODEL_BASENAME = 'real_seed_crisp_3blstm_128units_rd{}_bs64'.format(i+1)

        with open(RESULTS_NAME, 'a') as w:
           
            w.write('\t'.join(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']))
            w.write('\n')

        xnames = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if not 'labs' in f and not 'mult' in f and f.endswith('npy')]
        ynames = [f.replace('.npy', '_labs.npy') for f in xnames]
        
        nVal = int(np.ceil(len(xnames) * VAL_SIZE))
        
        val_xnames = [os.path.join(VAL_DIR, f) for f in os.listdir(VAL_DIR) if not 'labs' in f and not 'mult' in f and f.endswith('npy')]
        val_ynames = [f.replace('.npy', '_labs.npy') for f in val_xnames]
        
        data_init = SpeechSequence(xnames, ynames, BATCH_SIZE)
        val_data = SpeechSequence(val_xnames, val_ynames, BATCH_SIZE)
        
        model = Sequential()
        model.add(Masking(-1))
        model.add(LayerNormalization())
        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5), input_shape=(None, 39)))
        model.add(LayerNormalization())
        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5)))
        model.add(LayerNormalization())
        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5)))
        model.add(LayerNormalization())
        model.add(TimeDistributed(Dense(units=61, activation='softmax')))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        for e in range(EPOCHS):
        
            print('EPOCH {}/{}'.format(e+1, EPOCHS))
        
            h = model.fit(data_init, epochs=1, validation_data=val_data).history
            write_res(RESULTS_NAME, str(e+1), h)
            model_name = MODEL_BASENAME + '_epoch{}.tf'.format(e+1)
            model.save(os.path.join(MODEL_DIR, model_name), save_format='tf')
            