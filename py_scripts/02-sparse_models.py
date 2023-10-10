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
import tensorflow.compat.v1.keras.backend as tfb
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, balanced_accuracy_score
import tensorflow_addons as tfa

from wbce import weighted_binary_crossentropy, POS_WEIGHT

my_rng = random.Random(220720)
MAX_SEED = 2**32 - 1
SEEDS = [my_rng.randint(0, MAX_SEED) for x in range(10)]

DATA_DIR = r'F:\matt\timbuck10_train'
VAL_DIR = r'F:\matt\timbuck10_val'
BATCH_SIZE = 64
EPOCHS = 50
VAL_SIZE = 0.05 # 5% of training data
MODEL_DIR = 'timbuck_trained_models_repetitions'
RECREATE_VAL_LABELS = True
RECREATE_LABELS = True

MODELS = {
1 : os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd1_bs64_epoch19.tf'),
2 : os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd2_bs64_epoch43.tf'),
3 : os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd3_bs64_epoch24.tf'),
4 : os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd4_bs64_epoch27.tf'),
5 : os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd5_bs64_epoch31.tf'),
6 : os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd6_bs64_epoch36.tf'),
7 : os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd7_bs64_epoch29.tf'),
8 : os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd8_bs64_epoch19.tf'),
9 : os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd9_bs64_epoch29.tf'),
10 : os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd10_bs64_epoch28.tf'),
}

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
        Xs = [np.load(fname) for fname in self.x[start:last]]
        Ys = [np.load(fname) for fname in self.y[start:last]]
        
        X_batch = make_batch(Xs)
        Y_batch = make_batch(Ys)
        
        return X_batch, Y_batch
    
def sk_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')
    
tp = tf.keras.metrics.TruePositives()
fp = tf.keras.metrics.FalsePositives()
tn = tf.keras.metrics.TrueNegatives()
fn = tf.keras.metrics.FalseNegatives()

tp1 = tf.keras.metrics.TruePositives()
fn1 = tf.keras.metrics.FalseNegatives()
def bal_acc(y_true, y_pred):
    tp_num = tp(y_true, y_pred)
    fp_num = fp(y_true, y_pred)
    tn_num = tn(y_true, y_pred)
    fn_num = fn(y_true, y_pred)
    
    sensitivity = tp_num / (tp_num + fn_num)
    selectivity = tn_num / (tn_num + fp_num)
    
    return (sensitivity + selectivity) / 2
    
def sensitivity(y_true, y_pred):
    tp_num = tp1(y_true, y_pred)
    fn_num = fn1(y_true, y_pred)
    
    return tp_num / (tp_num + fn_num)
    
class Reset_CB(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        tp.reset_state()
        fp.reset_state()
        tn.reset_state()
        fn.reset_state()
        
        tp1.reset_state()
        fn1.reset_state()
        
    def on_epoch_end(self, epoch, logs=None):
        tp.reset_state()
        fp.reset_state()
        tn.reset_state()
        fn.reset_state()
        
        tp1.reset_state()
        fn1.reset_state()         
        
def write_res(fname, epoch_label, h):

    with open(fname, 'a') as w:
    
        train_loss = h['loss'][0]
        train_acc = h['accuracy'][0]
        train_binacc = h['binary_accuracy'][0]
        
        train_tp = h['true_positives'][0]
        train_tn = h['true_negatives'][0]
        train_fp = h['false_positives'][0]
        train_fn = h['false_negatives'][0]
        train_sens = train_tp / (train_tp + train_fn)
        train_spec = train_tn / (train_tn + train_fp)
        train_balacc = (train_sens + train_spec) / 2
        
        val_loss = h['val_loss'][0]
        val_acc = h['val_accuracy'][0]
        val_binacc = h['val_binary_accuracy'][0]
        
        val_tp = h['val_true_positives'][0]
        val_tn = h['val_true_negatives'][0]
        val_fp = h['val_false_positives'][0]
        val_fn = h['val_false_negatives'][0]
        val_sens = val_tp / (val_tp + val_fn)
        val_spec = val_tn / (val_tn + val_fp)
        val_balacc = (val_sens + val_spec) / 2
        
        print('train sensitivity:\t{}'.format(round(train_sens, 3)))
        print('train specificity:\t{}'.format(round(train_spec, 3)))
        print('train balanced acc:\t{}'.format(round(train_balacc, 3)))
        
        print('val sensitivity:\t{}'.format(round(val_sens, 3)))
        print('val specificity:\t{}'.format(round(val_spec, 3)))
        print('val balanced acc:\t{}'.format(round(val_balacc, 3)))
        
        vals = [train_loss, train_binacc, train_acc, train_tp, train_fp, train_tn, train_fn, train_sens, train_spec, train_balacc, val_loss, val_binacc, val_acc, val_tp, val_fp, val_tn, val_fn, val_sens, val_spec, val_balacc]
        vals = [str(x) for x in vals]
        
        w.write('{}\t{}'.format(epoch_label, '\t'.join(vals)))
        w.write('\n')
        
    return
    
def recreate_labels(mod, xnames, ynames, i):

    crisp_ynames = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('_labs.npy')]
    
    for xn, yn in tqdm(list(zip(xnames, ynames))):
    
        x = np.load(xn)
        y = np.load(yn)
        x = np.expand_dims(x, 0)
        
        yhat = mod.predict(x, verbose=0)
        yhat = np.squeeze(yhat)
        if yhat.ndim == 1: yhat = np.expand_dims(yhat, 0)
        if y.ndim == 1: y = np.expand_dims(y, 0)
        y_seq = np.argmax(y, 1)
        
        if len(y_seq) > 0:
            for t, lab in enumerate(y_seq):
        
                yvec = yhat[t,:]
                targ_prob = yvec[lab]
                y[t,:] = np.where(yvec >= targ_prob, 1, 0)
            
        newyname = yn.replace('_labs.npy', '_multiposthoc_sparse_rd{}.npy'.format(i))
        np.save(newyname, y)
    
if __name__ == '__main__':

    for i, s in zip(range(1, 11), SEEDS):
    
        tf.keras.utils.set_random_seed(s)
    
        RESULTS_NAME  = 'full_real_seed_multiposthoc_3blstm_128units_bs64_rd{}_res.txt'.format(i)
        MODEL = load_model(MODELS[i], compile=False)
        MODEL_BASENAME = 'full_real_seed_multiposthoc_3blstm_128units_bs64_rd{}'.format(i)

        with open(RESULTS_NAME, 'a') as w:
           
            w.write('\t'.join(['epoch', 'train_loss', 'train_binacc', 'train_acc', 'train_tp', 'train_fp', 'train_tn', 'train_fn', 'train_sens', 'train_spec', 'train_balacc', 'val_loss', 'val_binacc', 'val_acc', 'val_tp', 'val_fp', 'val_tn', 'val_fn', 'val_sens', 'val_spec', 'val_balacc']))
            w.write('\n')

        ynames = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('_labs.npy')]
        xnames = [x.replace('_labs.npy', '.npy') for x in ynames]
        
        if RECREATE_LABELS:
        
            print('RECREATING TRAINING LABELS')
            recreate_labels(MODEL, xnames, ynames, i)
            
        ynames = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('_multiposthoc_sparse_rd{}.npy'.format(i))]
        
        nVal = int(np.ceil(len(xnames) * VAL_SIZE))
        
        val_ynames = [os.path.join(VAL_DIR, f) for f in os.listdir(VAL_DIR) if f.endswith('_labs.npy')]
        val_xnames = [x.replace('_labs.npy', '.npy') for x in val_ynames]
        
        if RECREATE_VAL_LABELS:
            
            print('RECREATING VALIDATION LABELS')
            recreate_labels(MODEL, val_xnames, val_ynames, i)
            
        val_ynames = [os.path.join(VAL_DIR, f) for f in os.listdir(VAL_DIR) if f.endswith('multiposthoc_sparse_rd{}.npy'.format(i))]
        
        data = PreSpeechSequence(xnames, ynames, BATCH_SIZE)
        data_init = SpeechSequence(xnames, ynames, BATCH_SIZE)
        val_data = SpeechSequence(val_xnames, val_ynames, BATCH_SIZE)

        MODEL = tf.keras.models.Sequential()
        MODEL.add(Masking(-1))
        MODEL.add(LayerNormalization())
        MODEL.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5), input_shape=(None, 39)))
        MODEL.add(LayerNormalization())
        MODEL.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5)))
        MODEL.add(LayerNormalization())
        MODEL.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5)))
        MODEL.add(LayerNormalization())
        MODEL.add(TimeDistributed(Dense(units=61, activation='sigmoid'), name='multilabel_output'))

        MODEL.compile(loss=weighted_binary_crossentropy, optimizer='adam', metrics=['binary_accuracy', 'accuracy', tf.keras.metrics.FalsePositives(name='false_positives'), tf.keras.metrics.FalseNegatives(name='false_negatives'), tf.keras.metrics.TruePositives(name='true_positives'), tf.keras.metrics.TrueNegatives(name='true_negatives')])
        
        for e in range(EPOCHS):
        
            print('EPOCH {}/{}'.format(e+1, EPOCHS))
        
            h = MODEL.fit(data_init, epochs=1, validation_data=val_data, verbose=2).history
            if e == 0: MODEL.summary()
            
            write_res(RESULTS_NAME, str(e+1), h)
            model_name = MODEL_BASENAME + '_epoch{}.tf'.format(e+1)
            MODEL.save(os.path.join(MODEL_DIR, model_name), save_format='tf')
            