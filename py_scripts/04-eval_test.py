from textgrid import textgrid
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation
from scipy.io import wavfile
import numpy as np
import os
from tqdm import tqdm
import statistics
import tensorflow as tf
import python_speech_features as psf
import scipy as sp
from wbce import weighted_binary_crossentropy

TEST_DIR = 'F:/matt/timbuck10_test'
MODEL_DIR = 'timbuck_trained_models_repetitions'
tf.keras.utils.set_random_seed(20220825)

POS_WEIGHT = 33
RECREATE_LABELS = True


CRISP_MODELS = [
    os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd1_bs64_epoch19.tf'),
    os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd2_bs64_epoch43.tf'),
    os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd3_bs64_epoch24.tf'),
    os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd4_bs64_epoch27.tf'),
    os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd5_bs64_epoch31.tf'),
    os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd6_bs64_epoch36.tf'),
    os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd7_bs64_epoch29.tf'),
    os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd8_bs64_epoch19.tf'),
    os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd9_bs64_epoch29.tf'),
    os.path.join(MODEL_DIR, 'real_seed_crisp_3blstm_128units_rd10_bs64_epoch28.tf'),
]

SPARSE_MODELS = [
    os.path.join(MODEL_DIR, 'full_real_seed_multiposthoc_3blstm_128units_bs64_rd1_epoch29.tf'),
    os.path.join(MODEL_DIR, 'full_real_seed_multiposthoc_3blstm_128units_bs64_rd2_epoch20.tf'),
    os.path.join(MODEL_DIR, 'full_real_seed_multiposthoc_3blstm_128units_bs64_rd3_epoch29.tf'),
    os.path.join(MODEL_DIR, 'full_real_seed_multiposthoc_3blstm_128units_bs64_rd4_epoch12.tf'),
    os.path.join(MODEL_DIR, 'full_real_seed_multiposthoc_3blstm_128units_bs64_rd5_epoch36.tf'),
    os.path.join(MODEL_DIR, 'full_real_seed_multiposthoc_3blstm_128units_bs64_rd6_epoch34.tf'),
    os.path.join(MODEL_DIR, 'full_real_seed_multiposthoc_3blstm_128units_bs64_rd7_epoch22.tf'),
    os.path.join(MODEL_DIR, 'full_real_seed_multiposthoc_3blstm_128units_bs64_rd8_epoch16.tf'),
    os.path.join(MODEL_DIR, 'full_real_seed_multiposthoc_3blstm_128units_bs64_rd9_epoch32.tf'),
    os.path.join(MODEL_DIR, 'full_real_seed_multiposthoc_3blstm_128units_bs64_rd10_epoch23.tf'),
]

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
        # Ys = [onehot2acoustic(y) for y in Ys]
        
        X_batch = make_batch(Xs)
        Y_batch = make_batch(Ys)
        
        return X_batch, Y_batch
        
def make_batch(data):

    pad_to_size = max([d.shape[0] for d in data])
    
    for i, tensor in enumerate(data):
    
        to_pad = pad_to_size - tensor.shape[0]
        
        if to_pad > 0:
        
            width = ((0, to_pad), (0, 0))
            tensor = np.pad(tensor, pad_width=width, mode='constant', constant_values=0)
            
        data[i] = tensor
        
        
    return np.stack(data, 0)
    
def gather_sparse_metrics(h):
    
    loss = h[0]
    acc = h[2]
    binacc = h[1]
    
    tp = h[5]
    tn = h[6]
    fp = h[3]
    fn = h[4]
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    balacc = (sens + spec) / 2
    
    return {'loss': loss, 'sens': sens, 'spec': spec, 'balacc': balacc}
    
def recreate_labels(mod, xnames, ynames, i):

    crisp_ynames = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith('_labs.npy')]
    
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
        
def main():

    ynames = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith('_labs.npy')]
    xnames = [x.replace('_labs.npy', '.npy') for x in ynames]
    
    data = SpeechSequence(xnames, ynames, 1)
    
    with open('crisp_test_res.txt', 'a') as w:
    
        w.write('\t'.join(['model_name', 'loss', 'accuracy']))
        w.write('\n')
    
        for mod_name in CRISP_MODELS:
        
            MODEL = load_model(mod_name, compile=False)
            MODEL.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            h = MODEL.evaluate(data)
            
            w.write('\t'.join([mod_name, str(h[0]), str(h[1])]))
            w.write('\n')
            
    with open('sparse_test_res.txt', 'a') as w:
    
        w.write('\t'.join(['model_name', 'loss', 'sens', 'spec', 'balacc']))
        w.write('\n')
        
        for i, mod_name, recalc_name in zip(range(1, 11), SPARSE_MODELS, CRISP_MODELS):
        
            if RECREATE_LABELS:
            
                MODEL = load_model(recalc_name, compile=False)
                recreate_labels(MODEL, xnames, ynames, i)
                ynames = [y.replace('_labs.npy', '_multiposthoc_sparse_rd{}.npy'.format(i)) for y in ynames]
                data = SpeechSequence(xnames, ynames, 1)
            
            MODEL = load_model(mod_name, compile=False)
            h = MODEL.compile(loss=weighted_binary_crossentropy, optimizer='adam', metrics=['binary_accuracy', 'accuracy', tf.keras.metrics.FalsePositives(name='false_positives'), tf.keras.metrics.FalseNegatives(name='false_negatives'), tf.keras.metrics.TruePositives(name='true_positives'), tf.keras.metrics.TrueNegatives(name='true_negatives')])
            
            h = MODEL.evaluate(data, verbose=2)
            
            metrics = gather_sparse_metrics(h)
            w.write('\t'.join([mod_name, str(metrics['loss']), str(metrics['sens']), str(metrics['spec']), str(metrics['balacc'])]))
            w.write('\n')

if __name__ == '__main__':
    main()