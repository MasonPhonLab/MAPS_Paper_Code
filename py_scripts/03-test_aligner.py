from textgrid import textgrid
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation
from scipy.io import wavfile
import numpy as np
import os
from scipy.signal import medfilt
from julia import Main
from tqdm import tqdm
import statistics
import tensorflow as tf
import python_speech_features as psf
import scipy as sp

MODEL_DIR = r'D:\alignerv2\timbuck_trained_models_repetitions'
FRAME_LENGTH = 0.025 # 25 ms expressed as seconds
FRAME_INTERVAL = 0.01 # 10 ms expressed as seconds

TIMIT_DIR = r'D:\TIMIT'
BUCK_DIR = r'D:alignerv2\buck_out'
BUCK_DIR_S04 = r'D:\alignerv2\buck_out_s04'

# Flags for which evaluation to run
ALIGN_DIR = r'D:/timbuck_data/timbuck10_train' # train, test, or val set
MODEL_TYPE = 'CRISP' # crisp or sparse model

# When True, will overwrite evaluation files already made, and won't skip anything
#
# When False, will not overwrite existing files and will skip evaluating those
# files again
OVERWRITE=False
    

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

phones = 'h#	q	eh	dx	iy	r	ey	ix	tcl	sh	ow	z	s	hh	aw	m	t	er	l	w	aa	hv	ae	dcl	y	axr	d	kcl	k	ux	ng	gcl	g	ao	epi	ih	p	ay	v	n	f	jh	ax	en	oy	dh	pcl	ah	bcl	el	zh	uw	pau	b	uh	th	ax-h	em	ch	nx	eng'.split()

num2phn = {i: p for i, p in enumerate(phones)}
phn2num = {p: i for i, p in enumerate(phones)}

class PhoneLabel:

    def __init__(self, phone, duration):
    
        self.phone = phone
        self.duration = duration
        
    def __str__(self):
    
        return str([self.phone, self.duration])
    
def collapse(s):
    a = [s[0]]
    for symbol in s[1:]:
        if a[-1] != symbol:
            a.append(symbol)
    return a
    
def force_align(y, yhat):

    yhat = np.squeeze(yhat, 0)
    predictions = np.abs(np.log(yhat))
    seq = np.argmax(y, 1)
    seq = np.array(collapse(seq))
    a, M = Main.dtw_align(seq, predictions, False)
    a = [num2phn[p] for p in a]
    seq = [PhoneLabel(phone=a[0], duration=1)]
    durI = 1
    for elem in a[1:]:
        if not seq[-1].phone == elem:
            pl = PhoneLabel(phone=elem, duration=1)
            seq.append(pl)
        else:
            seq[-1].duration += 1
    return seq, M
    
def make_textgrid(seq, tgname, maxTime, interpolate=True, symm=True, probs=None):
    '''
    Side-effect of writing TextGrid to disk
    '''
    
    if interpolate and np.all(probs == None):
    
        raise ValueError('If using interpolation, the alignment matrix must also be passed in through the probs argument')
        
    
    tg = textgrid.TextGrid()
    tier = textgrid.IntervalTier()
    tier.name = 'phones'
    curr_dur = 0
    
    if len(seq) == 1:
        last_interval = textgrid.Interval(curr_dur, maxTime, seq[-1].phone)
        tier.intervals.append(last_interval)
        tg.tiers.append(tier)
        tg.write(tgname)
        return
    
    added_bits = []
    frame_durs = [s.duration for s in seq]
    cumu_frame_durs = [sum(frame_durs[0:i+1]) for i in range(len(frame_durs))]
    curr_dur = seq[0].duration * FRAME_INTERVAL + 0.015
    
    if interpolate:
    
        additional = interpolated_part(seq[0].duration-1, 0, probs, symm=symm)
        if curr_dur + additional < maxTime:
            curr_dur += additional
        added_bits.append(additional)
    
    tier.intervals.append(textgrid.Interval(0, curr_dur, seq[0].phone))

    for i, s in enumerate(seq[:-1]):
    
        if i == 0: continue
    
        label = s.phone
        duration = s.duration
    
        beginning = curr_dur
        dur = FRAME_INTERVAL * duration
        
        if interpolate:
        
            endCur = cumu_frame_durs[i] - 1
        
            dur -= added_bits[-1]
            additional = interpolated_part(endCur, i, probs, symm=symm)
            if beginning + dur + additional < maxTime:
                dur += additional
            added_bits.append(additional)
        
        ending = beginning + dur
        
        interval = textgrid.Interval(beginning, ending, label)
        tier.intervals.append(interval)
        
        curr_dur = ending
    
    last_interval = textgrid.Interval(curr_dur, maxTime, seq[-1].phone)
    tier.intervals.append(last_interval)
    tg.tiers.append(tier)
    tg.write(tgname)
    
def interpolated_part(endCur, phone_n, probs, symm=True):

    # symm=True: use flanking points to perform interpolation
    #   this functionality is relativel untested at this time,
    #   but keeping interface so that the record of what code was used
    #   is maintained
    #
    # symm=False: use point in question and following point to perform  
    #   interpolation
    

    phone1_curr = probs[endCur, phone_n]
    phone1_next = probs[endCur+1, phone_n]

    phone2_curr = probs[endCur, phone_n+1]
    phone2_next = probs[endCur+1, phone_n+1]
        
    if symm and endCur > 1:
        m1 = (probs[endCur + 1, phone_n] - probs[endCur - 1, phone_n]) / (FRAME_INTERVAL)
        m2 = (probs[endCur + 1, phone_n+1] - probs[endCur - 1, phone_n+1]) / (FRAME_INTERVAL)
        
    else:
        m1 = (phone1_next - phone1_curr) / FRAME_INTERVAL
        m2 = (phone2_next - phone2_curr) / FRAME_INTERVAL
    
    A = np.array([[-m1, 1], [-m2, 1]])
    b = [phone1_curr, phone2_curr]
    
    try:
        time_point, intersection_probability = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 0
    
    curr_dur = seq[0].duration * FRAME_INTERVAL + 0.015
    if 0 <= time_point < FRAME_INTERVAL:
        return time_point
        
    return 0
    
    
def prediction2tg(fname, maxTime):
    
    d = np.load(fname)
    phone_sequence = get_phone_sequence(d)
    
    cumudur = 0
    
    for p in phone_sequence:
    
        cumudur += FRAME_LENGTH + FRAME_INTERVAL * (p.duration - 1 )
    
    tgname = os.path.splitext(fname)[0] + '.TextGrid'
    make_textgrid(phone_sequence, tgname, maxTime)
    
Main.include("../jl_scripts/dtw_align.jl")

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
        
def make_batch(data):

    pad_to_size = max([d.shape[0] for d in data])
    
    for i, tensor in enumerate(data):
    
        to_pad = pad_to_size - tensor.shape[0]
        
        if to_pad > 0:
        
            width = ((0, to_pad), (0, 0))
            tensor = np.pad(tensor, pad_width=width, mode='constant', constant_values=0)
            
        data[i] = tensor
        
        
    return np.stack(data, 0)

if __name__ == '__main__':

    if MODEL_TYPE == 'CRISP':
        MODELS = CRISP_MODELS
    elif MODEL_TYPE == 'SPARSE':
        MODELS = SPARSE_MODELS
    else:
        raise ValueError('MODEL_TYPE variable can only take "CRISP" or "SPARSE" as its value.')
        
    if not OVERWRITE:
        print('Collecting existing TextGrid names...')
        already_present_textgrids = set(x for x in tqdm(os.listdir(ALIGN_DIR)) if x.endswith('TextGrid'))
    else:
        already_present_textgrids = set()

        
    for i, mod_name in enumerate(MODELS):
    
        base_name = os.path.splitext(os.path.basename(mod_name))[0]
    
        MODEL = load_model(mod_name, compile=False)
        
        print()
        print('.......MAKING TEXTGRIDS FOR {} MODEL {}.......'.format(MODEL_TYPE, i+1))
        print()
        
        fname2wavname = dict()
        
        for root, _, files in os.walk(TIMIT_DIR):
            for x in [x for x in files if x.endswith('.WAV')]:
                pathname = os.path.join(root, x)
                fname = os.path.basename(root) + x
                fname = fname.replace('.WAV', '.npy')
                fname2wavname[fname] = pathname
                
        for root, _, files in os.walk(BUCK_DIR):
            for x in [x for x in files if x.endswith('.wav')]:
                pathname = os.path.join(root, x)
                x = x.replace('.wav', '.npy')
                fname2wavname[x] = pathname
                
        for root, _, files in os.walk(BUCK_DIR_S04):
            for x in [x for x in files if x.endswith('.wav')]:
                pathname = os.path.join(root, x)
                x = x.replace('.wav', '.npy')
                fname2wavname[x] = pathname
                
        for fname in tqdm([x for x in os.listdir(ALIGN_DIR) if not 'labs' in x and not 'multi' in x and x.endswith('.npy')]):
        
            tgname = os.path.basename(fname)
            tgname = os.path.splitext(tgname)[0]
            tgname_interp = tgname + base_name + '_interp.TextGrid'
            tgname_noint = tgname + base_name + '_noint.TextGrid'
            
            if (tgname_interp in already_present_textgrids) and (tgname_noint in already_present_textgrids) and not OVERWRITE: continue
            elif already_present_textgrids: already_present_textgrids = set()
            
            wavname = fname2wavname[fname]
        
            sr, samples = wavfile.read(wavname)
            duration = samples.size / sr # convert samples to seconds
            
            labname = fname.replace('.npy', '') + '_labs.npy'
            
            fname = os.path.join(ALIGN_DIR, fname)
            labname = os.path.join(ALIGN_DIR, labname)
            
            mfcc = psf.mfcc(samples, sr, winstep=FRAME_INTERVAL)
            delta = psf.delta(mfcc, 2)
            deltadelta = psf.delta(delta, 2)
            
            x = np.hstack((mfcc, delta, deltadelta))
            x = np.expand_dims(x, axis=0)
            yhat = MODEL.predict(x, verbose=0)
            
            seq, M = force_align(np.load(labname), yhat)
            
            tgname = os.path.basename(fname)
            tgname = os.path.splitext(tgname)[0]
            tgname_interp = tgname + base_name + '_interp.TextGrid'
            tgname_noint = tgname + base_name + '_noint.TextGrid'
            tgpath_interp = os.path.join(ALIGN_DIR, tgname_interp)
            tgpath_noint = os.path.join(ALIGN_DIR, tgname_noint)

            make_textgrid(seq, tgpath_interp, duration, interpolate=True, symm=False, probs=M.T)
            make_textgrid(seq, tgpath_noint, duration, interpolate=False, symm=False, probs=M.T)
            