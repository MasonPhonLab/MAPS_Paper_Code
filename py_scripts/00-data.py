from scipy.io import wavfile
import os
import numpy as np
import keras
import sys
import textgrid
import re
import glob
from scipy.signal import hamming
from tqdm import tqdm
import python_speech_features as psf
import shutil
import random
import math
from tqdm import tqdm


traindir = r'C:\Users\mckelley\buckeye\TIMIT\TRAIN'
testdir = r'C:\Users\mckelley\buckeye\TIMIT\TEST'
buckeye_dir = r'C:\Users\mckelley\alignerv2\buck_out'
timdir = r'C:\Users\mckelley\buckeye\TIMIT\TRAIN'
tgoutdir = r'E:\timbuck_textgrids'

ERRLOG = 'errlog.txt'

FRAME_LENGTH = 0.025
FRAME_INTERVAL = 0.01

phones = 'h#	q	eh	dx	iy	r	ey	ix	tcl	sh	ow	z	s	hh	aw	m	t	er	l	w	aa	hv	ae	dcl	y	axr	d	kcl	k	ux	ng	gcl	g	ao	epi	ih	p	ay	v	n	f	jh	ax	en	oy	dh	pcl	ah	bcl	el	zh	uw	pau	b	uh	th	ax-h	em	ch	nx	eng'.split()
phn2num = {p: i for i, p in enumerate(phones)}
phn2num['sil'] = phn2num['h#']

foldings = {
    'ao': 'aa',
    'ax': 'ah',
    'ax-h': 'ah',
    'axr': 'er',
    'hv': 'hh',
    'ix': 'ih',
    'el': 'l',
    'em': 'm',
    'en': 'n',
    'nx': 'n',
    'eng': 'ng',
    'zh': 'sh',
    'pcl': 'p',
    'tcl': 't',
    'kcl': 'k',
    'bcl': 'b',
    'dcl': 'd',
    'gcl': 'g',
    'h#': 'sil',
    'pau': 'sil',
    'epi': 'sil',
    'ux': 'uw'        
}

def tg2boundaries(sr, fname, tierN=0):

    # print(fname)

    tg = textgrid.TextGrid()
    try:
        tg.read(fname)
    except ValueError:
    
        with open(ERRLOG, 'a') as w:
        
            w.write('Interval error in {}\n'.format(fname))
        
        return None, None
    
    tier = tg.tiers[tierN]
    
    boundaries = list()
    labels = list()
    
    for interval in tier.intervals:
    
        end_time = interval.maxTime
        end_time = end_time * sr
        
        label = interval.mark
        
        boundaries.append(end_time)
        label = re.sub(r'[0-9]', '', label)
        label = label.replace('sp', 'sil')
        labels.append(label.lower())
        
    return boundaries, labels

def calc_n_windows(x, sr):

    # requires x to be evenly divisibly by 16 (or whatever the number of samples per millisecond are)
    # 16 samples per millisecond
    # -24 to offset (last 24 milliseconds will only be included in the final window)
    # y = x // 16 - 24
    frameIntervalSamples = FRAME_INTERVAL * sr
    # y = x // 160 - 24
    y = x // frameIntervalSamples - 24
    return int(y)
    
def calculate_frames(sr, samples, apply_hamming=False):

    samples = np.reshape(samples, samples.shape[0])
    frameIntervalSamples = FRAME_INTERVAL * sr
    frameLengthSamples = FRAME_LENGTH * sr
    needed_to_pad = int(samples.shape[0] % frameIntervalSamples)
            
    if needed_to_pad > 0:
        samples = np.append(samples, np.zeros(needed_to_pad))
            
    windows = list()
            
    for i in range(calc_n_windows(samples.shape[0], sr)):
    
        start_samp = int(i * frameIntervalSamples)
        end_samp = int(start_samp + frameLengthSamples)
        
        window = samples[start_samp:end_samp]
        if apply_hamming:
            window = window * hamming(len(window))
        windows.append(window)
        
    return windows
    
def offset_beginning(windows, labels):

    offset = np.zeros((12, windows.shape[1]))
    windows = np.vstack((offset, windows))
    labels = [labels[0] for x in range(12)] + labels
    return windows, labels

def make_data(datadir, outdir):
    
    for root, _, files in os.walk(datadir):
        
        print(root, end='\r')

        fnames = [x for x in files if x.endswith('.WAV')]
        one_up = os.path.basename(root)
        
        for fname in fnames:
            
            sr, samples = wavfile.read(os.path.join(root, fname))
            windows = psf.mfcc(samples, sr)
            boundaries = list()
            labels = list()
            
            phn_name = os.path.join(root, os.path.splitext(fname)[0] + '.PHN')
            
            with open(phn_name, 'r') as f:                
                lines = f.readlines()

            # first field in the file is the beginning sample number, which isn't
            # needed for calculating where the labels are
            for line in lines:
                _, boundary, label = line.split()
                boundary = int(boundary)
                boundaries.append(boundary)
                labels.append(label)

            labelInfo = list(zip(boundaries, labels))
            labelInfoIdx = 0
            boundary, label = labelInfo[labelInfoIdx]
            nSegments = len(labelInfo)

            frameLengthSamples = FRAME_LENGTH * sr
            frameIntervalSamples = FRAME_INTERVAL * sr
            halfFrameLength = FRAME_LENGTH / 2

            # Begin generating sequence labels by looping through the acoustic
            # sample numbers

            labelSequence = list() # Holds the sequence of labels

            idxsToDelete = list() # To store indices for frames labeled as 'q'
            for i, window in enumerate(windows):
                win_end = frameLengthSamples + (i)*frameIntervalSamples

                # Move on to next label if current frame of samples is more than half
                # way into next labeled section and there are still more labels to
                # iterate through
                if labelInfoIdx < nSegments - 1 and win_end - boundary > halfFrameLength:

                    labelInfoIdx += 1
                    boundary, label = labelInfo[labelInfoIdx]

                labelSequence.append(label)
            
            
            deltas = psf.delta(windows, 2)
            deltadeltas = psf.delta(deltas, 2)
            feats = np.hstack((windows, deltas, deltadeltas))
            
            labelSequence = [foldings[label] if label in foldings else label for label in labelSequence]
            labelSequence = [phn2num[label] for label in labelSequence]
            
            labelSequence = keras.utils.np_utils.to_categorical(labelSequence, num_classes=61)
            
            newname = one_up + os.path.splitext(fname)[0] + '.npy'
            np.save(os.path.join(outdir, newname), feats)
            
            newname = one_up + os.path.splitext(fname)[0] + '_labs.npy'
            np.save(os.path.join(outdir, newname), labelSequence)
            
def buckeye_data(datadir, outdir):

    TRANSLATION = {
    'a':'ah',
    'aan':'aa',
    'aen':'ae',
    'ahn':'ah',
    'aon':'ao',
    'awn':'aw',
    'ayn':'ay',
    'ehn':'eh',
    'el':'l',
    'em':'m',
    'en':'n',
    'eng':'ng',
    'er':'r',
    'ern':'r',
    'eyn':'ey',
    'h':'hh',
    'hhn':'hh',
    'ihn':'ih',
    'iyn':'iy',
    'nx':'n',
    'own':'ow',
    'oyn':'oy',
    'tq':'t',
    'uhn':'uh',
    'uwn':'uw',
    '<sil>':'sil'
    }

    IGNORE = set([
    '<EXCLUDE-NAME>',
    '<EXCLUDE>',
    'EXCLUDE',
    '<EXCLUDE>',
    'IVER',
    '<IVER>',
    'IVER-LAUGH',
    '<IVER-LAUGH>',
    'LAUGH',
    '<LAUGH>',
    'NOISE',
    '<NOISE>',
    'UNKNOWN',
    '<UNKNOWN>',
    'VOCNOISE',
    '<VOCNOISE>',
    '{B_TRANS}',
    '{E_TRANS}',
    '',
    ])

    outset = set(os.listdir(outdir))
    
    dirs = [datadir]
    speakers_to_exclude = set(['s27', 's38', 's39', 's40'])

    for root in dirs:
    
        one_up = os.path.basename(root)
        
        fnames = glob.glob(os.path.join(root, '*.wav'))
        
        for fname in tqdm(fnames):
        
            sr, samples = wavfile.read(os.path.join(root, fname))
            if not sr == 16000:
                print('{} not sr=16000'.format(os.path.join(root, fname)))
                sys.exit()
            windows = psf.mfcc(samples, sr)
            
            tgname, _ = os.path.splitext(fname)
            tgname += '.TextGrid'
            
            if not os.path.isfile(os.path.join(root, tgname)):
            
                continue
            
            boundaries, labels = tg2boundaries(sr, os.path.join(root, tgname), 1)
            if boundaries == labels == None:
            
                continue
            
            labelInfo = list(zip(boundaries, labels))
            if not labelInfo: continue
            
            labelInfoIdx = 0            
            boundary, label = labelInfo[labelInfoIdx]            
            nSegments = len(labelInfo)

            frameLengthSamples = FRAME_LENGTH * sr
            frameIntervalSamples = FRAME_INTERVAL * sr
            halfFrameLength = FRAME_LENGTH / 2

            # Begin generating sequence labels by looping through the acoustic
            # sample numbers

            labelSequence = list() # Holds the sequence of labels

            idxsToDelete = list() # To store indices for frames to ignore
            for i, window in enumerate(windows):
                win_end = frameLengthSamples + (i)*frameIntervalSamples

                # Move on to next label if current frame of samples is more than half
                # way into next labeled section and there are still more labels to
                # iterate through
                if labelInfoIdx < nSegments - 1 and win_end - boundary > halfFrameLength:

                    labelInfoIdx += 1
                    boundary, label = labelInfo[labelInfoIdx]

                label = label.replace(';', '').replace('+1', '').replace('+', '')
                if label in TRANSLATION:
                    label = TRANSLATION[label]
                if label.upper() in IGNORE:
                    idxsToDelete.append(i)
                else:
                    labelSequence.append(label)
            
            windows = np.delete(windows, idxsToDelete, axis=0)
            
            deltas = psf.delta(windows, 2)
            deltadeltas = psf.delta(deltas, 2)
            feats = np.hstack((windows, deltas, deltadeltas))
            
            # one_up variable that is used in other fcns not needed here because Buckeye
            # naming conventions already make each filename unique
            newname = os.path.splitext(os.path.basename(fname))[0] + '.npy'
            np.save(os.path.join(outdir, newname), feats)
            
            labelSequence = [foldings[label] if label in foldings else label for label in labelSequence]
            labelSequence = [phn2num[label] for label in labelSequence]
            labelSequence = keras.utils.to_categorical(labelSequence, num_classes=61)
            
            newname = os.path.splitext(os.path.basename(fname))[0] + '_labs.npy'
            np.save(os.path.join(outdir, newname), labelSequence)
                
def make_buckeye_testset(datadir, testdir, speakers_to_move):

    for fname in os.listdir(datadir):
    
        if any([fname.startswith(x) for x in speakers_to_move]):
        
            currpath = os.path.join(datadir, fname)
            movepath = os.path.join(testdir, fname)
            shutil.move(currpath, movepath)
            
def make_timbuck_val(datadir, valdir, valsize=0.05):

    fnames = [x for x in os.listdir(datadir) if '_labs' not in x]
    nVal = int(math.ceil(valsize * len(fnames)))
    random.seed(20200622)
    random.shuffle(fnames)
    fnames = fnames[:nVal]
    
    for xname in tqdm(fnames):
    
        curr_x_path = os.path.join(datadir, xname)
        yname = os.path.splitext(xname)[0] + '_labs.npy'
        curr_y_path = os.path.join(datadir, yname)
        
        val_x_path = os.path.join(valdir, xname)
        val_y_path = os.path.join(valdir, yname)
        
        shutil.move(curr_x_path, val_x_path)
        shutil.move(curr_y_path, val_y_path)
        
# Convenience function for creating TextGrids for TIMIT data annotations
def timit2tg():

    for root, _, files in os.walk(timdir):
    
        print(root)
        oneup = os.path.basename(root)
    
        for fname in [x for x in files if x.endswith('.PHN') and 'BUCK' not in x]:
        
            tg = textgrid.textgrid.TextGrid()
            tier = textgrid.textgrid.IntervalTier()
            tier.name = 'phones'
            curr_dur = 0
            
            sr = 16000
            
            fname = os.path.join(root, fname)
            
            with open(fname, 'r') as f:
            
                for line in f.readlines():
                
                    begin, end, phone = line.split()
                    begin = int(begin) / sr
                    end = int(end) / sr
                    
                    if phone in foldings: phone = foldings[phone]
                    
                    if tier.intervals and tier.intervals[-1].mark == phone:
                    
                        tier.intervals[-1].maxTime = end
                        
                    else:
                        interval = textgrid.textgrid.Interval(begin, end, phone)
                        tier.intervals.append(interval)
           
            fname = os.path.basename(fname)
            tgname = oneup + os.path.splitext(fname)[0] + '.TextGrid'
            tgname = os.path.join(tgoutdir, tgname)
            tg.tiers.append(tier)
            tg.write(tgname)
            

if __name__ == '__main__':
            
    make_data(traindir, r'F:\matt\timbuck10_train')
    make_data(testdir, r'F:\matt\timbuck10_test')
    buckeye_data(buckeye_dir, r'C:\Users\mckelley\alignerv2\data\buck_phrases_mfcc')
    # 27: old female speaker
    # 38: old male speaker
    # 39: young female speaker
    # 40: young male speaker
    make_buckeye_testset(r'C:\Users\mckelley\alignerv2\data\buck_phrases_mfcc', r'C:\Users\mckelley\alignerv2\data\buck_phrases_test', ['s27', 's38', 's39', 's40'])
    make_timbuck_val(r'C:\Users\mckelley\alignerv2\data\timbuck10_train', r'C:\Users\mckelley\alignerv2\data\timbuck10_val', valsize=0.05)
    
    # add buckeye s04 to validation data since it wasn't originally extracted in Praat and re-extracting
    # re-training for Buckeye would be painful; plus, this gives us a holdout speaker in the validation set
    buckeye_data('C:/Users/mckelley/alignerv2/buck_out_s04', r'C:\Users\mckelley\alignerv2\data\timbuck10_val')
    timit2tg()
    