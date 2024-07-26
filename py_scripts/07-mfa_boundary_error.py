import statistics
import math
import os
from tqdm import tqdm
from sklearn.metrics import classification_report
import re
import pandas as pd
from more_itertools import peekable
from textgrid import textgrid
import sys

# Change "train" to "val" or "test" as needed
ALIGN_DIR = 'D:/mfa_products/New_products/Aligned_textgrids/timbuck_train_aligned'
GOLD_DIR = 'D:/timbuck_data/timbuck_textgrids'

# Change "train"  to "val" or "test" as needed
RES_DIR = 'mfa_boundary_eval_res/train'
RES_NAME = 'mfa_resfile_train.txt'

SKIPPED = 0

TIMIT_TRANSLATION = {
    'er': 'r',
}

BUCKEYE_TRANSLATION = {
    'a':'ah',
    'aan':'aa',
    'aen':'ae',
    'ahn':'ah',
    # 'aon':'ao',
    'aon': 'aa', # collapse ao to aa from TIMIT collapsings
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
    # '<sil>':'sil',
    # 'SIL': 'sil',
    '<sil>': 'sil',
    'SIL': 'sil',
    'zh': 'sh', # carried over from TIMIT collapsings
    'ao': 'aa', # carried over from TIMIT collapsings
}

BUCKEYE_IGNORE = set([
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

FILES_TO_IGNORE = set([
    's0103a_58.TextGrid',
])
    
SILENCE_SYNONYMS = set([
    'h#',
    'sil',
    'sp',
    'SIL',
    ''
])
    
def compare_tg(tg_p, tg_g):

    errs = []
    tim_errs = []
    buck_errs = []
    
    predicted = textgrid.TextGrid()
    try:
        predicted.read(os.path.join(ALIGN_DIR, tg_p), round_digits=100)
    except:
        print()
        print(f'Error reading {tg_p}. Halting.')
        sys.exit()
        
    gold = textgrid.TextGrid()
    try:
        gold.read(os.path.join(GOLD_DIR, tg_g), round_digits=100)
    except:
        print()
        print(tg_p)
        print(tg_g)
        sys.exit()
        
    if os.path.basename(tg_g).startswith('s'):
        gt = convert_buckeye_tier(gold.tiers[-1])
    else:
        gt = convert_timit_tier(gold.tiers[-1])

    predicted.tiers[-1] = convert_mfa_tier(predicted.tiers[-1])
        
    if gt[-1].mark.lower() in SILENCE_SYNONYMS and predicted.tiers[-1][-1].mark.lower() not in SILENCE_SYNONYMS:
        del gt[-1]
    
    if gt[0].mark.lower() in SILENCE_SYNONYMS and predicted.tiers[-1][0].mark.lower() not in SILENCE_SYNONYMS:
        del gt[0]

    if len(predicted.tiers[-1]) != len(gt):
        errs = alt_tg_compare(predicted.tiers[-1], gt)
        if errs == -1:
            global SKIPPED
            SKIPPED += 1
            print()
            print(tg_p)
            print([x.mark for x in predicted.tiers[-1]])
            print(tg_g)
            print([x.mark for x in gt])
            if SKIPPED > 1:
                sys.exit()
            return {'all': [], 'tim': [], 'buck': []}
        else:
            if os.path.basename(tg_g).startswith('M') or os.path.basename(tg_g).startswith('F'):
                tim_errs += errs
            else:
                buck_errs += errs

    else:
        for p, g in zip(predicted.tiers[-1], gt):
                
                e = g.maxTime - p.maxTime
                errs.append(e)
                    
                if os.path.basename(tg_g).startswith('M') or os.path.basename(tg_g).startswith('F'):
                    tim_errs.append(e)
                else:
                    buck_errs.append(e)
                    
    segments = [x.mark for x in predicted.tiers[-1]]
                    
    return {'all': errs, 'tim': tim_errs, 'buck': buck_errs, 'segments': segments}
    
def convert_timit_tier(t):

    to_delete = set()
    
    for i, x in enumerate(t):
        if x.mark in TIMIT_TRANSLATION:
            t[i].mark = TIMIT_TRANSLATION[x.mark]
    
    for i in range(len(t)-1):
    
        curr_mark = t[i].mark
        next_mark = t[i+1].mark
        
        if curr_mark == next_mark or curr_mark == '':
            to_delete.add(i)
            
    if t[-1].mark == '':
        to_delete.add(len(t)-1)
        
    new_t = [x for i, x in enumerate(t) if not i in to_delete]
    return new_t

def convert_mfa_tier(t):

    to_delete = set()
    
    for i in range(len(t)-1):

        if i == 0 and t[i].mark in SILENCE_SYNONYMS:
            to_delete.add(i)
            continue
    
        curr_mark = t[i].mark
        next_mark = t[i+1].mark
        
        if curr_mark == next_mark or curr_mark == '' or curr_mark == 'sp':
            to_delete.add(i)
        elif curr_mark in TIMIT_TRANSLATION:
            t[i].mark = TIMIT_TRANSLATION[curr_mark]
        elif curr_mark in SILENCE_SYNONYMS:
            t[i].mark = 'sil'
            
    if t[-1].mark == '':
        to_delete.add(len(t)-1)
        
    new_t = [x for i, x in enumerate(t) if not i in to_delete]

    return convert_buckeye_tier(new_t)
    
def convert_buckeye_tier(t):

    to_delete = set()

    for i in range(len(t)-1):
    
        curr_mark = t[i].mark
        next_mark = t[i+1].mark

        curr_mark = curr_mark.replace(';', '').replace('+1', '')
        next_mark = next_mark.replace(';', '').replace('+1', '')
        
        if curr_mark in BUCKEYE_TRANSLATION: curr_mark = BUCKEYE_TRANSLATION[curr_mark]
        if next_mark in BUCKEYE_TRANSLATION: next_mark = BUCKEYE_TRANSLATION[next_mark]
        
        if curr_mark == next_mark or curr_mark in BUCKEYE_IGNORE:
        
            to_delete.add(i)
            
    if t[-1].mark in BUCKEYE_IGNORE:
        to_delete.add(len(t)-1)

    new_t = [x for i, x in enumerate(t) if not i in to_delete]
    for i, x in enumerate(new_t):
        x.mark = x.mark.replace(';', '').replace('+1', '')
        if x.mark in BUCKEYE_TRANSLATION:
            x.mark = BUCKEYE_TRANSLATION[x.mark]
        new_t[i].mark = x.mark
            
    return new_t

def alt_tg_compare(p, g):

    p_itr = peekable(p)
    g_itr = peekable(g)

    results = []

    p_cur = next(p_itr)
    g_cur = next(g_itr)

    if p_cur.mark == 'sil' and g_cur.mark != 'sil' and p_itr.peek().mark == g_cur.mark:
        p_cur = next(p_itr)

    while True:
        try:
            if p_cur.mark == g_cur.mark:
                results.append(g_cur.maxTime - p_cur.maxTime)
                p_cur = next(p_itr)
                g_cur = next(g_itr)
            elif p_itr.peek().mark == g_cur.mark:
                p_cur = next(p_itr)
            else:
                g_cur = next(g_itr)
        except StopIteration:
            break

    if len(results) == len(p):
        return results
    elif len(p) > len(results) and all(x.mark in SILENCE_SYNONYMS for x in p[len(results):]):
        return results
    else:
        print(p[len(results):])
        return -1

if __name__ == '__main__':

    fnames = os.listdir(ALIGN_DIR)
    fnames.sort()

    with open(RES_NAME, 'a') as w: w.write('----------\n')

    with open(RES_NAME, 'a') as w: w.write('beginning mfa\n')
    
    errs = []
    tim_errs = []
    buck_errs = []
    
    file_names = []
    segments = []
    previous_segments = []
    following_segments = []
    sources = []

    if 'train' in ALIGN_DIR:
        mfa_res = [os.path.join(root, f) for root, _, fnames in os.walk(ALIGN_DIR) for f in fnames if f.endswith('TextGrid')]
        tg_gold = [os.path.join(GOLD_DIR, f) for _, _, fnames in os.walk(ALIGN_DIR) for f in fnames if f.endswith('TextGrid')]
    else:
        mfa_res = os.listdir(ALIGN_DIR)
        tg_gold = [os.path.join(GOLD_DIR, x) for x in mfa_res]
        mfa_res = [os.path.join(ALIGN_DIR, x) for x in mfa_res]
    
    skipped = 0
    print(len(mfa_res), len(tg_gold))
    
    for mfa, gold in tqdm(list(zip(mfa_res, tg_gold))):
    
        mfa_comp = compare_tg(mfa, gold)
        
        if len(mfa_comp['all']) != len(mfa_comp['segments']):
            print(os.path.basename(mfa))
            print(len(mfa_comp['all']), [round(x, 3) for x in mfa_comp['all']])
            print(len(mfa_comp['segments']), mfa_comp['segments'])
            g_tg = textgrid.TextGrid()
            g_tg.read(gold, round_digits=100)
            print(len(g_tg.tiers[-1]), [x.mark for x in g_tg.tiers[-1]])
            g_tg = convert_timit_tier(g_tg.tiers[-1])
            print(len(g_tg), [x.mark for x in g_tg])
            sys.exit()
        
        errs += mfa_comp['all']
        tim_errs += mfa_comp['tim']
        buck_errs += mfa_comp['buck']
        
        file_names += [os.path.basename(mfa)] * len(mfa_comp['all'])
        segments += mfa_comp['segments']
        previous_segments += ['#'] + mfa_comp['segments'][:-1]
        following_segments += mfa_comp['segments'][1:] + ['#']
        sources += ['buckeye' if os.path.basename(mfa).startswith('s') else 'timit'] * len(mfa_comp['all'])
    
    print()
    print('Skipped:\t', SKIPPED)
    SKIPPED = 0
    
    all_df_name = os.path.join(RES_DIR, 'mfa_all_res.csv')
    all_df = pd.DataFrame({'err': errs, 'filename': file_names, 'segment': segments, 'prev_segment': previous_segments, 'next_segment': following_segments, 'corpus': sources})
    all_df.to_csv(all_df_name, index=False)
    
    tim_df_name = os.path.join(RES_DIR, 'mfa_tim_res.csv')
    tim_df = pd.DataFrame({'tim_err': tim_errs})
    tim_df.to_csv(tim_df_name, index=False)
    
    buck_df_name = os.path.join(RES_DIR, 'mfa_buck_res.csv')
    buck_df = pd.DataFrame({'buck_err': buck_errs})
    buck_df.to_csv(buck_df_name, index=False)

    with open(RES_NAME, 'a') as w: w.write('finished mfa\n')
