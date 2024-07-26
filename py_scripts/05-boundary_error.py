from textgrid import textgrid
import statistics
import math
import os
from tqdm import tqdm
import re
import pandas as pd
from os.path import isfile
import sys

# Change "_train" to "_test" or "_val" to evaluate the test
# or val performance instead of train
ALIGN_DIR = 'D:/timbuck_data/timbuck10_val'
GOLD_DIR = 'D:/timbuck_data/timbuck_textgrids'
MODEL_DIR = 'timbuck_trained_models_repetitions'

RES_DIR = 'mfa_boundary_eval_res/val'

# counts if any labels were skipped due to label
# incommensurability. Ideally, this should be 0,
# and it is for our tests here
SKIPPED = 0

# When True, will overwrite evaluation files already made, and won't skip anything
#
# When False, will not overwrite existing files and will skip evaluating those
# files again
OVERWRITE = True

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

BUCKEYE_TRANSLATION = {
    'a':'ah',
    'aan':'aa',
    'aen':'ae',
    'ahn':'ah',
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
    '<sil>': 'h#',
    'SIL': 'h#',
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
    'sil'
])
    
def compare_tg(tg_p, tg_g):

    errs = []
    tim_errs = []
    buck_errs = []
    
    predicted = textgrid.TextGrid()
    predicted.read(os.path.join(ALIGN_DIR, tg_p), round_digits=100)
        
    gold = textgrid.TextGrid()
    try:
        gold.read(os.path.join(GOLD_DIR, tg_g), round_digits=100)
    except:
        print(f'TextGrid read failed. Halting.')
        print(tg_p)
        print(tg_g)
        sys.exit()
        
    if tg_g.startswith('s'):
        gt = convert_buckeye_tier(gold.tiers[-1])
    else:
        gt = convert_timit_tier(gold.tiers[-1])
        
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
            print(f'Skipped on {tg_p}')
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
    
    for i in range(len(t)-1):
    
        curr_mark = t[i].mark
        next_mark = t[i+1].mark
        
        if curr_mark == next_mark or curr_mark == '':
            to_delete.add(i)
            
    if t[-1].mark == '':
        to_delete.append(len(t)-1)
        
    new_t = [x for i, x in enumerate(t) if not i in to_delete]
    return new_t
    
def convert_buckeye_tier(t):

    to_delete = set()

    for i in range(len(t)-1):
    
        curr_mark = t[i].mark
        next_mark = t[i+1].mark

        curr_mark = curr_mark.replace(';', '').replace('+1', '')
        
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
    '''
    Compare TextGrids using iterators to try to get account for
    uneven numbers of labels. Sometimes occurs when including
    silence labels at beginning or end.
    '''

    p_itr = iter(p)
    g_itr = iter(g)

    results = []

    p_cur = next(p_itr)
    g_cur = next(g_itr)

    while True:
        try:
            if p_cur.mark == g_cur.mark:
                results.append(g_cur.maxTime - p_cur.maxTime)
                p_cur = next(p_itr)
                g_cur = next(g_itr)
            else:
                g_cur = next(g_itr)
        except StopIteration:
            break

    if len(results) == len(p):
        return results
    else:
        return -1

if __name__ == '__main__':

    fnames = os.listdir(ALIGN_DIR)
    fnames.sort()

    with open('resfile_test.txt', 'a') as w: w.write('----------\n')
    
    for fname in CRISP_MODELS + SPARSE_MODELS:

        with open('resfile_test.txt', 'a') as w: w.write('beginning {}\n'.format(fname))

        base = os.path.splitext(os.path.basename(fname))[0]
        interp_name = base + '_interp'
        noint_name = base + '_noint'
        
        all_df_name = os.path.join(RES_DIR, base + '_all_res.csv')
        tim_df_name = os.path.join(RES_DIR, base + '_tim_res.csv')
        buck_df_name = os.path.join(RES_DIR, base + '_buck_res.csv')
        
        # Skip evaluation if files already exist and RESUME set to True
        if (not OVERWRITE) and isfile(all_df_name) and isfile(tim_df_name) and isfile(buck_df_name):
            continue
        
        int_pred = [x for x in fnames if x.endswith('.TextGrid') and interp_name in x]
        noint_pred = [x.replace('_interp', '_noint') for x in int_pred]
        tg_gold = [re.sub(r'(real|full).*interp', '', x) for x in int_pred]
        
        int_errs = []
        noint_errs = []
        
        int_tim_errs = []
        noint_tim_errs = []
        
        int_buck_errs = []
        noint_buck_errs = []
        
        file_names = []
        segments = []
        previous_segments = []
        following_segments = []
        sources = []
        
        for interp, noint, gold in tqdm(list(zip(int_pred, noint_pred, tg_gold))):
        
            if re.sub(r'(real|full).*interp', '', interp) in FILES_TO_IGNORE:
                continue
            
            # If a segment's duration gets almost completely nuked by the interplation,
            # the textgrid library won't have a tier with it due to rounding behavior.
            # It will still load in Praat though.
            # Setting round_digits=2000 should mitigate this issue for the most part.
            int_tg = textgrid.TextGrid()
            int_tg.read(os.path.join(ALIGN_DIR, interp), round_digits=2000)
            
            noint_tg = textgrid.TextGrid()
            noint_tg.read(os.path.join(ALIGN_DIR, noint), round_digits=2000)
            
            # Check to make sure an interval didn't go missing due to rounding
            if len(int_tg.tiers[-1]) != len(noint_tg.tiers[-1]):
                print(f'Missing tier for {interp}')
                sys.exit()
                continue
        
            int_res = compare_tg(interp, gold)
            
            if len(int_res['all']) != len(int_res['segments']):
                print()
                print([round(x, 3) for x in int_res['all']])
                print(int_res['segments'])
                print(interp)
                print(gold)
                sys.exit()
            
            int_errs += int_res['all']
            int_tim_errs += int_res['tim']
            int_buck_errs += int_res['buck']
            
            file_names += [interp] * len(int_res['all'])
            segments += int_res['segments']
            previous_segments += ['#'] + int_res['segments'][:-1]
            following_segments += int_res['segments'][1:] + ['#']
            sources += ['buckeye' if os.path.basename(interp).startswith('s') else 'timit'] * len(int_res['all'])
            
            noint_res = compare_tg(noint, gold)
            
            noint_errs += noint_res['all']
            noint_tim_errs += noint_res['tim']
            noint_buck_errs += noint_res['buck']

            # Check to make sure the same number of errors are occurring
            if len(int_errs) != len(noint_errs):
                print(interp, noint)
                print([(x.mark, x.maxTime - x.minTime) for x in int_tg.tiers[-1]])
                print([(x.mark, x.maxTime - x.minTime) for x in noint_tg.tiers[-1]])
                print([(x.mark, x.maxTime - x.minTime) for x in gold_tg.tiers[-1]])
                sys.exit()
        
        print()
        print('Skipped:\t', SKIPPED)
        SKIPPED = 0
                
        print(len(int_errs), len(noint_errs))
        all_df = pd.DataFrame({'interp': int_errs, 'nointerp': noint_errs, 'filename': file_names, 'segment': segments, 'prev_segment': previous_segments, 'next_segment': following_segments, 'corpus': sources})
        all_df.to_csv(all_df_name, index=False)
        
        tim_df = pd.DataFrame({'interp': int_tim_errs, 'nointerp': noint_tim_errs})
        tim_df.to_csv(tim_df_name, index=False)
        
        buck_df = pd.DataFrame({'interp': int_buck_errs, 'nointerp': noint_buck_errs})
        buck_df.to_csv(buck_df_name, index=False)
        with open('resfile_test.txt', 'a') as w: w.write('finished {}\n'.format(fname))