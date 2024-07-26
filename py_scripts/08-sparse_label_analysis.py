import numpy as np
import os
from tqdm import tqdm

phones = 'h#	q	eh	dx	iy	r	ey	ix	tcl	sh	ow	z	s	hh	aw	m	t	er	l	w	aa	hv	ae	dcl	y	axr	d	kcl	k	ux	ng	gcl	g	ao	epi	ih	p	ay	v	n	f	jh	ax	en	oy	dh	pcl	ah	bcl	el	zh	uw	pau	b	uh	th	ax-h	em	ch	nx	eng'.split()
phn2num = {x: i for i, x in enumerate(phones)}

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
    # 'zh': 'sh',
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

to_keep = [i for i, x in enumerate(phones) if x not in foldings]

num2phn = {x: i for i, x in enumerate(phones)}

def create_distance_matrix(data_dir, out_name):
    
    print('Gathering label names...')
    labnames = [x for x in tqdm(os.listdir(data_dir)) if x.endswith('labs.npy')]
    print('\nGathering sparse names...')
    rd1names = [x for x in tqdm(os.listdir(data_dir)) if x.endswith('npy') and 'rd1.' in x and 'sparse' in x]
    
    scores = np.zeros((61, 61))
    
    for fl, fr in tqdm(list(zip(labnames, rd1names))):
        rdnames = [fr.replace('rd1', f'rd{i}') for i in range(1, 11)]
        tensor = np.stack([np.load(os.path.join(data_dir, x)) for x in rdnames])
        
        # Swap 0s and 1s to calculate something like hamming distance
        tensor = np.logical_not(np.array(tensor, dtype=bool))
        
        s = np.mean(tensor, axis=0)
        labs = np.load(os.path.join(data_dir, fl))
        labs = np.argmax(labs, 1)
        
        for i, la in enumerate(labs):
            scores[la,:] += s[i,:]
            scores[:,la] += s[i,:]

    # Add to both column and row to make symmetric
    scores = scores[to_keep, :]
    scores = scores[:, to_keep]
        
    np.savetxt(out_name, scores, delimiter=',')
    
if __name__ == '__main__':

    create_distance_matrix('D:/timbuck_data/timbuck10_train', 'train_sparse_distances.csv')
    create_distance_matrix('D:/timbuck_data/timbuck10_val', 'val_sparse_distances.csv')
    
    labs = np.array([x for x in phones if x not in foldings])
    np.savetxt("phone_labels.txt", labs, fmt='%s')
    