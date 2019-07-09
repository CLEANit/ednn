#!/usr/bin/env python

import sys
import h5py
import numpy as np
import random

def main():
    f1 = h5py.File(sys.argv[1])
    f2 = h5py.File(sys.argv[2])
    n1 = len(f1['Y'])
    n2 = len(f2['Y'])
    n_images = n1 + n2
    h5file = h5py.File('combined.h5')
    h5file.create_dataset('Y', (n_images,) + f1['Y'].shape[1:])
    h5file.create_dataset('X', (n_images,) + f1['X'].shape[1:])

    a = range(n_images)
    random.shuffle(a)

    for i, val in enumerate(a):
        if val >= n1:
            h5file['X'][i] = f2['X'][val - n1]
            h5file['Y'][i] = f2['Y'][val - n1]
        else:
            h5file['X'][i] = f1['X'][val]
            h5file['Y'][i] = f1['Y'][val]
        sys.stdout.write('Progress: %0.2f%%\r' % (100. * i / n_images))            
 
    for key, val in f1.attrs.items():
        h5file.attrs[key] = val
    h5file.attrs['min'] = min(f1.attrs['min'], f2.attrs['min'])
    h5file.attrs['max'] = max(f1.attrs['max'], f2.attrs['max'])
    h5file.close()
if __name__ == '__main__':
    main()
