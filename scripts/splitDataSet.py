#!/usr/bin/env python

import h5py
import subprocess
import sys
import random

# this file takes in a dataset (already randomly ordered) and splits it up into training and testing sets

def randomIndices(file, test_pct):
    length = len(file['Y'])
    indices = set([])
    while len(indices) < int(length * test_pct):
        indices.add(random.randint(0, length - 1))
    return list(indices)

def initFiles(file, fname, test_pct):
    training_file = h5py.File('train/' + fname , 'w')
    testing_file = h5py.File('test/' + fname , 'w')

    n_images = len(file['Y'])

    n_testing = int(n_images * test_pct)
    n_training = n_images - n_testing
    training_file.create_dataset('Y', (n_training,) + file['Y'].shape[1:])
    training_file.create_dataset('X', (n_training,) + file['X'].shape[1:])

    testing_file.create_dataset('Y', (n_testing,) + file['Y'].shape[1:])
    testing_file.create_dataset('X', (n_testing,) + file['X'].shape[1:])

    return testing_file, training_file

def syncAttrs(file, testing_file, training_file):
    for key, val in file.attrs.items():
        testing_file.attrs[key] = val
        training_file.attrs[key] = val

def writeToFiles(file, testing_file, training_file, test_indices):

    training_indices = list(set(range(len(file['Y']))) - set(test_indices))

    for i, val in enumerate(test_indices):
        testing_file['X'][i] = file['X'][val]
        testing_file['Y'][i] = file['Y'][val]
        sys.stdout.write('Progress (testing set): %0.2f%%\r' % ( 100. * i / len(test_indices)))

    for i, val in enumerate(training_indices):
        training_file['X'][i] = file['X'][val]
        training_file['Y'][i] = file['Y'][val]
        sys.stdout.write('Progress (training set): %0.2f%%\r' % ( 100. * i / len(training_indices)))



    training_file.close()
    testing_file.close()
    file.close()


def main():
    test_pct = float(sys.argv[2])
    fname = sys.argv[1]
    file = h5py.File(fname)

    subprocess.call(['mkdir -p train test'], shell=True)
    test_indices = randomIndices(file, test_pct)
    print 'Generated indices.'
    testing_file, training_file = initFiles(file, fname, test_pct)
    print 'Created new files.'
    syncAttrs(file, testing_file, training_file)
    writeToFiles(file, testing_file, training_file, test_indices)
    print 'Wrote data to new files.'

if __name__ == '__main__':
    main()
