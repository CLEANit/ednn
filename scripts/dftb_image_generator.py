#!/usr/bin/env python

import ase.io as aseio
import os
import subprocess
import multiprocessing
from multiprocessing import Pool
from ase.calculators.dftb import Dftb
from ase.calculators.calculator import kptdensity2monkhorstpack
import argparse
from vasp2images import randRotateAndTranslateWrap
import copy
import h5py
import numpy as np
import sys


def parseArgs():
    parser = argparse.ArgumentParser(description='Turn your vasprun.xml file to tiled images.')
    parser.add_argument('--file', help='Input coordinates, must be a POSCAR form', dest='file', default='POSCAR')
    parser.add_argument('--dim', help='Dimension: either 2 or 3.', dest='dim', type=int)
    parser.add_argument('--grid', help='Size of grid to create.', dest='grid', default=[128]*3, type=int, nargs='+')
    parser.add_argument('--name', help='Name of images file.', dest='name', default='images.h5')
    parser.add_argument('--gpu', help='Use GPUs.', action='store_true', dest='gpu')
    parser.add_argument('--forces', help='Make the labels forces, not energies.', action='store_true', dest='forces', default=False)
    parser.add_argument('--num', help='Number of configurations (default 10000)', dest='num', type=int, default=10000)
    return parser.parse_args()


def getImageAndEnergy(stuff):
    index, atoms, dftb_options, parser = stuff
    system = copy.deepcopy(atoms)
    system.rattle(stdev=0.1)
    system.wrap()
    dftb_options['atoms'] = system
    dftb_options['label'] = str(index) + '/'
    system.set_calculator(Dftb(**dftb_options))
    image, E = randRotateAndTranslateWrap((parser, system))
    return image, E

def createH5File(parser):
    n_images = parser.num
    f = h5py.File(parser.name , 'w')
    xlength = parser.grid[0]
    ylength = parser.grid[1]
    if parser.dim == 2:
        f.create_dataset('X', (n_images, xlength, ylength, 1))
        f.create_dataset('Y', (n_images, 1))
    elif parser.dim == 3:
        zlength = parser.grid[2]
        f.create_dataset('X', (n_images, xlength, ylength, zlength, 1))
        f.create_dataset('Y', (n_images, 1))
    else:
        print 'ERROR: You must supply the dimension.'
        exit(-1)
    return f

if __name__ == '__main__':
    parser = parseArgs()
    os.environ['DFTB_PREFIX'] = '/home/kryczko/dftbplus-17.1.x86_64-linux/matsci-0-3/'
    os.environ['DFTB_COMMAND'] = '/home/kryczko/dftbplus-17.1.x86_64-linux/bin/dftb+'
    
    atoms = aseio.read(parser.file, format='vasp')
    dftb_options = {}
    for elem in list(set(atoms.get_chemical_symbols())):
        dftb_options['Hamiltonian_MaxAngularMomentum_' + elem] = '"p"'
    dftb_options['kpts'] = kptdensity2monkhorstpack(atoms, kptdensity=1.0)
    
    batch_size = multiprocessing.cpu_count()
    pool = Pool(batch_size)
    h5_file = createH5File(parser)
    iters = parser.num / batch_size + 1
    for i in range(iters):
        start = i * batch_size
        end = (i + 1) * batch_size
        if end > parser.num:
            end = parser.num

        results = pool.map(getImageAndEnergy, zip(range(batch_size), [atoms] * (end - start), [dftb_options] * (end - start), [parser] * (end - start)))
        images, energies = [], []
        for im, e in results:
            images.append(im)
            energies.append(e)

        images = np.array(images, dtype=np.float32)
        energies = np.array(energies, dtype=np.float32)
        h5_file['X'][start:end] = images.reshape(images.shape + (1,)) 
        h5_file['Y'][start:end] = energies.reshape((len(energies), 1))
        sys.stdout.write('Progress: %0.2f%%\r' % (100. * i / iters))
        sys.stdout.flush()

    h5_file.attrs['min'] = np.min(h5_file['Y'][:])
    h5_file.attrs['max'] = np.max(h5_file['Y'][:])
    h5_file.close()
