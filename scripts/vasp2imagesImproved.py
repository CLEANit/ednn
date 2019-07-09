#!/usr/bin/env python
# coding=utf-8

import numpy as np
import h5py
import sys
import ase.io as aseio
import random
import argparse
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import multiprocessing
from numba import cuda, jit
import math
import matplotlib.pyplot as plt
import progressbar

bar = progressbar.ProgressBar(widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(marker='█'),
    ' (', progressbar.ETA(), ') ',
])

# @cuda.jit(device=True)
def pbc_round(input):
     i = int(input)
     if (abs(input - i) >= 0.5):
         if (input > 0):
             i += 1
         if (input < 0):
             i -= 1
     return i

vec_pbc_round = np.vectorize(pbc_round)

def openWithAse(filename):
    return aseio.read(filename, index=':', format='vasp-xml')

def createH5File(parser, data):
    n_images = len(data)
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
        
def parseArgs():
    parser = argparse.ArgumentParser(description='Turn your vasprun.xml file to tiled images.')
    parser.add_argument('--file', help='XML file name.', dest='xml_file')
    parser.add_argument('--dim', help='Dimension: either 2 or 3.', dest='dim', type=int)
    parser.add_argument('--grid', help='Size of grid to create.', dest='grid', default=[128]*3, type=int, nargs='+')
    parser.add_argument('--name', help='Name of images file.', dest='name', default='images.h5')
    return parser.parse_args()

def randRotateAndTranslateWrap(args):
    parser, snapshot, energy = args

    cell = snapshot.cell
    extent = (np.array(parser.grid) / np.diagonal(cell)[:parser.dim]).astype(int)
    max_extent = np.max(extent)
    if parser.dim == 3:
        '''
            Not optimized yet!!!
        '''
        xgrid = np.linspace(0, cell[0][0], parser.grid[0])
        ygrid = np.linspace(0, cell[1][1], parser.grid[1])
        zgrid = np.linspace(0, cell[2][2], parser.grid[2])
        X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid)
        if energy:
            E = snapshot.get_potential_energy()
        N = snapshot.get_atomic_numbers()

        positions = snapshot.get_positions(wrap=True)
        image = np.zeros((parser.grid[0], parser.grid[1], parser.grid[2]), dtype=np.float32)

        for j, atom in enumerate(positions):
            dx = X - atom[0]
            dx -= vec_pbc_round(dx / cell[0][0]) * cell[0][0]
            dy = Y - atom[1]
            dy -= vec_pbc_round(dy / cell[1][1]) * cell[1][1]
            dz = Z - atom[2]
            dz -= vec_pbc_round(dz / cell[2][2]) * cell[2][2]
            dr = dx**2 + dy**2 + dz**2
            image += N[j] * np.exp(-dr / (2 * (0.2**2)))
        if energy:
            return image, E
        else:
            return image

    elif parser.dim == 2:
        xgrid = np.linspace(0, cell[0][0], parser.grid[0])
        ygrid = np.linspace(0, cell[1][1], parser.grid[1])
        if energy:
            E = snapshot.get_potential_energy()
        Z = snapshot.get_atomic_numbers()

        snapshot.wrap()
        positions = snapshot.get_positions()
        image = np.zeros((parser.grid[0], parser.grid[1]), dtype=np.float32)
        for j, atom in enumerate(positions):
            '''
                Old way of doing things....very slow.
            '''
            # dx = X - atom[0]
            # dx -= vec_pbc_round(dx / cell[0][0]) * cell[0][0]
            # dy = Y - atom[1]
            # dy -= vec_pbc_round(dy / cell[1][1]) * cell[1][1]
            # dr = dx**2 + dy**2
            # image += Z[j] * np.exp(-dr / (2 * (0.2**2)))

            ''' 
                New way of doing things....much faster.
            '''
            xindex = int(atom[0] / (cell[0][0] / parser.grid[0]))
            yindex = int(atom[1] / (cell[1][1] / parser.grid[1]))
            x_range = np.array(range(xindex - extent[0], xindex + extent[0] + 1)) % parser.grid[0]
            y_range = np.array(range(yindex - extent[1], yindex + extent[1] + 1)) % parser.grid[1]

            dx = xgrid[x_range] - atom[0]
            dx -= vec_pbc_round(dx / cell[0][0]) * cell[0][0]
            dy = ygrid[y_range] - atom[1]
            dy -= vec_pbc_round(dy / cell[1][1]) * cell[1][1]
            dx, dy = np.meshgrid(dx, dy)
            image[np.ix_(y_range, x_range)] += Z[j] * np.exp(-(dx**2 + dy**2) / (2 * (0.2**2)))
        if energy:
            return image, E
        else:
            return image

def main():
    parser = parseArgs()
    print 'Reading in file:', parser.xml_file
    data = openWithAse(parser.xml_file)
    print 'Done.'
    h5_file = createH5File(parser, data)
    workers = multiprocessing.cpu_count()
    batch_size = workers
    n_batches = len(data) / batch_size + 1
    pool = Pool(workers)
    for i in bar(range(n_batches)):
        start = i * batch_size
        end = (i + 1) * batch_size
        if end > len(data):
            end = len(data)
        results = pool.map(randRotateAndTranslateWrap,
                             zip([parser] * (end - start),
                             data[start:end], [True] * (end - start)))
        images, energies = [], []
        for im, e in results:
            images.append(im)
            energies.append(e)

        images = np.array(images, dtype=np.float32)
        energies = np.array(energies, dtype=np.float32)
        h5_file['X'][start:end] = images.reshape(images.shape + (1,)) 
        h5_file['Y'][start:end] = energies.reshape((len(energies), 1))
            # sys.stdout.write('Progress: %0.2f%%\r' % (100. * i / n_batches))
            # sys.stdout.flush()
    # for i in range(0, len(h5_file['Y']), 100):
    #    plt.imshow(h5_file['X'][i].reshape((256, 256)))
    #    plt.show()
    h5_file.attrs['cell_dimensions'] = data[0].cell      
    h5_file.attrs['min'] = np.min(h5_file['Y'][:])
    h5_file.attrs['max'] = np.max(h5_file['Y'][:])
    h5_file.close()

if __name__ == '__main__':
    main()
