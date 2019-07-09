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
import os
import time

bar = progressbar.ProgressBar(widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(marker='█'),
    ' (', progressbar.ETA(), ') ',
])

def pbc_round(input):
     i = int(input)
     if (abs(input - i) >= 0.5):
         if (input > 0):
             i += 1
         if (input < 0):
             i -= 1
     return i

vec_pbc_round = np.vectorize(pbc_round)

def createH5File(parser, n_images):
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
    parser = argparse.ArgumentParser(description='Turn your xyz files to images!')
    parser.add_argument('--dir', help='Directory holding xyz files.', dest='dirname')
    parser.add_argument('--dim', help='Dimension: either 2 or 3.', dest='dim', type=int)
    parser.add_argument('--supercell', help='Supercell dimensions', dest='supercell', type=float, nargs='+')
    parser.add_argument('--grid', help='Size of grid to create.', dest='grid', default=[128]*3, type=int, nargs='+')
    parser.add_argument('--name', help='Name of images file.', dest='name', default='images.h5')
    return parser.parse_args()


def singleAtomWork(args):
    atom, Z, extent, grid, cell, xgrid, ygrid = args
    xindex = int(atom[0] / (cell[0] / grid[0]))
    yindex = int(atom[1] / (cell[1] / grid[1]))
    x_range = np.array(range(xindex - extent[0], xindex + extent[0] + 1)) % grid[0]
    y_range = np.array(range(yindex - extent[1], yindex + extent[1] + 1)) % grid[1]

    dx = xgrid[x_range] - atom[0]
    dx -= vec_pbc_round(dx / cell[0]) * cell[0]
    dy = ygrid[y_range] - atom[1]
    dy -= vec_pbc_round(dy / cell[1]) * cell[1]
    dx, dy = np.meshgrid(dx, dy)
    incr = Z * np.exp(-(dx**2 + dy**2) / (2 * (0.2**2)))
    return x_range, y_range, incr

def randRotateAndTranslateWrap(args):
    parser, filename, pool = args
    snapshot = aseio.read(filename, index=0, format='xyz')
    cell = parser.supercell

    xgrid = np.linspace(0, cell[0], parser.grid[0])
    ygrid = np.linspace(0, cell[1], parser.grid[1])
    Z = snapshot.get_atomic_numbers()
    positions = snapshot.get_positions(wrap=True)
    n_atoms = positions.shape[0]


    extent = (np.array(parser.grid) / cell[:parser.dim]).astype(int)
    max_extent = np.max(extent)

    if parser.dim == 3:
        ''' 
            Not tested yet!
        '''
        zgrid = np.linspace(0, cel[2], parser.grid[2])
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
            xindex = int(atom[0] / (cell[0] / parser.grid[0]))
            yindex = int(atom[1] / (cell[1] / parser.grid[1]))
            zindex = int(atom[2] / (cell[2] / parser.grid[2]))
            x_range = np.array(range(xindex - extent[0], xindex + extent[0] + 1)) % parser.grid[0]
            y_range = np.array(range(yindex - extent[1], yindex + extent[1] + 1)) % parser.grid[1]
            z_range = np.array(range(zindex - extent[2], zindex + extent[2] + 1)) % parser.grid[2]

            dx = xgrid[x_range] - atom[0]
            dx -= vec_pbc_round(dx / cell[0]) * cell[0]
            dy = ygrid[y_range] - atom[1]
            dy -= vec_pbc_round(dy / cell[1]) * cell[1]
            dz = zgrid[z_range] - atom[2]
            dz -= vec_pbc_round(dy / cell[2]) * cell[2]
            dx, dy, dz = np.meshgrid(dx, dy, dz)
            image[np.ix_(y_range, x_range, z_range)] += Z[j] * np.exp(-(dx**2 + dy**2 + dz**2) / (2 * (0.2**2)))
        return image

    elif parser.dim == 2:
        image = np.zeros((parser.grid[0], parser.grid[1]), dtype=np.float32)
        stuff = pool.map(singleAtomWork, zip(positions, Z, [extent] * n_atoms, [parser.grid] * n_atoms, [cell] * n_atoms, [xgrid] * n_atoms, [ygrid] * n_atoms))
        
        for x_range, y_range, incr in stuff:
            image[np.ix_(y_range, x_range)] += incr

        # for j, atom in enumerate(positions):
        #     '''
        #         Old way of doing things....very slow.
        #     '''
        #     # dx = X - atom[0]
        #     # dx -= vec_pbc_round(dx / cell[0]) * cell[0]
        #     # dy = Y - atom[1]
        #     # dy -= vec_pbc_round(dy / cell[1]) * cell[1]
        #     # dr = dx**2 + dy**2
        #     # image += Z[j] * np.exp(-dr / (2 * (0.2**2)))

        #     ''' 
        #         New way of doing things....much faster.
        #     '''
        #     xindex = int(atom[0] / (cell[0] / parser.grid[0]))
        #     yindex = int(atom[1] / (cell[1] / parser.grid[1]))
        #     x_range = np.array(range(xindex - extent[0], xindex + extent[0] + 1)) % parser.grid[0]
        #     y_range = np.array(range(yindex - extent[1], yindex + extent[1] + 1)) % parser.grid[1]

        #     dx = xgrid[x_range] - atom[0]
        #     dx -= vec_pbc_round(dx / cell[0]) * cell[0]
        #     dy = ygrid[y_range] - atom[1]
        #     dy -= vec_pbc_round(dy / cell[1]) * cell[1]
        #     dx, dy = np.meshgrid(dx, dy)
        #     image[np.ix_(y_range, x_range)] += Z[j] * np.exp(-(dx**2 + dy**2) / (2 * (0.2**2)))
        return image.reshape((parser.grid[0], parser.grid[1], 1))

def getFiles(parser):
    return [os.path.join(parser.dirname, f) for f in os.listdir(parser.dirname)]

def getEnergies(files, h5_file):
    for i, filename in enumerate(files):
        energy = float(open(filename, 'r').readlines()[1])
        h5_file['Y'][i] = energy

def main():
    parser = parseArgs()
    files = getFiles(parser)
    n_files = len(files)
    h5_file = createH5File(parser, n_files)
    workers = multiprocessing.cpu_count()
    # workers = 1
    batch_size = workers
    # n_batches = n_files / batch_size + 1
    pool = Pool(workers)
    # for i in bar(range(n_batches)):
    #     start = i * batch_size
    #     end = (i + 1) * batch_size
    #     if end > n_files:
    #         end = n_files
    start_t = time.time()
    images = map(randRotateAndTranslateWrap,
                             zip([parser] * (n_files),
                             files, [pool] * n_files))
    print 'Image generation process took:', time.time() - start_t, 'seconds.'
    h5_file['X'][:] = images
            # sys.stdout.write('Progress: %0.2f%%\r' % (100. * i / n_batches))
            # sys.stdout.flush()
    # for i in range(0, len(h5_file['Y']), 100):
    #    plt.imshow(h5_file['X'][i].reshape((256, 256)))
    #    plt.show()

    getEnergies(files, h5_file)
    h5_file.attrs['min'] = np.min(h5_file['Y'][:])
    h5_file.attrs['max'] = np.max(h5_file['Y'][:])
    h5_file.close()
if __name__ == '__main__':
    main()

