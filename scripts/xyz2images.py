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

bar = progressbar.ProgressBar(widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(marker='█'),
    ' (', progressbar.ETA(), ') ',
])

@cuda.jit(device=True)
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
    parser.add_argument('--gpu', help='Use GPUs.', action='store_true', dest='gpu', default=True)
    parser.add_argument('--n_gpus', help='How many GPUs?', dest='n_gpus', type=int)
    return parser.parse_args()

@cuda.jit()
def makeGaussian3D(image, X, Y, Z, positions, N, cell):
    i = cuda.grid(1)
    if i > (X.shape[0]):
        return
    for j in range(positions.shape[0]):
        atom = positions[j]
        dx = X[i] - atom[0]
        dx -= pbc_round(dx / cell[0]) * cell[0]
        dy = Y[i] - atom[1]
        dy -= pbc_round(dy / cell[1]) * cell[1]
        dz = Z[i] - atom[2]
        dz -= pbc_round(dz / cell[2]) * cell[2]
        dr = dx**2 + dy**2 + dz**2
        image[i] += N[j] * math.exp(-dr / (2 * (0.2**2)))

@cuda.jit()
def makeGaussian2D(image, X, Y, positions, N, cell):
    i = cuda.grid(1)
    if i > (X.shape[0]):
        return
    for j in range(positions.shape[0]):
        atom = positions[j]
        dx = X[i] - atom[0]
        dx -= pbc_round(dx / cell[0]) * cell[0]
        dy = Y[i] - atom[1]
        dy -= pbc_round(dy / cell[1]) * cell[1]
        dr = dx**2 + dy**2
        image[i] += N[j] * math.exp(-dr / (2 * (0.2**2)))


def randRotateAndTranslateWrap(args):
    parser, filename, gpu_id = args
    snapshot = aseio.read(filename, index=0, format='xyz')
    cuda.select_device(gpu_id)

    cell = parser.supercell

    if parser.dim == 3:
        xgrid = np.linspace(0, cell[0], parser.grid[0])
        ygrid = np.linspace(0, cell[1], parser.grid[1])
        zgrid = np.linspace(0, cell[2], parser.grid[2])
        X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid)

        N = snapshot.get_atomic_numbers()

        positions = snapshot.get_positions(wrap=True)
        image = np.zeros((parser.grid[0], parser.grid[1], parser.grid[2]), dtype=np.float32)
        if parser.gpu:
            tpb = 1024
            bpg = parser.grid[0] * parser.grid[1] * parser.grid[2] / tpb
            X = X.astype(np.float32).reshape(parser.grid[0] * parser.grid[1] * parser.grid[2],)
            Y = Y.astype(np.float32).reshape(parser.grid[0] * parser.grid[1] * parser.grid[2],)
            Z = Z.astype(np.float32).reshape(parser.grid[0] * parser.grid[1] * parser.grid[2],)
            image = image.reshape(parser.grid[0] * parser.grid[1] * parser.grid[2],)
            makeGaussian3D[bpg, tpb](image, X, Y, Z, positions, N, cell)
            image = image.reshape((parser.grid[0], parser.grid[1], parser.grid[2], 1))
        else:
            for j, atom in enumerate(positions):
                dx = X - atom[0]
                dx -= vec_pbc_round(dx / cell[0]) * cell[0]
                dy = Y - atom[1]
                dy -= vec_pbc_round(dy / cell[1]) * cell[1]
                dz = Z - atom[2]
                dz -= vec_pbc_round(dz / cell[2]) * cell[2]
                dr = dx**2 + dy**2 + dz**2
                image += N[j] * np.exp(-dr / (2 * (0.2**2)))
        return image

    elif parser.dim == 2:
        xgrid = np.linspace(0, cell[0], parser.grid[0])
        ygrid = np.linspace(0, cell[1], parser.grid[1])
        X, Y = np.meshgrid(xgrid, ygrid)
        Z = snapshot.get_atomic_numbers()

        # snapshot.rotate([random.uniform(0,1), random.uniform(0,1), 1], [random.uniform(0,1), random.uniform(0,1), 1], center='COU')
        # snapshot.translate(np.array([random.uniform(0, cell[0]), random.uniform(0, cell[1]), 0.]))
        positions = snapshot.get_positions(wrap=True)
        image = np.zeros((parser.grid[0], parser.grid[1]), dtype=np.float32)
        if parser.gpu:
            tpb = 1024
            bpg = parser.grid[0] * parser.grid[1] / tpb
            X = X.astype(np.float32).reshape(parser.grid[0] * parser.grid[1],)
            Y = Y.astype(np.float32).reshape(parser.grid[0] * parser.grid[1],)
            image = image.reshape(parser.grid[0] * parser.grid[1],)
            makeGaussian2D[bpg, tpb](image, X, Y, positions, Z, cell)
            image = image.reshape((parser.grid[0], parser.grid[1], 1))
        else:
            for j, atom in enumerate(positions):
                dx = X - atom[0]
                dx -= vec_pbc_round(dx / cell[0]) * cell[0]
                dy = Y - atom[1]
                dy -= vec_pbc_round(dy / cell[1]) * cell[1]
                dr = dx**2 + dy**2
                image += Z[j] * np.exp(-dr / (2 * (0.2**2)))
        return image

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
    n_batches = n_files / parser.n_gpus + 1
    parser.supercell = np.array(parser.supercell)
    n_gpus = parser.n_gpus
    p = Pool(n_gpus)
    for i in bar(range(n_batches)):
        start = i * n_gpus
        end = (i + 1) * n_gpus

        images = p.map(randRotateAndTranslateWrap, zip([parser] * (end - start), files[start:end], range(end - start)))
        h5_file['X'][start:end] = images
        sys.stdout.write('Progress: %0.2f%%\r' % (100. * i / n_batches))
    getEnergies(files, h5_file)
      
if __name__ == '__main__':
    main()
