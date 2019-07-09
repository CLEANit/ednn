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
    parser.add_argument('--gpu', help='Use GPUs.', action='store_true', dest='gpu', default=False)
    return parser.parse_args()

@cuda.jit()
def makeGaussian3D(image, X, Y, Z, positions, N, cell):
    i = cuda.grid(1)
    if i > (X.shape[0]):
        return
    for j in range(positions.shape[0]):
        atom = positions[j]
        dx = X[i] - atom[0]
        dx -= pbc_round(dx / cell[0][0]) * cell[0][0]
        dy = Y[i] - atom[1]
        dy -= pbc_round(dy / cell[1][1]) * cell[1][1]
        dz = Z[i] - atom[2]
        dz -= pbc_round(dz / cell[2][2]) * cell[2][2]
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
        dx -= pbc_round(dx / cell[0][0]) * cell[0][0]
        dy = Y[i] - atom[1]
        dy -= pbc_round(dy / cell[1][1]) * cell[1][1]
        dr = dx**2 + dy**2
        image[i] += N[j] * math.exp(-dr / (2 * (0.2**2)))


def randRotateAndTranslateWrap(args):
    parser, snapshot, gpu_id, energy = args

    cell = snapshot.cell
    extent = (np.array(parser.grid) / np.diagonal(cell)[:parser.dim]).astype(int)
    max_extent = np.max(extent)
    if parser.dim == 3:
        xgrid = np.linspace(0, cell[0][0], parser.grid[0])
        ygrid = np.linspace(0, cell[1][1], parser.grid[1])
        zgrid = np.linspace(0, cell[2][2], parser.grid[2])
        X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid)
        if energy:
            E = snapshot.get_potential_energy()
        N = snapshot.get_atomic_numbers()

        positions = snapshot.get_positions(wrap=True)
        image = np.zeros((parser.grid[0], parser.grid[1], parser.grid[2]), dtype=np.float32)
        if parser.gpu:
            cuda.select_device(gpu_id)
            tpb = 1024
            bpg = parser.grid[0] * parser.grid[1] * parser.grid[2] / tpb
            X = X.astype(np.float32).reshape(parser.grid[0] * parser.grid[1] * parser.grid[2],)
            Y = Y.astype(np.float32).reshape(parser.grid[0] * parser.grid[1] * parser.grid[2],)
            Z = Z.astype(np.float32).reshape(parser.grid[0] * parser.grid[1] * parser.grid[2],)
            image = image.reshape(parser.grid[0] * parser.grid[1] * parser.grid[2],)
            makeGaussian3D[bpg, tpb](image, X, Y, Z, positions, N, cell)
            image = image.reshape((parser.grid[0], parser.grid[1], parser.grid[2]))
        else:
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
        X, Y = np.meshgrid(xgrid, ygrid)
        if energy:
            E = snapshot.get_potential_energy()
        Z = snapshot.get_atomic_numbers()

        snapshot.wrap()
        positions = snapshot.get_positions()
        image = np.zeros((parser.grid[0], parser.grid[1]), dtype=np.float32)
        if parser.gpu:
            cuda.select_device(gpu_id)
            tpb = 1024
            bpg = parser.grid[0] * parser.grid[1] / tpb
            X = X.astype(np.float32).reshape(parser.grid[0] * parser.grid[1],)
            Y = Y.astype(np.float32).reshape(parser.grid[0] * parser.grid[1],)
            image = image.reshape(parser.grid[0] * parser.grid[1],)
            makeGaussian2D[bpg, tpb](image, X, Y, positions, Z, cell)
            image = image.reshape((parser.grid[0], parser.grid[1]))
        else:
            for j, atom in enumerate(positions):
                '''
                    Old way of doing things....very slow.
                '''
                dx = X - atom[0]
                dx -= vec_pbc_round(dx / cell[0][0]) * cell[0][0]
                dy = Y - atom[1]
                dy -= vec_pbc_round(dy / cell[1][1]) * cell[1][1]
                dr = dx**2 + dy**2
                image += Z[j] * np.exp(-dr / (2 * (0.2**2)))

                ''' 
                    New way of doing things....much faster.
                '''
                # xindex = int(atom[0] / (cell[0][0] / parser.grid[0]))
                # yindex = int(atom[1] / (cell[1][1] / parser.grid[1]))
                # x_range = np.array(range(xindex - max_extent, xindex + max_extent + 1)) % parser.grid[0]
                # y_range = np.array(range(yindex - max_extent, yindex + max_extent + 1)) % parser.grid[1]

                # dx = xgrid[x_range] - atom[0]
                # dx -= vec_pbc_round(dx / cell[0][0]) * cell[0][0]
                # dy = ygrid[y_range] - atom[1]
                # dy -= vec_pbc_round(dy / cell[1][1]) * cell[1][1]
                # dx, dy = np.meshgrid(dx, dy)
                # image[np.ix_(y_range, x_range)] += Z[j] * np.exp(-(dx**2 + dy**2) / (2 * (0.2**2)))
        if energy:
            return image, E
        else:
            return image

def gpuWork(stuff):
    batch, batch_size, gpu_id, data_len, data, parser = stuff
    workers = multiprocessing.cpu_count() / len(cuda.list_devices())
    pool = ThreadPool(workers)
    cuda.select_device(gpu_id)

    start = batch * batch_size
    end = (batch + 1) * batch_size
    if end > data_len:
        end = data_len
    results = pool.map(randRotateAndTranslateWrap,
                     zip([parser] * (end - start),
                     data[start:end],
                     [gpu_id,]*len(data[start:end]), [True]*len(data[start:end])))

    images, energies = [], []
    for im, e in results:
        images.append(im)
        energies.append(e)

    images = np.array(images, dtype=np.float32)
    energies = np.array(energies, dtype=np.float32)
    pool.close()
    pool.join()
    return images, energies

def main():
    parser = parseArgs()
    print 'Reading in file:', parser.xml_file
    data = openWithAse(parser.xml_file)
    print 'Done.'
    h5_file = createH5File(parser, data)
    workers = multiprocessing.cpu_count()
    batch_size = workers
    n_batches = len(data) / batch_size + 1
    gpu_ids = range(len(cuda.list_devices()))
    if parser.gpu:
        batch_size /= len(cuda.list_devices())
        total = 0
        n_batches = len(data) / (batch_size) + 1
        outer_pool = ThreadPool(len(cuda.list_devices()))

        for batch in range(0, n_batches, len(gpu_ids)):
            start = batch
            end = batch + len(gpu_ids)
            if end > n_batches:
                end = n_batches

            batches = range(start, end)
            batch_sizes = [batch_size] * len(batches)
            data_lens = [len(data)] * len(batches)
            results = outer_pool.map(gpuWork, zip(batches, batch_sizes, gpu_ids, data_lens, [data] * len(batches), [parser] * len(batches)))
            for i, (x, y) in enumerate(results):
                write_start = total * batch_size
                write_stop = (total + 1) * batch_size
                if write_stop > len(data):
                    write_stop = len(data)
                h5_file['X'][write_start:write_stop] = x.reshape(x.shape + (1,)) 
                h5_file['Y'][write_start:write_stop] = y.reshape((len(y), 1))
                total += 1
            sys.stdout.write('Progress: %0.2f%%\r' % (100. * batch / n_batches))
            sys.stdout.flush()
    else:
        pool = Pool(workers)
        for i in bar(range(n_batches)):
            start = i * batch_size
            end = (i + 1) * batch_size
            if end > len(data):
                end = len(data)
            results = pool.map(randRotateAndTranslateWrap,
                                 zip([parser] * (end - start),
                                 data[start:end],
                                 [0,]*(end - start), [True] * (end - start)))
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
