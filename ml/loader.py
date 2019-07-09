#!/usr/bin/env

import tensorflow as tf
import h5py
import os
import numpy as np
import time
import random

class Loader():
    def __init__(self, logger, parser, config):
        start = time.time()
        self.logger = logger
        self.parser = parser
        self.config = config
        self.logger.info('Initializing Loader.')
        self.mapping = {}
        if parser.train:
            self.files = self.readDir(os.getcwd() + '/train')
            self.h5_files = self.prepareData(self.files)
        elif parser.test:
            self.files = self.readDir(os.getcwd() + '/test')
            self.h5_files = self.prepareData(self.files)
        self.logger.info('Initializing loader took %5.5f s.' % (time.time() - start))

    def readDir(self, dir_name):
        if os.path.isdir(dir_name):
            files = os.listdir(dir_name)
            if len(files) == 0:
                self.logger.error('No files found in dir: %s' % dir_name)
                exit(-1)
            else:
                return [dir_name + '/' + elem for elem in  os.listdir(dir_name)]
        else:
            self.logger.error('No dir called: %s, please put your h5 files in there.' % dir_name)


    def prepareData(self, files):
        h5_files = []
        filenames = []
        for fname in files:
            filenames.append(fname)
            h5_files.append(h5py.File(fname, 'r'))
            # self.mapping[fname] = h5py.File(fname, 'r')

        self.min = np.inf
        self.max = -np.inf
        self.x_shape = None
        self.y_shape = None
        self.total = 0
        self.image_counts = {}
        self.validation_indices= {}
        self.filenames = filenames
        for i, h5file in enumerate(h5_files):
            x_shape = h5file[self.config['x_label']].shape
            y_shape = h5file[self.config['y_label']].shape
            
            if x_shape[0] != y_shape[0]:
                self.logger.error('The datasets X and Y must have the same length!')
                exit(-1)
            self.image_counts[filenames[i]] = y_shape[0]

            self.total += y_shape[0]

            min_y = np.min(h5file[self.config['y_label']])
            
            if min_y < self.min:
                self.min = min_y

            max_y = np.max(h5file[self.config['y_label']])
            if max_y > self.max:
                self.max = max_y

            if i > 0:
                if self.x_shape != x_shape[1:]:
                    self.logger.error('All of the X datasets must have the same shape!')
                    exit(-1)
            else:
                self.x_shape = x_shape[1:]

            if i > 0:
                if self.y_shape != y_shape[1:]:
                    self.logger.error('All of the Y datasets must have the same shape!')
                    exit(-1)
            else:
                self.y_shape = y_shape[1:]

            self.n_validations = int(self.config['validation_size'] / len(h5_files) * y_shape[0])
            if self.n_validations == 0:
                skipper = 1
            else:
                skipper = y_shape[0] / self.n_validations
            
            indices = range(0, y_shape[0], skipper)
            self.validation_indices[filenames[i]] = indices
            if self.parser.train:
                h5file.close()
        # self.logger.info('Normalizing data...')
        # for h5file in h5_files:
        #   h5file[self.config['y_label']][:] += self.min
        #   h5file[self.config['y_label']][:] /= (self.max - self.min)


        return h5_files

    def getTotalImages(self):
        if self.parser.train:
            return self.total - self.n_validations
        else:
            return self.total
    def getTotalValidationImages(self):
        return self.n_validations

    def getImageCountsPerFile(self):
        return self.image_counts

    def getValidationIndicesPerFile(self):
        return self.validation_indices

    def getMin(self):
        return self.min

    def getMax(self):
        return self.max

    def getXShape(self):
        return self.x_shape

    def getYShape(self):
        return self.y_shape

    def getH5Files(self):
        return self.h5_files

    def getFiles(self):
        return self.files

    def getH5FileFromStr(self, name):
        return self.mapping[name]

