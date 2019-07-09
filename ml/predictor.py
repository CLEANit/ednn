#!/usr/bin/env python

import time
import tensorflow as tf
import numpy as np

class Predictor():
    def __init__(self, logger, config, parser, DNN, gpu_id=0, reuse=False):
        start = time.time()
        self.logger = logger
        self.logger.info('Initializing trainer.')

        self.config = config
        self.DNN = DNN
        self.n_epoch = self.config['n_epoch']
        self.batch_size = self.config['batch_size']
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True

        self.learning_rate = self.config['learning_rate']
        self.name = self.config['name']
        self.n_gpu = self.config['n_gpu']
        self.focus = self.config['focus']
        self.context = self.config['context']
        self.padding = self.config['padding']
        self.tile = self.config['tile']
        self.parser = parser
        self.gpu_id = gpu_id
        self.reuse = reuse

        self.session = tf.Session(config=self.tf_config)

        if self.tile:
            with tf.device('/gpu:' + str(gpu_id)):
                with tf.name_scope('GPU_{0}'.format(gpu_id)) as scope:
                    self.X, self.Y = tf.placeholder(tf.float32, shape=(None,) + self.getTileXShape()), tf.placeholder(tf.float32, shape=(None,1))
                    self.tiledDNN(reuse=reuse)
        else:
            with tf.device('/gpu:' + str(gpu_id)):
                with tf.name_scope('GPU_{0}'.format(gpu_id)) as scope:
                    self.X, self.Y = tf.placeholder(tf.float32, shape=(None,) + self.getXShape()), tf.placeholder(tf.float32, shape=(None,1))
                    self.model = self.DNN(self.X)
        
        self.saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1e9)

        self.saver.restore(self.session, tf.train.latest_checkpoint('./checkpoints'))
        self.logger.info('Successfully initialized on GPU: %i' % (gpu_id))


    def getXShape(self):
        return tuple(self.parser.grid) + (1,)

    def getTileXShape(self):
        return tuple(np.add(self.getXShape(), 2 * np.array(self.context + [0])))

    def tiledDNN(self, reuse):
    # All the input, consisting of `n_tiles` tiles with dimensions from `tile_shape`.
    # The dimensions in tile_shape (in our tested case 2D, eg, width and height) account for padding.
    # Split the input into a list of `n_tiles` tensors, 
    
        all_gradients = []
        all_losses = []
        v_tile_inputs = []
        og_image_size = np.array(self.getXShape()[:-1])
        split_image_size = np.add(self.focus, 2 * np.array(self.context))
        counts = og_image_size / split_image_size
        
        if len(counts) == 2:
            for i in range(counts[0]):
                for j in range(counts[1]):
                    v_tile_inputs.append(tf.slice(self.X,
                             [0] + list(np.multiply([i,j], split_image_size)) + [0],
                             [-1] + list(split_image_size) + [1])
                             )
        elif len(counts) == 3:
            for i in range(counts[0]):
                for j in range(counts[1]):
                    for k in range(counts[2]):
                        v_tile_inputs.append(tf.slice(
                                                self.X,
                                                [0] + list(np.multiply([i,j,k], split_image_size)) + [0],
                                                [-1] + list(split_image_size) + [1]
                                                )
                                            )
        # Our output components corresponding to each tile:
        v_tile_outputs = []
        for i, v in enumerate(v_tile_inputs):
        # Create the output that corresponds to the tile 'v'.
        # IMPORTANT, use 'scope' to share variables!
        # This means that the fully connected layers will share weights and biases
        # and each tile is treated with a homogenous subnetwork.
            v_output = self.DNN(v, reuse)
        # print v_output.get_shape().as_list()

            v_tile_outputs.append(v_output)
            reuse = True # We reuse our variables in subsequent tiles
        # Return the tiled outputs:
        self.model = tf.add_n(v_tile_outputs)

    def padImage(self, images):
        if len(self.context) == 3:
            return np.pad(images, 
                    (
                        (self.context[0], self.context[0]),
                        (self.context[1], self.context[1]),
                        (self.context[2], self.context[2]),
                        (0,0)
                    ),
                mode=self.padding
                )
        elif len(self.context) == 2:
            return np.pad(images, 
                    (
                        (self.context[0], self.context[0]),
                        (self.context[1], self.context[1]),
                        (0,0)
                    ),
                mode=self.padding
                )

    def padImages(self, images):
        if len(self.context) == 3:
            return np.pad(images, 
                    (   (0,0),
                        (self.context[0], self.context[0]),
                        (self.context[1], self.context[1]),
                        (self.context[2], self.context[2]),
                        (0,0)
                    ),
                mode=self.padding
                )
        elif len(self.context) == 2:
            return np.pad(images, 
                    (   (0,0),
                        (self.context[0], self.context[0]),
                        (self.context[1], self.context[1]),
                        (0,0)
                    ),
                mode=self.padding
                )

    def predict(self, X):
        if self.tile:
            X = self.padImages(X)
            return self.session.run(self.model, feed_dict={self.X: X}) * (self.parser.max_e - self.parser.min_e) + self.parser.min_e
        else:
            return self.session.run(self.model, feed_dict={self.X: X}) * (self.parser.max_e - self.parser.min_e) + self.parser.min_e

