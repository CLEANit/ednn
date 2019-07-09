#!/usr/bin/env 

import tensorflow as tf
import numpy as np
import multiprocessing
import time
import threading
from random import shuffle
import subprocess
import sys
import tflearn
import copy
import h5py

'''
This function:
- reads the h5 files and passes data to the model
- filters out validation images and labels so we do not test with them
'''
def readPart(conn):
    while True:
        try:
            args = conn.recv()
            h5filename, batch_size, x_label, y_label, x_shape, y_shape, start, end = args
            if np.all([elem == None for elem in args]):
                print 'INFO: worker received cancellation signal. Will now join.'
                break
            h5file = h5py.File(h5filename, 'r', libver='latest', swmr=True)
            if batch_size == 1:
                x_data = h5file[x_label][start:end].reshape((1,) + x_shape)
                y_data = h5file[y_label][start:end].reshape((1,) + y_shape)
            else:
                x_data = h5file[x_label][start:end]
                y_data = h5file[y_label][start:end]
            conn.send([x_data, y_data])
        except:
            conn.send([None, None])
    print 'INFO: worker broke out of loop. Will now join'

def queueWorker(self, coord, h5filename, start_index, end_index):
    bX = self.batch_X
    bY = self.batch_Y
    validation_indices = self.loader.getValidationIndicesPerFile()
    parent_conn, child_conn = multiprocessing.Pipe()
    process = multiprocessing.Process(target=readPart, args=(child_conn,))
    process.start()
    try:
        while not self.coordinator.should_stop():
            for i in range((end_index  - start_index) / self.batch_size + 1):
                start = i * self.batch_size + start_index
                end = (i + 1) * self.batch_size + start_index
                if end > end_index:
                    end = end_index
                mask = np.ones(end - start, dtype=bool)
                indices = list(filter(lambda x: x < end and x >= start, validation_indices))
                mask[list(np.array(indices) - start)] = False
                parent_conn.send([h5filename, self.batch_size, self.config['x_label'], self.config['y_label'], self.loader.getXShape(), self.loader.getYShape(), start, end])
                x_data, y_data = parent_conn.recv()
                if x_data is None and y_data is None:
                    print 'Something went wrong with reading the HDF5 file...'
                    process.join()
                    break

                x_data = x_data[mask, ...]

                y_data = y_data[mask, ...]
                self.session.run(self.enqueue,
                    feed_dict={
                        bX: x_data,
                        bY: y_data
                    }
                    )
                if self.coordinator.should_stop():
                    self.logger.info('The coordinator has requested to stop.')
                    process.join()
                    break
    except tf.errors.CancelledError:
        self.logger.info('Received cancellation signal. Parent thread is shutting down.')
        parent_conn.send([None] * 8)
        process.join()     

'''
This is the main class, please read the comments function by function
'''
class Trainer():
    '''
    The initialization function:
    - reads the config file
    - inits the data queuing system
    - inits the model
    '''
    def __init__(self,
                 logger,
                 loader,
                 DNN, 
                 loss_function,
                 config,
                 tf_config=tf.ConfigProto(),
                 test=False
                ):
        start = time.time()
        self.logger = logger
        self.logger.info('Initializing trainer.')

        self.loader = loader
        self.config = config
        if self.config['n_threads'] == 'auto':
            # self.n_threads = 1  
            self.n_threads = multiprocessing.cpu_count()
        else:
            self.n_threads = config['n_threads']
        self.DNN = DNN
        self.loss_function = loss_function
        self.n_epoch = self.config['n_epoch']
        self.n_epochs_completed = None
        self.batch_size = self.config['batch_size']
        
        self.tf_config = tf_config
        self.tf_config.gpu_options.allow_growth = True
        self.tf_config.allow_soft_placement = True

        self.learning_rate = self.config['learning_rate']
        self.name = self.config['name']
        self.test = test
        self.n_gpu = self.config['n_gpu']
        # self.forces = self.config['forces']
        self.multi_scale = self.config['multi-scale']

        try:
            self.single_file = self.config['single-file']
        except:
            self.single_file = False
        self.branches = self.config['branches']
        self.best_validation_loss = np.inf

        subprocess.call(['mkdir -p checkpoints output'], shell=True)

        self.session = tf.Session(config=self.tf_config)
#        self.initModel()
        if not test:
            self.initDataPipe()
            self.logger.info('Initialized the data queuing system.')
        self.initModel()
        self.valid_X, self.valid_model = self.getNetworkForValidation()
        self.logger.info('Initializing trainer took %5.5f s.' % (time.time() - start))

    def queueWorker(self, coord, h5filename, start_index, end_index):
        bX = self.batch_X
        bY = self.batch_Y
        validation_indices = self.loader.getValidationIndicesPerFile()
        parent_conn, child_conn = multiprocessing.Pipe()
        process = multiprocessing.Process(target=readPart, args=(child_conn,))
        process.start()
        try:
            while not coord.should_stop():
                for i in range((end_index  - start_index) / self.batch_size + 1):
                    start = i * self.batch_size + start_index
                    end = (i + 1) * self.batch_size + start_index
                    if end > end_index:
                        end = end_index
                    mask = np.ones(end - start, dtype=bool)
                    indices = list(filter(lambda x: x < end and x >= start, validation_indices))
                    mask[list(np.array(indices) - start)] = False
                    parent_conn.send([h5filename, self.batch_size, self.config['x_label'], self.config['y_label'], self.loader.getXShape(), self.loader.getYShape(), start, end])
                    x_data, y_data = parent_conn.recv()
                    if x_data is None and y_data is None:
                        print 'Something went wrong with reading the HDF5 file...'
                        process.join()
                        break

                    x_data = x_data[mask, ...]

                    y_data = y_data[mask, ...]
                    self.session.run(self.enqueue,
                        feed_dict={
                        bX: x_data,
                        bY: y_data
                    }
                    )
                    if coord.should_stop():
                        self.logger.info('The coordinator has requested to stop.')
                        parent_conn.send([None] * 8)
                        process.join()
                        break
        except tf.errors.CancelledError:
            self.logger.info('Received cancellation signal. Parent thread is shutting down.')
            parent_conn.send([None] * 8)
            process.join()

    '''
    This function:
    - Creates the data queuing system
    - Use a random shuffle queue
    - Create placeholders to store the images
    - Spawn processes to pass data into the net
    '''
    def initDataPipe(self):
        self.queue = tf.RandomShuffleQueue(
                capacity=100,
                min_after_dequeue=90,
                dtypes=[tf.float32, tf.float32],
                shapes=(self.loader.getXShape(), self.loader.getYShape())
            )
        self.batch_X = tf.placeholder(tf.float32, shape=(None,) + self.loader.getXShape())
        self.batch_Y = tf.placeholder(tf.float32, shape=(None,) + self.loader.getYShape())

        self.enqueue = self.queue.enqueue_many([self.batch_X, self.batch_Y])

        self.coordinator = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=self.session, coord=self.coordinator)
        self.threads = []
        for i, (h5file, image_count) in enumerate(self.loader.getImageCountsPerFile().items()):
            chunk_size = image_count / self.n_threads
            chunk_leftover = image_count % self.n_threads
            for j in range(self.n_threads):
                start_index = j * chunk_size
                end_index = (j + 1) * chunk_size
                if j == self.n_threads - 1:
                    end_index += chunk_leftover
#                print start_index, end_index
                self.threads.append(threading.Thread(target=self.queueWorker, args=(self.coordinator, h5file, start_index, end_index)))

        for t in self.threads:
            t.daemon = True
            t.start()

    '''
    This function:
    - First checks if we have a multi-scale configuration (will cause errors if not)
    - returns the new image sizes
    '''
    def getXShapes(self):
        if self.multi_scale:
            tuples = []
            for branch in self.branches:
                tuples.append(tuple(np.add(self.loader.getXShape(), 2 * np.array(branch['context'] + [0]))))
            return tuples

    '''
    This function:
    - Averages the gradients across different GPUs
    - This was given to me from Kyle which was taken from a cifar tutorial
    '''
    def average_gradients(self, tower_grads):
       average_grads = []
       for grad_and_vars in zip(*tower_grads):
           grads = []
           for g,_ in grad_and_vars:
                   expanded_g = tf.expand_dims(g, 0)
                   grads.append(expanded_g)
           grad = tf.concat(axis=0,values=grads)
           grad = tf.reduce_mean(grad, 0)
           v = grad_and_vars[0][1]
           grad_and_var = (grad, v)
           average_grads.append(grad_and_var)
       return average_grads

    '''
    This function:
    - preprocess images depending on the branch type
    - Only supports: default, pooled, fft
    '''
    def preProcessImages(self, X, branch):
        if branch['type'] == 'default':
            return X
        elif branch['type'] == 'pooled':
            og_image_size = np.array(self.loader.getXShape()[:-1])
            if len(branch['context']) == 3:
                return tflearn.layers.conv.avg_pool_3d(X, list(og_image_size / np.array(branch['size'])), padding='valid')
            elif len(branch['context']) == 2:
                return tflearn.layers.conv.avg_pool_2d(X, list(og_image_size / np.array(branch['size'])), padding='valid')
        elif branch['type'] == 'fftr':
            if len(branch['context']) == 3:
                return tf.real(tf.fft3d(tf.cast(X, tf.complex64)))
            elif len(branch['context']) == 2:
                return tf.real(tf.fft2d(tf.cast(X, tf.complex64)))
        elif branch['type'] == 'ffti':
            if len(branch['context']) == 3:
                return tf.imag(tf.fft3d(tf.cast(X, tf.complex64)))
            elif len(branch['context']) == 2:
                return tf.imag(tf.fft2d(tf.cast(X, tf.complex64)))


    '''
    This function:
    - Creates the multiScale DNN
    '''
    def multiScaleDNN(self):
        # All the input, consisting of `n_tiles` tiles with dimensions from `tile_shape`.
        # The dimensions in tile_shape (in our tested case 2D, eg, width and height) account for padding.
        # Split the input into a list of `n_tiles` tensors, 
        
        all_gradients = []
        all_losses = []
        fc_reuse = False
        for gpu_num in range(self.n_gpu):
            outputs = []
            if not self.test:
                X, Y = self.nextBatch()
            else:
                X, Y = self.X, self.Y
            with tf.device('/gpu:' + str(gpu_num)):
                with tf.name_scope('GPU_{0}'.format(gpu_num)) as scope:
                    with tf.variable_scope(tf.get_variable_scope()):
                        for branch_num, branch in enumerate(self.branches):
                            if gpu_num == 0:
                                reuse = False
                            else:
                                reuse = True
                            self.logger.info('Creating branch %i on GPU %i. Reusing network = %s' % (branch_num, gpu_num, str(reuse)))
                            focus = branch['focus']
                            context = branch['context']
                            X_cp = self.preProcessImages(X, branch)
                            og_image_size = X_cp.get_shape().as_list()[1:-1]
                            X_cp = tf.py_func(self.padImages, [X_cp, context, branch['padding']], tf.float32)
                            v_tile_inputs = []
                            split_image_size = np.add(focus, 2 * np.array(context))
                            counts = og_image_size / np.array(focus)
                            
                            if not np.sum(counts):
                                self.logger.error('Sum of context and focus should be less or equal to image dimensions.')
                                exit(-1)

                            if len(counts) == 2:
                                for i in range(counts[0]):
                                    for j in range(counts[1]):
                                        v_tile_inputs.append(tf.slice(X_cp,
                                                 [0] + list(np.multiply([i,j], focus)) + [0],
                                                 [-1] + list(split_image_size) + [1])
                                                 )
                            elif len(counts) == 3:
                                for i in range(counts[0]):
                                    for j in range(counts[1]):
                                        for k in range(counts[2]):
                                            v_tile_inputs.append(tf.slice(
                                                                    X_cp,
                                                                    [0] + list(np.multiply([i,j,k], focus)) + [0],
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
                                v_output = self.DNN(v, reuse, '-branch-' + str(branch_num))
                            # print v_output.get_shape().as_list()

                                v_tile_outputs.append(v_output)
                                reuse = True # We reuse our variables in subsequent tiles
                            # Return the tiled outputs:
                            self.logger.info('Created branch %i on GPU %i.' % (branch_num, gpu_num))
                            outputs.append(tf.add_n(v_tile_outputs))
                        if len(self.branches) > 1:
                            output = tf.concat(outputs, 1)
                            output = tflearn.layers.core.fully_connected(output, 1, reuse=fc_reuse, scope='multi-scale-fc', name='multi-scale-fc')
                            loss = self.loss_function(Y, output, self.n_epochs_completed)
                        else:
                            loss = self.loss_function(Y, outputs[0], self.n_epochs_completed)
                        all_losses.append(loss)
                        grad = self.optimizer.compute_gradients(loss)
                        all_gradients.append(grad)
                        fc_reuse = True

        gradients = self.average_gradients(all_gradients)
        return gradients, tf.reduce_mean(all_losses)

    '''
    This function:
    - Sets up a regular DNN on multiple GPU
    '''
    def parallelNet(self):
        all_gradients = []
        all_losses = []
        reuse = False
        for gpu_num in range(self.n_gpu):
            if not self.test:
                X, Y = self.nextBatch()
            else:
                X, Y = self.X, self.Y
            with tf.device('/gpu:' + str(gpu_num)):
                with tf.name_scope('GPU_{0}'.format(gpu_num)) as scope:
                    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                        model = self.DNN(X, reuse=reuse)
                        loss = self.loss_function(Y, model, self.n_epochs_completed)
                        grad = self.optimizer.compute_gradients(loss)
                        all_losses.append(loss)
                        all_gradients.append(grad)
                        reuse = True
        self.model = model
        gradients = self.average_gradients(all_gradients)
        return gradients, tf.reduce_mean(all_losses)

    '''
    This function:
    - inits everything we need to run 
    - This is the main initialization function
    '''
    def initModel(self):
        # init the model

        if self.test:
            self.X, self.Y = tf.placeholder(tf.float32, shape=(None,) + self.loader.getXShape()), tf.placeholder(tf.float32, shape=(None,) + self.loader.getYShape())


        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        if self.multi_scale:
            if self.single_file:
                self.image_placeholders = []
                self.models = []
                for branch_num, branch in enumerate(self.branches):
                    gpu_images = []
                    gpu_models = []
                    reuse = False
                    for gpu_num in range(self.n_gpu):
                        with tf.device('/gpu:' + str(gpu_num)):
                            with tf.name_scope('GPU_{0}'.format(gpu_num)) as scope:
                                with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                                    img = tf.placeholder(tf.float32, shape=(None,) + tuple(np.add(branch['focus'], 2 * np.array(branch['context']))) + (1,))
                                    model = self.DNN(img, reuse=reuse, branch='-branch-' + str(branch_num))
                                    reuse = True
                                    print gpu_num, branch_num, reuse
                                    gpu_images.append(img)
                                    gpu_models.append(model)
                    self.image_placeholders.append(gpu_images) 
                    self.models.append(gpu_models)
                self.session.run(tf.global_variables_initializer())
                self.grads, self.loss = self.getEDNNOutput()

            else:
                self.grads, self.loss = self.multiScaleDNN()
        else:
            self.grads, self.loss = self.parallelNet()
        
        self.step = self.optimizer.apply_gradients(self.grads)

        self.session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1e9)

        if self.test:
            self.saver.restore(self.session, tf.train.latest_checkpoint('./checkpoints'))
            self.X, self.model = self.getNetworkForValidation()
            self.logger.info('Successfully restored graph from latest checkpoint. Will test with this model.')
        else:
            try:
                self.saver.restore(self.session, tf.train.latest_checkpoint('./checkpoints'))
                self.logger.info('Successfully restored graph from latest checkpoint. Continuing training.')
                self.loss_vs_epoch = open('output/loss_vs_epoch.dat', 'r')
                self.n_epochs_completed = int(float(self.loss_vs_epoch.readlines()[-1].split()[0])) + 1
                self.logger.info('Number of epochs completed: %i' % (self.n_epochs_completed))
                self.n_batches_completed = self.n_epochs_completed * self.loader.getTotalImages() / self.batch_size / self.n_gpu
                loss_vs_batch = np.loadtxt('output/loss_vs_batch.dat')
                np.savetxt('output/loss_vs_batch.dat', loss_vs_batch[:self.n_batches_completed, :])
                
                self.loss_vs_batch = open('output/loss_vs_batch.dat', 'a')

                self.loss_vs_epoch = open('output/loss_vs_epoch.dat', 'a')

                if self.multi_scale:
                    self.branch_dep = open('output/branch_dep.dat', 'a')


            except:
                self.logger.warning('Could not open latest checkpoint file. Starting training from scratch.')
                self.loss_vs_batch = open('output/loss_vs_batch.dat', 'w')
                self.loss_vs_epoch = open('output/loss_vs_epoch.dat', 'w')

                if self.multi_scale:
                    self.branch_dep = open('output/branch_dep.dat', 'w')

                self.n_epochs_completed = 0
                self.n_batches_completed = 0
    
    def outputLoss(self):
        self.loss_vs_batch.write('%5.10e\t%5.10e\n' % (self.n_batches_completed, self.current_loss))
        self.loss_vs_batch.flush()

        # self.loss_vs_batch_avg = (self.loss_vs_batch_avg * (self.n_batches_completed - 1) + self.current_loss) / self.n_batches_completed
        # self.run_loss_vs_batch.write('%5.10e\t%5.10e\n' % (self.n_batches_completed, self.loss_vs_batch_avg))
        # self.run_loss_vs_batch.flush()

    def batchDep(self):
        fc = [v for v in tf.global_variables() if v.name == 'multi-scale-fc/W:0'][0]
        fc_val = self.session.run(fc)
        self.branch_dep.write('%i\t' % (self.n_epochs_completed))
        for branch in fc_val:
            branch = branch[0]
            self.branch_dep.write('\f\t' % (branch))
        self.branch_dep.write('\n')
        self.branch_dep.flush()

    def checkPoint(self, current_time):
        # tflearn.is_training(False, session=self.session)
        self.validation_loss = self.validation()
        if self.multi_scale and len(self.branches) > 1:
            self.batchDep()
        # tflearn.is_training(True, session=self.session)
        self.logger.info('Epoch: %1.5e\tBatch: %1.5e\tTraining loss: %1.5e\tValidation loss: %1.5e\tTime: %1.5e (min)' % (self.n_epochs_completed, self.n_batches_completed, self.current_loss, self.validation_loss, current_time / 60.))
        self.loss_vs_epoch.write('%5.10e\t%5.10e\t%5.10e\t%5.10e\n' % (self.n_epochs_completed, self.current_loss, self.validation_loss, current_time / 60.))
        self.loss_vs_epoch.flush()
        # if self.n_epochs_completed == 0:
        #     self.loss_vs_epoch_avg = self.current_loss
        #     self.loss_vs_epoch_val_avg = self.validation_loss
        # else:
        #     self.loss_vs_epoch_avg = (self.loss_vs_epoch_avg * (self.n_epochs_completed - 1) + self.current_loss) / self.n_epochs_completed
        #     self.loss_vs_epoch_val_avg = (self.loss_vs_epoch_val_avg * (self.n_epochs_completed - 1) + self.validation_loss) / self.n_epochs_completed
        # self.run_loss_vs_epoch.write('%5.10e\t%5.10e\t%5.10e\n' % (self.n_epochs_completed, self.loss_vs_epoch_avg, self.loss_vs_epoch_val_avg))
        # self.run_loss_vs_epoch.flush()

        if self.validation_loss < self.best_validation_loss:
            self.best_validation_loss = self.validation_loss
            self.saver.save(self.session, 'checkpoints/best_' + self.name, global_step=self.n_epochs_completed,write_meta_graph=False)
        self.saver.save(self.session, 'checkpoints/' + self.name, global_step=self.n_epochs_completed, write_meta_graph=False)

    def nextBatch(self):
        return self.queue.dequeue_many(self.batch_size)

    def getBatchSize(self):
        return self.batch_size * self.n_gpu

    def train(self):
        start = time.time()
        while self.n_epochs_completed < self.n_epoch:
            _, self.current_loss = self.session.run([self.step, self.loss])
            self.n_batches_completed += 1 
            self.outputLoss()
            self.n_epochs_completed = self.n_gpu * self.n_batches_completed * self.batch_size / self.loader.getTotalImages()
            if self.n_batches_completed % (self.loader.getTotalImages() / self.getBatchSize()) == 0:
                #_ = p.amap(self.checkPoint, [(time.time() - start)])
                try:
                    self.checkpoint_thread.join()
                    self.logger.info('Checkpointing thread joined.')
                except:
                    self.logger.warning('Unable to join checkpoint thread. If this is the first epoch completed then you can ignore this message.')
                    pass
                if self.n_epochs_completed < self.n_epoch:
                    self.checkpoint_thread = threading.Thread(target=self.checkPoint, args=(time.time() - start, ))
                    self.checkpoint_thread.daemon = True
                    self.checkpoint_thread.start()
                    start = time.time()
                # self.checkPoint((time.time() - start))
        self.session.run(self.queue.close(cancel_pending_enqueues=True))
        self.coordinator.request_stop()
        self.coordinator.join(self.threads)
        self.logger.info('Training has finished!')

    def breakDownImages(self, h5file, indices):
        new_images = []
        labels = []
        cell = h5file.attrs['cell_dimensions']
        image_shape = self.loader.getXShape()[:-1]
        image_shape = np.add(np.array(self.focus) * 2, np.array(image_shape))
        images = self.padImages(h5file[self.config['x_label']][indices]).reshape((len(indices),) + tuple(image_shape))
        for i in range(len(indices)):
            forces = h5file[self.config['y_label']][indices[i]]
            distances = h5file['atomic_positions'][indices[i]]
            for force, dist in zip(forces, distances):
                if len(self.focus) == 3:
                    xbin = int(dist[0] / (cell[0][0] / self.loader.getXShape()[0])) + self.focus[0]
                    ybin = int(dist[1] / (cell[1][1] / self.loader.getXShape()[1])) + self.focus[1]
                    zbin = int(dist[2] / (cell[2][2] / self.loader.getXShape()[2])) + self.focus[2]
                    image = images[i][xbin - (self.focus[0] / 2): xbin + (self.focus[0] / 2) + 1, ybin - (self.focus[1] / 2): ybin + (self.focus[1] / 2) + 1, zbin - (self.focus[2] / 2): zbin + (self.focus[2] / 2) + 1]
                    new_images.append(image.reshape(tuple(image.shape) + (1,)))
                    labels.append(force.reshape(3))
        new_images = np.array(new_images)
        labels = np.array(labels)
        return new_images, labels


    def padImages(self, images, context, padding):
        if len(context) == 3:
            return np.pad(images, 
                    (   (0,0),
                        (context[0], context[0]),
                        (context[1], context[1]),
                        (context[2], context[2]),
                        (0,0)
                    ),
                mode=padding
                )
        elif len(context) == 2:
            return np.pad(images, 
                    (   (0,0),
                        (context[0], context[0]),
                        (context[1], context[1]),
                        (0,0)
                    ),
                mode=padding
                )
    
    def singlePrediction(self, X):
        if self.tile:
            X = self.padImages(X)
            X = X.reshape((1,) + self.getXShape())
            return self.session.run(self.model, feed_dict={self.X: X})[0]
        else:
            X = X.reshape((1,) + self.loader.getXShape())
            return self.session.run(self.model, feed_dict={self.X: X})[0]

    def getEDNNOutput(self):
        # All the input, consisting of `n_tiles` tiles with dimensions from `tile_shape`.
        # The dimensions in tile_shape (in our tested case 2D, eg, width and height) account for padding.
        # Split the input into a list of `n_tiles` tensors, 
        
        all_gradients = []
        all_losses = []
        fc_reuse = False
        for gpu_num in range(self.n_gpu):
            outputs = []
            if not self.test:
                X, Y = self.nextBatch()
            else:
                X, Y = self.X, self.Y
            with tf.device('/gpu:' + str(gpu_num)):
                with tf.name_scope('GPU_{0}'.format(gpu_num)) as scope:
                    with tf.variable_scope(tf.get_variable_scope()):
                        for branch_num, branch in enumerate(self.branches):
                            focus = branch['focus']
                            context = branch['context']
                            X_cp = self.preProcessImages(X, branch)
                            og_image_size = X_cp.get_shape().as_list()[1:-1]
                            X_cp = tf.py_func(self.padImages, [X_cp, context, branch['padding']], tf.float32)
                            v_tile_inputs = []
                            split_image_size = np.add(focus, 2 * np.array(context))
                            counts = og_image_size / np.array(focus)
                            
                            if not np.sum(counts):
                                self.logger.error('Sum of context and focus should be less or equal to image dimensions.')
                                exit(-1)

                            if len(counts) == 2:
                                for i in range(counts[0]):
                                    for j in range(counts[1]):
                                        v_tile_inputs.append(tf.slice(X_cp,
                                                 [0] + list(np.multiply([i,j], focus)) + [0],
                                                 [-1] + list(split_image_size) + [1])
                                                 )
                            elif len(counts) == 3:
                                for i in range(counts[0]):
                                    for j in range(counts[1]):
                                        for k in range(counts[2]):
                                            v_tile_inputs.append(tf.slice(
                                                                    X_cp,
                                                                    [0] + list(np.multiply([i,j,k], focus)) + [0],
                                                                    [-1] + list(split_image_size) + [1]
                                                                    )
                                                                )
                            # Our output components corresponding to each tile:
                            v_tile_outputs = []
                            for i, v in enumerate(v_tile_inputs):
                                v_output = self.session.run(self.models[branch_num][gpu_num], feed_dict={self.image_placeholders[branch_num][gpu_num]: self.session.run(v)})
                                v_tile_outputs.append(v_output)

                            outputs.append(tf.add_n(v_tile_outputs))
                        if len(self.branches) > 1:
                            output = tf.concat(outputs, 1)
                            output = tflearn.layers.core.fully_connected(output, 1, reuse=fc_reuse, scope='multi-scale-fc', name='multi-scale-fc')
                            loss = self.loss_function(Y, output, self.n_epochs_completed)
                        else:
                            loss = self.loss_function(Y, outputs[0], self.n_epochs_completed)
                        all_losses.append(loss)
                        grad = tf.gradients(loss, self.models[0][0])
                        # grad = self.optimizer.compute_gradients(loss)
                        all_gradients.append(grad)
                        fc_reuse = True
        print all_losses, all_gradients, 
        gradients = self.average_gradients(all_gradients)
        return gradients, tf.reduce_mean(all_losses)


    def getNetworkForValidation(self):
        X = tf.placeholder(tf.float32, shape=(None,) + self.loader.getXShape())
        if self.multi_scale:
            outputs = []
            with tf.device('/gpu:' + str(0)):
                with tf.name_scope('GPU_0') as scope:
                    with tf.variable_scope(tf.get_variable_scope()):
                        for branch_num, branch in enumerate(self.branches):
                            focus = branch['focus']
                            context = branch['context']
                            X_cp = self.preProcessImages(X, branch)
                            og_image_size = X_cp.get_shape().as_list()[1:-1]
                            X_cp = tf.py_func(self.padImages, [X_cp, context, branch['padding']], tf.float32)
                            v_tile_inputs = []
                            split_image_size = np.add(focus, 2 * np.array(context))
                            counts = og_image_size / np.array(focus)
                            
                            if len(counts) == 2:
                                for i in range(counts[0]):
                                    for j in range(counts[1]):
                                        v_tile_inputs.append(tf.slice(X_cp,
                                                 [0] + list(np.multiply([i,j], focus)) + [0],
                                                 [-1] + list(split_image_size) + [1])
                                                 )
                            elif len(counts) == 3:
                                for i in range(counts[0]):
                                    for j in range(counts[1]):
                                        for k in range(counts[2]):
                                            v_tile_inputs.append(tf.slice(
                                                                    X_cp,
                                                                    [0] + list(np.multiply([i,j,k], focus)) + [0],
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
                                v_output = self.DNN(v, True, '-branch-' + str(branch_num))
                            # print v_output.get_shape().as_list()

                                v_tile_outputs.append(v_output)
                                reuse = True # We reuse our variables in subsequent tiles
                            # Return the tiled outputs:
                            outputs.append(tf.add_n(v_tile_outputs))
                        if len(self.branches) > 1:
                            output = tf.concat(outputs, 1)
                            model = tflearn.layers.core.fully_connected(output, 1, reuse=True, scope='multi-scale-fc')
                        else:
                            model = outputs[0]
                    return X, model
        else:
            model = self.DNN(X, reuse=True)
            return X, model

    def validation(self):
        if self.loader.getTotalValidationImages() == 0:
            return 0.
        validation_indices = self.loader.getValidationIndicesPerFile()
        mse = 0.
        counter = 0
        X, model = self.valid_X, self.valid_model
        # X, model = self.getNetworkForValidation()
        for h5filename, indices in validation_indices.items():
            h5file = h5py.File(h5filename, 'r')
            for i in range(len(indices) / self.batch_size + 1):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                if end > len(indices) - 1:
                    end = len(indices)
                if start == end:
                     break
                x_data = h5file[self.config['x_label']][indices[start:end]]
                y_data = h5file[self.config['y_label']][indices[start:end]]
                P = self.session.run(model, feed_dict={X:x_data})
                l = np.sum((y_data-P[0:end - start])**2)
                mse += l
        return mse / self.loader.getTotalValidationImages()

    def predict(self):
        true_vs_pred = open('output/true_vs_pred.dat', 'w')

        if np.prod(self.loader.getYShape()) != self.loader.getYShape()[0]:
            f = h5py.File('output/predictions.h5', 'w')
            f.create_dataset('predictions', shape=(self.loader.getTotalImages(),) + self.loader.getYShape())
            f.create_dataset('trues', shape=(self.loader.getTotalImages(),) + self.loader.getYShape())
        
        for h5file in self.loader.getH5Files():
            for i in range(self.loader.getTotalImages() / self.batch_size):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                if end > self.loader.getTotalImages():
                    end = -1
                if self.batch_size == 1:
                    x_data = h5file[self.config['x_label']][start:end].reshape((1,) + self.loader.getXShape())
                    y_data = h5file[self.config['y_label']][start:end].reshape((1,) + self.loader.getYShape())
                else:
                    x_data = h5file[self.config['x_label']][start:end]              
                    y_data = h5file[self.config['y_label']][start:end]
                
                P = self.session.run(self.model, feed_dict={self.X: x_data})

                # if we are not predicting one number, let's write out the predictions to an h5 file
                if np.prod(self.loader.getYShape()) != self.loader.getYShape()[0]:
                    f['predictions'][start:end] = P 
                    f['trues'][start:end] = y_data

                else:
                    for y, p in zip(y_data, P):
                        for elem_y, elem_p in zip(y, p):
                            true_vs_pred.write('%5.10e\t%5.10e\t' % (elem_y, elem_p))
                        true_vs_pred.write('\n')
        true_vs_pred.close()






