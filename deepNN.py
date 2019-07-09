from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d
import tensorflow as tf

def deepNN(x, reuse=False, branch='-branch-0'):
    network = conv_2d(x, 64, 3, strides=[1,2,2,1], activation = 'relu', scope='s1' + branch, reuse=reuse)
    network = conv_2d(network, 64, 3, strides=[1,2,2,1], activation = 'relu', scope='s2' + branch, reuse=reuse)

    network = conv_2d(network, 16, 4,  activation = 'relu', scope='s3' + branch, reuse=reuse)
    network = conv_2d(network, 16, 4,  activation = 'relu', scope='s4' + branch, reuse=reuse)
    network = conv_2d(network, 16, 4,  activation = 'relu', scope='s5' + branch, reuse=reuse)
    network = conv_2d(network, 16, 4,  activation = 'relu', scope='s6' + branch, reuse=reuse)
    network = conv_2d(network, 16, 4,  activation = 'relu', scope='s7' + branch, reuse=reuse)
    network = conv_2d(network, 16, 4,  activation = 'relu', scope='s8' + branch, reuse=reuse)

    network = conv_2d(network, 64, 3, strides=[1,2,2,1], activation = 'relu', scope='s9' + branch, reuse=reuse)

    network = conv_2d(network, 32, 3,  activation = 'relu', scope='s10' + branch, reuse=reuse)
    network = conv_2d(network, 32, 3,  activation = 'relu', scope='s11' + branch, reuse=reuse)
    network = conv_2d(network, 32, 3,  activation = 'relu', scope='s12' + branch, reuse=reuse)
    network = conv_2d(network, 32, 3,  activation = 'relu', scope='s13' + branch, reuse=reuse)
    network = fully_connected(network, 1024, activation = 'relu', scope ='s14' + branch, reuse=reuse)
    network = fully_connected(network, 7, activation ='relu', scope = 's15' + branch, reuse=reuse)
    return network

def lossFunction(Y, output, n_epoch):
    return tf.losses.mean_squared_error(Y, output)
