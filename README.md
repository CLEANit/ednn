## Extensive deep neurals networks (EDNNs)

Please cite this if you use the code: 
```
@article{mills2019extensive,
  title={Extensive deep neural networks for transferring small scale learning to large scale systems},
  author={Mills, Kyle and Ryczko, Kevin and Luchak, Iryna and Domurad, Adam and Beeler, Chris and Tamblyn, Isaac},
  journal={Chemical Science},
  year={2019},
  publisher={Royal Society of Chemistry}
}
```

This repo is a working version of EDNNs using the Tensorflow framework.

### Features
- It is the multi-scale version, which means it can handle short and long range interactions. 
- It can utilize multiple GPUs
- It can read very large datasets using HDF5

### Installation
Run the install script to set your path:
```
bash install.sh
```

### Example Usage
Check out the example directory. 

1) We have a file that contains the deep neural network and the loss function (deepNN.py). Here you can define a network, but you must not modify the function names. 

2) Your training data (hdf5 format) goes into a directory called 'train' and your testing data (again, hdf5 format) goes into a directory called 'test'. You may have multiple files, but this feature has not been thoroughly tested yet. 

3) See the input.yaml file, here we can set hyperparameters, number of epochs, batch sizes, number of gpus, etc. We want to make sure the labels in our HDF5 files are correctly put here. The focus and context sizes are defined in the multi-scale part of the input file.

4) To run using GPUs:
```
ednn --train
```
To run without GPUs:
```
CUDA_VISIBLE_DEVICES= ednn --train
```
Once finished, you can then run
```
ednn --test
```
You will see that directories called output and checkpoints are created. You can look in the output directory to see the loss data, as well as your predictions for the model.
