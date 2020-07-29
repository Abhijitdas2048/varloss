import pickle
import numpy as np
import os
from keras import backend as K

path = "/home/cvprisi19/CE_keras/cifar-10-python/"
def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        d = pickle.load(f, encoding ='bytes')
        d_decoded = {}
        for k, v in d.items():
                    d_decoded[k.decode('utf8')] = v
                    d = d_decoded
    data = d['data']
    labels = d[label_key]
    
    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

def load_data():
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    #y_train = np.reshape(y_train, (len(y_train), 1))
    #y_test = np.reshape(y_test, (len(y_test), 1))
    #print(x_train.dtype)

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
	
    return (x_train, y_train), (x_test, y_test)
