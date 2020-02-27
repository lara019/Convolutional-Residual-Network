from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split


import numpy as np
import resnet
import time
import datetime
import os
import cv2

nb_classes = 10

def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist, fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("mnist: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = np_utils.to_categorical(y_train.astype('float32'))
    y_test = np_utils.to_categorical(y_test.astype('float32'))

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    data = (x_train, y_train, x_val, x_test, y_test, y_val)
    np.save('xtrainMnist.npy', data, allow_pickle=True)

    return (x_train, y_train, x_val), (x_test, y_test, y_val)


def load_cifar10():
    # The data, shuffled and split between train and test sets:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # subtract mean and normalize
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image
    X_train /= 128.
    X_test /= 128.

    return (X_train, Y_train), (X_test, Y_test)

def load_fashion():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist, fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print("mnist: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = np_utils.to_categorical(y_train.astype('float32'))
    y_test = np_utils.to_categorical(y_test.astype('float32'))

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    data = (x_train, y_train, x_val, x_test, y_test, y_val)
    np.save('xtrainFashion.npy', data, allow_pickle=True)

    return (x_train, y_train, x_val), (x_test, y_test, y_val)

def load_COIL_100_fromFile( ):
    file = '../DataSet/xtrainCOIL_100.npy'

    data = np.load(file, allow_pickle=True)
    x_train, y_train, x_val, x_test, y_test, y_val = data

    return (x_train, y_train, x_val), (x_test, y_test, y_val)


def load_coil100():
    """
    this function builds a data frame which contains 
    the path to image and the tag/object name using the prefix of the image name
    """
    print("load_COIL_100")
    return load_COIL_100_fromFile()
    #path = '/content/gdrive/My Drive/Colab Notebooks/CapsNet-Fashion-MNIST/datasets/coil-100/coil-100/'
    path = '../DataSet/coil-100/'
    numImages = 9
    #path = 'datasets/test/'
    files = os.listdir(path)

    x=[]
    y=[]
    t = 0

    for f in files:
        #print(os.path.join(path + f))
        if (os.path.splitext(f)[1] == ".png"):
            x.append( cv2.imread(os.path.join(path + f)) )
            y.append(os.path.splitext(f)[0] )

    x_train = np.asarray(x)

    ''' x_train = other_utils.rgb2gray(x_train)
    plot(x_train)

    x_train = resize28(x_train, 28, 28)
    plot(x_train)'''
    y_train = []
    for l in y:
        end = l.find('__',0)
        y_train.append(int(l[3:end])-1)

    uniq = np.unique(y_train)
    print("unique: ", uniq)

    y_train = np.asarray(y_train)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
    #x_train = x_train.reshape(-1, tam, tam, 1).astype('float32') / 255.
    x_train = x_train.astype('float32') / 255.
    #x_test = x_test.reshape(-1, tam, tam, 1).astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = np_utils.to_categorical(y_train.astype('float32'))
    y_test = np_utils.to_categorical(y_test.astype('float32'))

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    data = (x_train, y_train, x_val, x_test, y_test, y_val)
    np.save('xtrainCOIL_100.npy', data, allow_pickle=True)

    return (x_train, y_train, x_val), (x_test, y_test, y_val)
