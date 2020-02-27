"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping


import numpy as np
import resnet
import time
import datetime

from train import train_DA, train_sin_DA, train, save
from load import load_cifar10, load_mnist, load_fashion


# input image dimensions
img_rows, img_cols = 28, 28
img_channels = 1
nb_classes = 10
modelo = 'fashion'

def trainFashion(res_model):
	print('test trainFashion '+ str(res_model))

	(x_train, y_train, x_val), (x_test, y_test, y_val) = load_fashion()
	data = (x_train, y_train), (x_test, y_test)

	if int(res_model) == 18:
		model_r18 = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
		nombre = modelo + '_resnet18'
		train(model_r18, nombre, data)
	
	if int(res_model) == 34:
		model_r34 = resnet.ResnetBuilder.build_resnet_34((img_channels, img_rows, img_cols), nb_classes)
		nombre = modelo + '_resnet34'
		train(model_r34, nombre, data)
	
	if int(res_model) ==50:
		model_r50 = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
		nombre = modelo + '_resnet50'
		train(model_r50, nombre, data)


