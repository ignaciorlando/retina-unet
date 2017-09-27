
import sys
from os import path, makedirs
# Import from sibling directory ..\api
sys.path.append(path.dirname(path.abspath(__file__)) + "/..")

import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam, SGD, rmsprop
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, BatchNormalization, Cropping2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import metrics

import numpy as np


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)



def build(image_size, config):

    # reorder according to backend
    if K.image_dim_ordering() == 'th':
        input_shape = (3,) + image_size
    else:
        input_shape = image_size + (3, )

    # get batch normalization from config
    batch_normalizations = config['batch_normalization'].split()
    # get dropout probability
    dropout_prob = float(config['dropout'])

    # Define input shape
    inputs = Input(input_shape)

    # First block - Downsampling
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    if dropout_prob > 0.0:
        conv1 = Dropout(dropout_prob)(conv1)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Second block - Downsampling
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    if dropout_prob > 0.0:
        conv2 = Dropout(dropout_prob)(conv2)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Third block - Downsampling
    conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    if dropout_prob > 0.0:
        conv3 = Dropout(dropout_prob)(conv3)
    conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Fourth block - Downsampling
    conv4 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    if dropout_prob > 0.0:
        conv4 = Dropout(dropout_prob)(conv4)
    conv4 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Fifth block - Downsampling
    conv5 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(pool4)
    if dropout_prob > 0.0:
        conv5 = Dropout(dropout_prob)(conv5)
    conv5 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv5)
    
    # Upsampling from conv5
    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=3)
    conv6 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(up6)
    if dropout_prob > 0.0:
        conv6 = Dropout(dropout_prob)(conv6)    
    conv6 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv6)
    
    # Upsampling from conv6
    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=3)
    conv7 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(up7)
    if dropout_prob > 0.0:
        conv7 = Dropout(dropout_prob)(conv7)
    conv7 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(conv7)

    # Upsampling from conv7
    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=3)
    conv8 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(up8)
    if dropout_prob > 0.0:
        conv8 = Dropout(dropout_prob)(conv8)    
    conv8 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv8)

    # Upsampling from conv8
    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=3)
    conv9 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(up9)
    if dropout_prob > 0.0:
        conv9 = Dropout(dropout_prob)(conv9)    
    conv9 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(conv9)
    
    # Last block
    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    # Define models inputs and outputs
    model = Model(inputs=inputs, outputs=conv10)

    return model


