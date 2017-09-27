from skimage import io
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.interpolation import zoom
import numpy as np



def get_image_data_generator(data_type, config):

    if (data_type=="train"):
        # datagen will be as follows:
        datagen_images = ImageDataGenerator(
            # For more information about these parameters
            # visit: https://keras.io/preprocessing/image/
            featurewise_center=bool(config['featurewise_center']),
            featurewise_std_normalization=bool(config['featurewise_std_normalization']),
            horizontal_flip=bool(config['horizontal_flip']),
            vertical_flip=bool(config['vertical_flip']),
            rescale=float(config['rescale'])
        )
        datagen_labels = ImageDataGenerator(
            horizontal_flip=bool(config['horizontal_flip']),
            vertical_flip=bool(config['vertical_flip']),
            rescale=1./255.
        )
    else: # data_type == "validation" or data_type == "test"
        # datagen will be only rescale
        datagen_images = ImageDataGenerator(
            rescale=float(config['rescale'])
        )
        datagen_labels = ImageDataGenerator(
            rescale=1./255.
        )
            
    # return our datagen objects
    return datagen_images, datagen_labels