#!/usr/bin/env python3

import sys
from os import path, makedirs

from configparser import ConfigParser

import keras.backend as K
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, CSVLogger
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from core.augmentation import data_augmentation
from core.preprocess import load_pickle_subset
from core.models import unet
from core.evaluation_metrics.evaluation_metrics import dice_coef_loss, dice_coef

from shutil import rmtree
from numpy import random
import ntpath



REPRESENTATIVE_SAMPLES = 1000



def configure_optimizer(config):
    optimizer = []
    # initialize the optimizer
    if config['optimizer']=='SGD':
        optimizer = SGD(lr=float(config['lr']),
                    decay=float(config['decay']),
                    momentum=float(config['momentum']),
                    nesterov=(config['nesterov']=='True'))
    return optimizer


def configure_model(config, image_shape):
    model = []
    # initialize the CNN architecture
    if config['architecture']=='unet':
        model = unet.build((image_shape[0], image_shape[1]), config)
    return model


def configure_evaluation_metrics(config):
    # split the metrics
    splitted_input = config['metrics'].split()
    # assign evaluation metrics
    metrics = []
    if 'accuracy' in splitted_input:
        metrics = metrics + ['accuracy']
    if 'dice_coef' in splitted_input:
        metrics = metrics + [dice_coef]        
    return metrics


def configure_data_generators(type_data, input_data_path, image_shape, batch_size, config, seed):
    # initialize the image data generator
    image_generator, label_generator = data_augmentation.get_image_data_generator(type_data, config)
    # images flow from directory
    images_flow = image_generator.flow_from_directory(
        path.join(input_data_path, type_data, 'img'),
        target_size=(image_shape[0], image_shape[1]),
        batch_size=batch_size,
        class_mode=None,
        seed=seed)
    # labels flow from directory
    labels_flow = label_generator.flow_from_directory(
        path.join(input_data_path, type_data, 'gt'),
        target_size=(image_shape[0], image_shape[1]),
        batch_size=batch_size,
        class_mode=None,
        color_mode='grayscale',
        seed=seed)
    # zip both the image and the label generators
    full_flow = zip(images_flow, labels_flow)

    return full_flow, image_generator, label_generator, images_flow, labels_flow

    
def configure_loss(config):
    # configure loss function
    if 'dice_coef_loss' in config['loss']:
        return [ dice_coef_loss ]
    else:
        return [ config['loss'] ]




def run_training(input_data_path, output_path, config_file):
    
    # =============== MODEL CONFIGURATION ===============

    # fix random seed for reproducibility
    seed = 7
    random.seed(seed)

    # append the name of the configuration file to the output path
    output_path = path.join(output_path, ntpath.basename(config_file)[:-4])
    name_experiment = ntpath.basename(config_file)[:-4]

    # create output directory if it does not exist
    if not path.exists(output_path):
        makedirs(output_path)

    # read the configuration file    
    config = ConfigParser()
    config.read(config_file)

    # write configuration file in the output folder
    with open(path.join(output_path, name_experiment), 'w') as config_output_file:
        config.write(config_output_file)

    # get image shape
    image_shape = [int(i) for i in config['input']['image_shape'].split() ]

    
    
    # =============== INITIALIZE THE MODEL ===============

    # initialize the CNN architecture
    model = configure_model(config['architecture'], image_shape)

    # initialize the optimizer
    optimizer = configure_optimizer(config['optimizer'])

    # assign evaluation metrics
    metrics = configure_evaluation_metrics(config['evaluation'])

    # initialize the loss function
    loss = configure_loss(config['loss'])

    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # check how the model looks like
    #plot_model(model, to_file=path.join(output_path, name_experiment + '_model.png')) 
    json_string = model.to_json()
    open(path.join(output_path, name_experiment +'_architecture.json'), 'w').write(json_string)      

    
    
    # =============== INITIALIZE DATA GENERATORS ===============

    # assign the batch sizes
    training_batch_size = int(config['training']['batch_size'])
    validation_batch_size = int(config['validation']['batch_size'])

    # initialize the image data generator
    # - for the training data
    train_generator, train_image_gen, train_label_gen, train_image_flow, train_label_flow = configure_data_generators(
        'train', input_data_path, image_shape, training_batch_size, config['augmentation'], seed)
    # - for the validation data
    val_generator, val_image_gen, val_label_gen, val_image_flow, val_label_flow = configure_data_generators(
        'validation', input_data_path, image_shape, validation_batch_size, config['augmentation'], seed)
    
    # fit the generators of the subset of the training set
    if (config['augmentation']['featurewise_center']=='True') or (config['augmentation']['featurewise_std_normalization']=='True'):
        # load pickles for computing statistics
        X_subset = load_pickle_subset.load_pickle_subset(path.join(input_data_path, 'train', 'img', '0'), 
                                                         REPRESENTATIVE_SAMPLES, 
                                                         image_shape)
        # fit to this subset
        train_image_gen.fit(X_subset[0:REPRESENTATIVE_SAMPLES,:,:,:], seed=seed)
        val_image_gen.fit(X_subset[0:REPRESENTATIVE_SAMPLES,:,:,:], seed=seed)

    
    
    # =============== CONFIGURE THE TRAINING PROCESS ===============

    # initialize Tensorboard callbacks
    tensorboard_cb = TensorBoard(log_dir=output_path, write_images=True)  
    checkpointer = ModelCheckpoint(filepath=path.join(output_path, name_experiment +'_best_weights.h5'), 
                                   verbose=1, monitor='val_loss', mode='auto', save_best_only=True)

    # get the number of epochs
    N_epochs = int(config['training']['epochs'])

    
    
    # ======================= TRAIN THE MODEL =======================
    
    model.fit_generator(train_generator, 
                        steps_per_epoch = train_image_flow.samples // training_batch_size,
                        epochs = N_epochs,
                        validation_data = val_generator,
                        validation_steps = val_image_flow.samples // validation_batch_size,
                        callbacks = [tensorboard_cb, checkpointer]
                        )


    # SAVE THE WEIGHTS
    model.save_weights(path.join(output_path, model.name + '_last_weights.h5'), overwrite=True)






    


def usage():
    print('ERROR: Usage: run_training.py <data_path> <output_path> [--image_shape] [--batch_size]')

import argparse
import sys

if __name__ == '__main__':

    if len(sys.argv) < 3:
        usage()
        exit()
    else:
        # create an argument parser to control the input parameters
        parser = argparse.ArgumentParser()
        parser.add_argument("data_path", help="directory with training/validation directories", type=str)
        parser.add_argument("output_path", help="directory to save the models", type=str)
        parser.add_argument("config_file", help="configuration file", type=str)

        args = parser.parse_args()

        # call the main function
        run_training(args.data_path, args.output_path, args.config_file)