
import numpy as np
from os import listdir, path
from scipy import ndimage, misc
from six.moves import cPickle as pickle
import random

def load_pickle_subset(root_dir_path, image_count=10000, image_size=(256, 256, 3)):
    '''
    Returns two pickle files, sub_dataset and sub_labels, containing a subset of
    image_count images from the dataset in root_dir_path. In case the files do not exist, it
    creates them.
    '''

    # Prepare filenames
    sub_dataset_pickle_filename = path.join(root_dir_path, 'sub_dataset.pickle')

    try:

        # Load data set
        with open(sub_dataset_pickle_filename, 'rb') as f:
            dataset = pickle.load(f)

    except FileNotFoundError:

        # Get all the files in the folder
        image_filenames = listdir(root_dir_path)
        # Identify the number of images
        total_number_of_images = len(image_filenames)
        # if the number of images is lower than the image count...
        if image_count > total_number_of_images:
            image_count = total_number_of_images

        # allocate memory for all the images to pickle
        dataset = np.ndarray(shape=(image_count, image_size[0], image_size[1], image_size[2]), dtype=np.uint8)

        print('Pickling images...')
        # iterate for each of the images
        for i in range(0, image_count):
            # pick a random image
            random_filename = random.choice(image_filenames)
            # Read the image and copy it to the array
            dataset[i, :, :, :] = misc.imresize(ndimage.imread(path.join(root_dir_path, random_filename)),image_size)

        # Dump pickle files
        dump_pickle(dataset, sub_dataset_pickle_filename)

    return dataset



def dump_pickle(np_array, dst_filename):
    '''
    Given a numpy array with data, pickle it in a file with name dst_filename.
    '''

    try:
        with open(dst_filename, 'wb') as f:
            pickle.dump(np_array, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', dst_filename, ':', e)


