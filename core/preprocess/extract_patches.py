import numpy as np
import random
import configparser
from scipy import misc

from os import path, makedirs, listdir

import re


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]



def preprocess(image):
    return image




def generate_random_patches(img_folder, gt_folder, patch_height, patch_width, N_subimgs, output_folder_images, output_folder_labels):
    
    # get filenames
    image_filenames = sorted(listdir(img_folder), key=natural_key)
    gt_filenames = sorted(listdir(gt_folder), key=natural_key)

    # initialize output folders
    if len(image_filenames) > 0:
        if not path.exists(output_folder_images):
            makedirs(output_folder_images)
        if not path.exists(output_folder_labels):
            makedirs(output_folder_labels)

    # for each image
    for i in range(0, len(image_filenames)):
        # identify current image
        current_image_filename = image_filenames[i]
        current_gt_filename = gt_filenames[i]
        print('Processing image ' + current_image_filename)
        # open image and labels
        image = preprocess(misc.imread(path.join(img_folder, current_image_filename)))
        labels = (misc.imread(path.join(gt_folder, current_gt_filename)) / 255).astype('int32')
        # get image shape
        image_shape = image.shape
        # extract N_subimgs patches
        for j in range(0, N_subimgs):
            
            # generate a random center
            x = random.randint(patch_width, image_shape[0] - patch_width)
            y = random.randint(patch_height, image_shape[1] - patch_height)
            # get a random patch around the center
            if len(image_shape) > 2:
                random_patch = image[x - int(patch_width/2) : x + int(patch_width/2), y - int(patch_height/2) : y + int(patch_height/2), :]
            else:
                random_patch = image[x - int(patch_width/2) : x + int(patch_width/2), y - int(patch_height/2) : y + int(patch_height/2)]
            random_patch_labels = labels[x - int(patch_width/2) : x + int(patch_width/2), y - int(patch_height/2) : y + int(patch_height/2)]
            # save it
            misc.imsave(path.join(output_folder_images, current_image_filename[:-4] + str(j) + current_image_filename[-4:]), 
                random_patch)
            misc.imsave(path.join(output_folder_labels, current_gt_filename[:-4] + str(j) + current_gt_filename[-4:]), 
                random_patch_labels)                

