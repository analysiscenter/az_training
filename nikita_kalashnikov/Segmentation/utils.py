""" Plot fucntions for segmentation task.
"""
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt

def plot_augm_masks(batch):
    """ Plots augmentated images and coresponding masks.
    """
    _, axis = plt.subplots(nrows=2, ncols=5, figsize=(17, 17))
    plt.subplots_adjust(top=0.5, bottom=0.1, left=0., right=1,
                        hspace=0.3, wspace=0.2)
    my_iter = zip(axis.flatten(), chain(batch.images[:5], batch.masks[:5]))
    for ax, image in my_iter:
        image = np.array(image)
        if image.shape[0] == 3:
            w = image.shape[1]
            image = image.reshape(w, w, -1)
            image /= 255
        ax.axis('off')
        ax.imshow(image)

def plot_pred_mask(test_batch, pred):
    """ Plots ground-truth masks and predicted ones.
    """
    w = test_batch.images[0].shape[1]
    _, axis = plt.subplots(nrows=3, ncols=3, figsize=(17, 17))
    iterator = chain(test_batch.images[:3], test_batch.masks[:3], pred[:3])
    for ax, image in zip(axis.flatten(), iterator):
        if image.shape[0] == 3:
            image = image.reshape(w, w, 3) / 255
        elif image.shape[0] == 2:
            image = np.where(image[1, :, :] > image[0, :, :], 1, 0)
        ax.imshow(image)
