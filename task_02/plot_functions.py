"""Visualization functions."""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive

def plot_examples(images, masks, proba):
    """Plot images, masks, and predicted probabilities.

    Parameters
    ----------

    images, masks, proba : list with one element which is list of np.arrays

    """
    images, masks = images[0], masks[0]
    n_examples = len(images)
    plt.figure(figsize=(15, 3.5*n_examples))
    for i in range(n_examples):
        plt.subplot(n_examples, 4, i*4 + 1)
        plt.title('Image')
        plt.imshow(images[i], vmin=0, vmax=1)
        plt.subplot(n_examples, 4, i*4 + 2)
        plt.title('Mask')
        plt.imshow(masks[i])
        plt.subplot(n_examples, 4, i*4 + 3)
        plt.title('Prediction')
        plt.imshow(proba[i][:, :, 1] > 0.5)
        plt.subplot(n_examples, 4, i*4 + 4)
        plt.title('Predicted probability')
        plt.imshow(proba[i][:, :, 1], vmin=0, vmax=1)


    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.1, 0.95, 0.75, 0.01])
    plt.colorbar(cax=cax, orientation='horizontal')
    plt.show()

def get_rgb(image, noise):
    """Get RGB image from greyscale image and noise.
    """
    rgb_noise = np.dstack([noise] * 3)
    rgb_image = np.zeros((*image.shape, 3))
    rgb_image[:, :, 1] = image
    rgb_image[:, :, 0] = image
    return np.max([rgb_noise, rgb_image], axis=0)

def plot_noised_image(image, noise):
    """Plot RGB image with highlighted MNIST digit.
    """
    plt.imshow(get_rgb(image, noise))
    plt.show()

def plot_examples_highlighted(images, noise, masks, proba):
    """Plot images, masks, and predicted probabilities.

    Parameters
    ----------

    images, masks, proba : list with one element which is list of np.arrays

    """
    images, noise, masks = images[0], noise[0], masks[0]
    n_examples = len(images)
    plt.figure(figsize=(15, 3.5*n_examples))
    for i in range(n_examples):
        plt.subplot(n_examples, 4, i*4 + 1)
        plt.title('Image')
        plt.imshow(get_rgb(images[i], noise[i]), vmin=0, vmax=1)
        plt.subplot(n_examples, 4, i*4 + 2)
        plt.title('Mask')
        plt.imshow(masks[i])
        plt.subplot(n_examples, 4, i*4 + 3)
        plt.title('Prediction')
        plt.imshow(proba[i][:, :, 1] > 0.5)
        plt.subplot(n_examples, 4, i*4 + 4)
        plt.title('Predicted probability')
        plt.imshow(proba[i][:, :, 1], vmin=0, vmax=1)


    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.3, 0.92, 0.5, 0.01])
    plt.colorbar(cax=cax, orientation='horizontal')
    plt.show()

def plot_example_interactive(images, masks, proba, index):
    """Plot images, masks, and predicted probabilities.

    Parameters
    ----------

    images, masks, proba : list with one element which is list of np.arrays

    index : int
        inex of image to plot

    """
    images, masks = images[0], masks[0]
    def interactive_f(threshold):
        """Function for interactive plot.
        """
        plt.figure(figsize=(18, 10))
        plt.subplot(131)
        plt.title('Image')
        plt.imshow(images[index])
        plt.subplot(132)
        plt.title('Mask')
        plt.imshow(masks[index])
        plt.subplot(133)
        plt.title('Prediction')
        plt.imshow(proba[index][:, :, 1] > threshold, vmin=0, vmax=1)
        plt.show()

    interactive_plot = interactive(interactive_f, threshold=(0.0, 1.0, 0.05))
    output = interactive_plot.children[-1]
    output.layout.height = '350px'
    return interactive_plot
