import pickle
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive

def plot_examples(prediction, n_examples=5):
    images, masks, proba = prediction[0]
    IMAGE_SIZE = prediction[0][2][2].shape[0]
    n_examples = 3
    fig = plt.figure(figsize = (15, 12))
    for i in range(n_examples):
        plt.subplot(n_examples, 4, i*4 + 1)
        plt.title('Image')
        plt.imshow(images[i].reshape([IMAGE_SIZE, IMAGE_SIZE]), vmin=0, vmax=1)
        plt.subplot(n_examples, 4 , i*4 + 2)
        plt.title('Mask')
        plt.imshow(masks[i].reshape([IMAGE_SIZE, IMAGE_SIZE]))
        plt.subplot(n_examples, 4, i*4 + 3)
        plt.title('Prediction')
        plt.imshow(proba[i][:,:,1].reshape([IMAGE_SIZE, IMAGE_SIZE]) > 0.5)
        plt.subplot(n_examples, 4, i*4 + 4)
        plt.title('Predicted probability')
        plt.imshow(proba[i][:,:,1].reshape([IMAGE_SIZE, IMAGE_SIZE]), vmin=0, vmax=1)


    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.025, 0.8])
    plt.colorbar(cax=cax)
    plt.show()

def plot_example_interactive(prediction, index):
    images, masks, proba = prediction[0]
    IMAGE_SIZE = prediction[0][2][2].shape[0]
    def f(threshold):
        plt.figure(figsize= (18,10))
        plt.subplot(131)
        plt.title('Image')
        plt.imshow(images[index].reshape([IMAGE_SIZE, IMAGE_SIZE]))
        plt.subplot(132)
        plt.title('Mask')
        plt.imshow(masks[index].reshape([IMAGE_SIZE, IMAGE_SIZE]))
        plt.subplot(133)
        plt.title('Prediction')
        plt.imshow(proba[index][:,:,1].reshape([IMAGE_SIZE, IMAGE_SIZE])> threshold, vmin=0, vmax=1)
        plt.show()

    interactive_plot = interactive(f, threshold=(0.0, 1.0, 0.05))
    output = interactive_plot.children[-1]
    output.layout.height = '350px'
    return interactive_plot