from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np


def get_plots(pipeline, n_examples=10, mode='sc', inverse=True, title=None):
    batch = pipeline.next_batch(n_examples, shuffle=True)
    images = np.squeeze(batch.data.images)
    if 's' in mode:
        proba = np.squeeze(pipeline.get_variable('predicted_proba')[-1])
        get_separate_masks(images, proba, inverse, title)
    if 'c' in mode:
        predicted_masks = np.squeeze(pipeline.get_variable('predicted_labels')[-1])
        get_masks(images, predicted_masks, inverse, title)

def get_separate_masks(images, proba, inverse, title):
    """Show results of segmentation networks
    """
    n_examples = len(images)
    grey_cmap ='Greys' + ('_r' if inverse else '')

    n_rows = 12
    plt.figure(figsize=(25, 2*n_examples))
    if title is not None:
        plt.suptitle(title, fontsize=26)
    for i in range(n_examples):
        plt.subplot(n_examples, n_rows, i*n_rows + 1)
        if i == 0:
            plt.title('Image')
        plt.imshow(images[i], vmin=0, vmax=1, cmap=grey_cmap)
        plt.axis('off')
        for j in range(10):
            plt.subplot(n_examples, n_rows, i*n_rows + 2 + j)
            if i == 0:
                plt.title('Mask {}'.format(j))
            plt.imshow(proba[i][:, :, j], cmap=grey_cmap)
            plt.axis('off')
        plt.subplot(n_examples, n_rows, i*n_rows + 12)
        if i == 0:
            plt.title('Not a digit')
        plt.imshow(proba[i][:, :, -1], cmap=grey_cmap)
        plt.axis('off')
    plt.subplots_adjust(bottom=0.2, right=0.8, top=0.9)
    cax = plt.axes([0.15, 0.15, 0.65, 0.01])
    plt.colorbar(cax=cax, orientation='horizontal')
    plt.show()

def get_masks(images, predicted_masks, inverse, title):
    """Show results of segmentation networks
    """
    n_examples = len(images)
    grey_cmap ='Greys' + ('_r' if inverse else '')
    
    cmap = colors.ListedColormap(['purple', 'r', 'green', 'blue', 'y', 'w', 'grey', 'magenta', 'orange', 'pink', 'black'])
    bounds=np.arange(-0.5, 11.5, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    n_rows = 2
    plt.figure(figsize=(25, 5*n_examples))
    if title is not None:
        plt.suptitle(title, fontsize=26)
    for i in range(n_examples):
        plt.subplot(n_examples, n_rows, i*n_rows + 1)
        if i == 0:
            plt.title('Image')
        plt.imshow(images[i], vmin=0, vmax=1, cmap=grey_cmap)
        plt.axis('off')
        plt.subplot(n_examples, n_rows, i*n_rows + 2)
        if i == 0:
            plt.title('Masks')
        plt.imshow(predicted_masks[i], cmap=cmap, norm=norm)
        plt.axis('off')
    plt.subplots_adjust(bottom=0.2, right=0.8, top=0.9)
    cax = plt.axes([0.15, 0.95, 0.65, 0.01])
    plt.colorbar(orientation='horizontal', cax=cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=range(10))
    plt.show()