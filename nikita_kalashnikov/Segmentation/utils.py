import PIL
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

def plot_augm_masks(batch):
    _, axis = plt.subplots(nrows=2, ncols=5, figsize=(17, 17))
    plt.subplots_adjust(top=0.5, bottom=0.1, left=0., right=1,
                        hspace=0.3, wspace=0.2)
    iter = zip(axis.flatten(), chain(batch.images[:5], batch.mask[:5]))
    for ax, image in iter:
        ax.axis('off')
        ax.imshow(np.array(image))

