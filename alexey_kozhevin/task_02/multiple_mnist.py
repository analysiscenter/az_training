#pylint:disable=attribute-defined-outside-init

"""Auxilary module to demonstrate segmentation networks"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from dataset.dataset.opensets import MNIST
from dataset.dataset import ImagesBatch, Pipeline, B, V, action, inbatch_parallel

class MultiMNIST(ImagesBatch):
    """Batch class for multiple MNIST images in random locations of image."""

    components = 'images', 'labels', 'masks'

    @action
    def normalize_images(self):
        """Normalize pixel values to (0, 1)."""
        self.images = self.images / 255
        return self

    @action
    @inbatch_parallel(init='indices', post='assemble', components=('images', 'masks'))
    def mix_digits(self, *args, **kwargs):
        """
        Creates image of image_shape with random number (not greater than max_digits)
        of MNIST digits in random places.
        """
        _ = args
        image_shape = kwargs['image_shape']
        max_digits = kwargs['max_digits']
        n_digits = np.random.randint(1, max_digits+1)
        digits = np.random.choice(len(self.images), min([n_digits, len(self.images)]))
        large_image = np.zeros(image_shape)
        mask = np.zeros(image_shape)

        for i in digits:
            image = np.squeeze(self.images[i])
            coord0 = np.random.randint(0, image_shape[0]-image.shape[0])
            coord1 = coord0 + image.shape[0]
            coord2 = np.random.randint(0, image_shape[1]-image.shape[1])
            coord3 = coord2 + image.shape[1]
            mask_region = mask[coord0:coord1, coord2:coord3]
            mask[coord0:coord1, coord2:coord3] = np.max([mask_region, (self.labels[i]+1)*(image > 0.1)], axis=0)
            old_region = large_image[coord0:coord1, coord2:coord3]
            large_image[coord0:coord1, coord2:coord3] = np.max([image, old_region], axis=0)
        large_image = np.expand_dims(np.array(large_image), axis=-1)
        mask = np.array(mask) - 1
        mask[mask == -1] = 10
        return large_image, mask

    @action
    def make_masks(self):
        """Create masks for images in batch."""
        masks = np.ones_like(self.images) * 10
        coords = np.where(self.images > 0)
        masks[coords] = self.labels[coords[0]]
        self.masks = np.squeeze(masks)
        return self

def demonstrate_model(model, filters=64, max_iter=100, batch_size=64, shape=(100, 100), mode='mnist'):
    """Train model and show plots to demonstrate result."""
    mnist = MNIST(batch_class=MultiMNIST)
    print('Demonstarate {}'.format(model.__name__))
    model_config = {'loss': 'softmax_cross_entropy',
                    'input_block/inputs': 'images',
                    'optimizer': {'name':'Adam',
                                  'use_locking': True},
                    'inputs':    {'images': {'shape': (None, None, 1)},
                                  'masks':  {'shape': (None, None),
                                             'classes': 11,
                                             'transform': 'ohe',
                                             'name': 'targets'}},
                    'filters': filters,
                    'num_blocks': 3,
                    'output': {'ops': ['proba', 'labels']}}


    train_template = (Pipeline()
                      .make_masks()
                      .init_variable('loss_history', init_on_each_run=list)
                      .init_variable('current_loss', init_on_each_run=0)
                      .init_model('dynamic', model, 'conv', config=model_config)
                      .train_model('conv', fetches='loss',
                                   feed_dict={'images': B('images'),
                                              'masks': B('masks')},
                                   save_to=V('current_loss'))
                      .update_variable('loss_history', V('current_loss'), mode='a'))

    train_pp = (train_template << mnist.train)

    print("Start training...")
    start = time()
    for _ in range(max_iter):
        train_pp.next_batch(batch_size, shuffle=True, n_epochs=None, drop_last=True, prefetch=0)
    print("Training time: {:4.2f} min".format((time() - start)/60))

    plt.title('Train loss')
    plt.plot(train_pp.get_variable('loss_history'))
    plt.ylim((0, 2))
    plt.show()

    if mode == 'multimnist':
        test_template = Pipeline().mix_digits(image_shape=shape, max_digits=5)
    elif mode == 'mnist':
        test_template = Pipeline().make_masks()

    test_template = (test_template
                     .import_model('conv', train_pp)
                     .init_variable('predicted_proba', init_on_each_run=list)
                     .init_variable('predicted_labels', init_on_each_run=list)
                     .predict_model('conv', fetches=['predicted_proba', 'predicted_labels'],
                                    feed_dict={'images': B('images'),
                                               'masks': B('masks')},
                                    save_to=[V('predicted_proba'), V('predicted_labels')], mode='a'))

    test_ppl = (test_template << mnist.test)
    get_plots(test_ppl, mode='c', inverse=True, n_examples=10)

def get_plots(pipeline, n_examples=10, mode='sc', inverse=True, title=None, batch_size=100):
    """Show results of segmentation networks."""
    batch = pipeline.next_batch(batch_size, shuffle=True)
    images = np.squeeze(batch.data.images)[:n_examples]
    if 's' in mode:
        proba = np.squeeze(pipeline.get_variable('predicted_proba')[-1])
        _get_separate_masks(images, proba, inverse, title)
    if 'c' in mode:
        predicted_masks = np.squeeze(pipeline.get_variable('predicted_labels')[-1])
        _get_masks(images, predicted_masks, inverse, title)

def _get_separate_masks(images, proba, inverse, title):
    n_examples = len(images)
    grey_cmap = 'Greys' + ('_r' if inverse else '')

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

def _get_masks(images, predicted_masks, inverse, title):
    n_examples = len(images)
    grey_cmap = 'Greys' + ('_r' if inverse else '')
    cmap = colors.ListedColormap(['purple', 'r', 'green', 'blue', 'y', 'w',
                                  'grey', 'magenta', 'orange', 'pink', 'black'])
    bounds = np.arange(-0.5, 11.5, 1)
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
