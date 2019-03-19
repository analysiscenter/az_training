""" Plot function.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def show_conv(image, train, ker_size=5, filters=6):
    """ Plots features maps after the first convolution layer.

    Parameters
    ----------
    image: np.array
        Image of the digit.

    train: Batchflow.pipeline.Pipeline
        Traiing pipeline.

    ker_size: int
        The kernel size of the layer.

    filters: int
        The number of filters in the layer.
    """
    layers_gen = train.models.models['my_model'].model[0].modules()
    model = list(layers_gen)[3]
    param = list(model.parameters())
    weights = param[0].data.numpy()
    bias = param[1].data.numpy()
    shape = (image.shape[0] - ker_size + 1, image.shape[0] - ker_size + 1)
    res = []
    for w, b in zip(weights, bias):
        conv = np.zeros(shape=shape)
        for i in range(shape[0]):
            for j in range(shape[0]):
                conv[i, j] = np.sum(image[i:i + ker_size, j:j + ker_size] * w) + b
        res.append(Image.fromarray(conv))

    _, axis = plt.subplots(1, filters+1, figsize=(17, 17))

    axis[0].imshow(Image.fromarray(image))
    axis[0].axes.get_xaxis().set_visible(False)
    axis[0].axes.get_yaxis().set_visible(False)
    axis[0].set_title('Original')
    for i, conv in enumerate(res):
        axis[i+1].imshow(conv)
        axis[i+1].set_title('filter №{}'.format(i + 1))
        axis[i+1].axes.get_xaxis().set_visible(False)
        axis[i+1].axes.get_yaxis().set_visible(False)