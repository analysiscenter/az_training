import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import softmax
from PIL import Image


def show_digits(batch, pred=None):
    n_cols = 5
    n_rows = 1 + (batch.size - 1) // 5
    _, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(17, 17))
    plt.subplots_adjust(top=0.5, bottom=0.1, left=0., right=1,
                        hspace=0.3, wspace=0.2, )
    for i, axis in enumerate(ax.flatten()):
        image = batch.images[i]
        if i == batch.size:
            break
        axis.set_title('Digit - {}'.format(batch.labels[i]))
        axis.axis('off')
        if type(image) is not Image.Image:
            image = batch.images[i].reshape(66, 66, -1).astype('uint8')
            image = Image.fromarray(image, mode='RGB')
        axis.imshow(image)
        if pred is not None:
            p = softmax([pred[i]])
            label = np.argmax(p)
            axis.set_title('Real - {} \n Pred - {}, \
                        conf - {:.2f}'.format(label, np.argmax(p), p.max()))
