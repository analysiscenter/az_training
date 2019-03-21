""" Plot function.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import softmax
from PIL import Image


def show_digits(batch, pred=None):
    """ The plot function.

    Parameters
    ----------
    batch : MyBatch
        Batch with augmentated data.

    pred : ndarray
        Array with predictions.
    """
    n_cols = 5
    n_rows = 1 + (batch.size - 1) // 5
    _, batch_axis = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(17, 17))
    plt.subplots_adjust(top=0.5, bottom=0.1, left=0., right=1,
                        hspace=0.3, wspace=0.2, )
    for i, axis in enumerate(batch_axis.flatten()):
        image = batch.images[i]
        if i == batch.size:
            break
        axis.set_title('Digit - {}'.format(batch.labels[i]))
        axis.axis('off')
        if not isinstance(image, Image.Image):
            image = batch.images[i].reshape(64, 64, -1).astype('uint8')
            image = Image.fromarray(image, mode='RGB')
        axis.imshow(image)
        if pred is not None:
            p = softmax([pred[i]])
            label = batch.labels[i]
            axis.set_title('Real - {} \n Pred - {}, \
                        conf - {:.2f}'.format(label, np.argmax(p), p.max()))

def plot_bar(metrics):
    """ Plot bars for true positive and false positive rates.

    Parameters
    ----------

    metrics : ClassificationMetric
        Object which stores classifacation metrics on the test data.
    """
    false_positive = metrics.evaluate('false_positive')
    true_positive = metrics.evaluate('true_positive')
    tp_rate = true_positive / (true_positive + false_positive)
    mpl_fig = plt.figure(figsize=(13, 5))
    axis = mpl_fig.add_subplot(111)
    x = range(10)
    axis.bar(x, tp_rate, width=0.6, color=(0.2588, 0.4433, 1.0))
    axis.bar(x, 1-tp_rate, width=0.6, color=(1.0, 0.5, 0.62),
             bottom=tp_rate)
    axis.set_xticks(x)
    axis.set_xlabel('classes')
    axis.set_ylabel('rate')
    axis.legend(['tp_rate', 'fp_rate'], bbox_to_anchor=(1.15, 1))
    axis.set_title('True positive and false positive rates for each class')
