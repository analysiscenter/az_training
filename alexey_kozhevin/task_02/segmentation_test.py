import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append('..')

from dataset import Pipeline, DatasetIndex, Dataset, B, V

from dataset.opensets import MNIST
from dataset.models.tf import UNet, LinkNet, FCN32, FCN8
from noised_mnist import NoisedMnist                                          # Batch subclass with loading and noise actions

IMAGE_SIZE = 64     # image size
MNIST_SIZE = 65000  # MNIST database size
BATCH_SIZE = 32     # batch size for NN training
MAX_ITER = 500      # number of iterations for NN training
DATA_FORMAT = 'channels_last'


def binary_iou(masks, predictions, data_format='channels_last'):
    ind = np.index_exp[:, :, :, 1] if data_format == 'channels_last' else np.index_exp[:, 1, :, :]
    predictions = predictions[ind]
    intersection = np.sum(np.logical_and((predictions > 0.5), masks))
    union = np.sum(np.logical_or((predictions > 0.5), masks))
    return intersection / union

def loss(labels, logits):
    labels = tf.transpose(labels, [0, 2, 3, 1])
    logits = tf.transpose(logits, [0, 2, 3, 1])
    res = tf.losses.softmax_cross_entropy(labels, logits)
    return res

def demonstrate_model(model, mnistset, data_format='channels_last'):
    level = 1           # the highest level of noise; [0, 1]
    n_fragments = 80    # number of noise fragments per image  
    size = 4            # size of noise fragment; 1, ..., 27
    distr = 'uniform'   # distribution of fragments of image; 'uniform' or 'normal'

    shape = (1, IMAGE_SIZE, IMAGE_SIZE) if data_format == 'channels_first' else (IMAGE_SIZE, IMAGE_SIZE, 1)

    placeholders_config = {
                           'images': {'shape': shape,
                                      'type': 'float32',
                                      'data_format': data_format,
                                      'name': 'reshaped_images'},
                    
                           'masks': {'shape': (IMAGE_SIZE, IMAGE_SIZE),
                                     'type': 'int32',
                                     'transform': 'ohe',
                                     'data_format': data_format,
                                     'classes': 2,
                                     'name': 'targets'}
                           }

    model_config = {'inputs': placeholders_config,
                    'input_block/inputs': 'images',
                    'batch_norm': {'momentum': 0.1},
                    'output': dict(ops=['proba']),
                    'loss': 'ce' if data_format == 'channels_last' else loss,
                    'optimizer': 'Adam'}

    train_feed_dict = {'images': B('images'),
                       'masks': B('masks')}        

    test_feed_dict = {'images': B('images'),
                      'masks': B('masks')}
    print('Create pipelines...')

    load_template = (Pipeline()
                 .random_location(IMAGE_SIZE)      # put MNIST at random location
                 .make_masks()                     # create mask for MNIST image location
                 .create_noise('mnist_noise', level, n_fragments, size, distr)
                 .add_noise())
    if data_format == 'channels_first':
        load_template = load_template.swap_axis()
    
    ppl_train = ((load_template << mnistset.train)                         # load data from file
            .init_model('static', model, 'NN', config=model_config)
            .init_variable('loss', init_on_each_run=list)
            .train_model('NN',
                         fetches='loss',
                         feed_dict=train_feed_dict,
                         save_to=V('loss'), mode='a'))

    print('Start training...')
    start = time()
    for i in range(MAX_ITER):
        ppl_train.next_batch(BATCH_SIZE, n_epochs=None, shuffle=True)
        #print(ppl_train.get_variable('loss')[-1])
    stop = time()
    print("Train time: {:05.3f} min".format((stop-start)/60))

    ppl_test = ((load_template << mnistset.test)
             .import_model('NN', ppl_train)
             .init_variable('predictions', init_on_each_run=list)
             .predict_model('NN',                                      
                           fetches='predicted_proba',
                           feed_dict=test_feed_dict,
                           save_to=V('predictions'),
                           mode='a'))

    batch = ppl_test.next_batch(100, n_epochs=None)
    images = batch.data.images
    masks = batch.data.masks
    noise = batch.data.noise
    predictions = ppl_test.get_variable('predictions')[-1]
    iou = binary_iou(masks, predictions, data_format)

    print(model.__name__, 'Test IoU: {0:.3f}'.format(iou))

    return iou

mnistset = MNIST(batch_class=NoisedMnist)
res = dict()
models = [UNet, FCN8, FCN32, LinkNet]
for model in models:
    res[model.__name__] = demonstrate_model(model, mnistset, DATA_FORMAT)

print('-' * 20)

for model in res:
    print(model, 'Test IoU: {0:.3f}'.format(res[model]))
