import sys
from time import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append("..")
from dataset.dataset.image import ImagesBatch
from dataset import Pipeline, DatasetIndex, Dataset
from dataset.dataset.opensets import MNIST

from vgg import VGGModel

BATCH_SIZE = 256
mnist = MNIST()

train_pp = (mnist.train.p
            .init_variable('loss_history', init_on_each_run=list)
            .init_variable('current_loss', init_on_each_run=0)
            .init_variable('pred_label', init_on_each_run=list)
            .init_variable('images_shape', (0, 0))
            .update_variable('images_shape', lambda batch: batch.images.shape[1:])
            .init_model('dynamic', VGGModel, 'VGG',
                        config={'session': {'config': tf.ConfigProto(allow_soft_placement=True)},
                                'loss': 'sigmoid_cross_entropy',
                                'optimizer': {'name':'Adam', 'use_locking': True},
                                'images_shape': 'images_shape',
                                'b_norm': True,
                                'momentum': 0.1})
            .train_model('VGG',
                         fetches=['loss', 'predicted_labels'],
                         feed_dict={'images': 'images',
                                    'labels': 'labels',
                                    'training': True},
                         save_to=['current_loss', 'pred_label'])
            .print_variable('current_loss')
            .save_to_variable('loss_history', 'current_loss', mode='a')
            .next_batch(BATCH_SIZE, shuffle=True, n_epochs=1, drop_last=True, prefetch=0))

test_pp = (mnist.test.p
           .import_model('VGG', train_pp)
           .init_variable('all_predictions', init_on_each_run=list)
           .predict_model('VGG',
                          fetches='predicted_labels',
                          feed_dict={'images': 'images',
                                     'labels': 'labels',
                                     'training': False},
                          append_to='all_predictions')
           .run(BATCH_SIZE, shuffle=True, n_epochs=1, drop_last=False)
)

print("Predictions")
for pred in test_pp.get_variable("all_predictions"):
    print(pred.shape)