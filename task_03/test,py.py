import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append('..')

from dataset import Dataset, DatasetIndex
from linknet import LinkNetBatch

MNIST_SIZE = 65000
BATCH_SIZE = 512
MAX_ITER = 100

ind = DatasetIndex(np.arange(MNIST_SIZE))
mnistset = Dataset(ind, batch_class=LinkNetBatch)
mnistset.cv_split([0.9, 0.1])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

images = []
masks = []

ppl = mnistset.train.pipeline()\
        .load_images()\
        .random_location() \
        .create_mask()\
        .add_noise()\
        .get_images(images, masks)\

ppl.next_batch(10)

plt.imshow(masks[0][0])
plt.show()

plt.imshow(images[0][0])
plt.show()
