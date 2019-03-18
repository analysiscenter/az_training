""""Custom Batch class for images """
import numpy as np
from PIL import Image
from batchflow import ImagesBatch
from batchflow.decorators import action, inbatch_parallel, any_action_failed


class MyBatch(ImagesBatch):
    components = 'images', 'labels'

    def _init_fn(self):
        return [[i, j] for i, j in zip(self.images,
                                       np.random.permutation(self.images))]

    @action
    @inbatch_parallel(init='_init_fn')
    def custom_noise(self, image_to, image_from):
        n = np.random.randint(10, 17)
        for _ in range(n):
            shape = np.random.randint(3, 7, size=2)
            x_from, x_to = np.random.randint(low=0, size=2,
                                             high=image_from.width-shape[0])
            y_from, y_to = np.random.randint(low=0, size=2,
                                             high=image_from.height-shape[1])
            crop = image_from.crop((x_from, y_from,
                                    x_from + shape[0], y_from + shape[1]))
            image_to.paste(crop, box=(x_to, y_to))
        return image_to

    @action
    def labels_to_long(self):
        self.labels = np.array(self.labels, dtype='int64')
        self.images = self.images.reshape(-1, 3, 66, 66).astype('float32')
        return self

    @action
    @inbatch_parallel(init='images', post='post_fn')
    def to_RGB(self, image):
        return image.convert('RGB')

    def post_fn(self, list_of_res):
        if any_action_failed(list_of_res):
            print('failed')
        else:
            self.images = np.array(list_of_res, dtype=object)
        return self
