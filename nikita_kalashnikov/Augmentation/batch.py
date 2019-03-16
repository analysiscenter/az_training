import numpy as np
from batchflow import ImagesBatch
from batchflow.decorators import action, inbatch_parallel

class MyBatch(ImagesBatch):
    components = 'images', 'labels'

    def _init_fn(self):
        return [[i, j] for i, j in zip(self.images, np.random.permutation(self.images))]
    
    @action
    @inbatch_parallel(init='_init_fn')
    def custom_noise(self, image_to, image_from):
        n = np.random.randint(7, 13)
        for i in range(n):
            shape = np.random.randint(3, 7)
            x_from, y_from = np.random.randint(low=0, high=66-shape, size=2)
            x_to, y_to = np.random.randint(low=0, high=66-shape, size=2) 
            crop = image_from.crop((x_from, y_from,
                                     x_from + shape, y_from + shape))
            image_to.paste(crop, box=(x_to, y_to))       
        return image_to