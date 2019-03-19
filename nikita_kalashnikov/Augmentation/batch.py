""""Custom Batch class for images.
"""
import numpy as np
from batchflow import ImagesBatch
from batchflow.batch_image import transform_actions
from batchflow.decorators import action, inbatch_parallel, any_action_failed


@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
class MyBatch(ImagesBatch):
    """ Batch class for augmentation task.
    """
    components = 'images', 'labels'
    @action
    @inbatch_parallel(init='indices')
    def custom_noise(self, ind):
        """ Adding the noise to the image from other images.
        All manipulations performed inplace.
        """
        index = self.get_pos(None, None, ind)
        for image_from in self.images:
            n = 30 // self.size
            for _ in range(n):
                shape = np.random.randint(3, 7, size=2)
                x_from, x_to = np.random.randint(low=0, size=2,
                                                 high=image_from.width-shape[0])
                y_from, y_to = np.random.randint(low=0, size=2,
                                                 high=image_from.height-shape[1])
                crop = image_from.crop((x_from, y_from,
                                        x_from + shape[0], y_from + shape[1]))
                self.images[index].paste(crop, box=(x_to, y_to))

    @action
    def preprocess_labels(self):
        """Cast labels to type int64.
        """
        self.labels = np.array(self.labels, dtype='int64')
        return self

    def _preprocess_images_(self, image):
        """ Reshape images and cast arrays to type float32.
        """
        return np.array(image).reshape(3, 66, 66).astype('float32')

    @action
    @inbatch_parallel(init='images', post='post_fn')
    def to_rgb(self, image):
        """ Convert image to RGB format.
        """
        return image.convert('RGB')

    def post_fn(self, list_of_res):
        """ Batch assemble fucntion.
        """
        if any_action_failed(list_of_res):
            print('failed')
        else:
            self.images = np.array(list_of_res, dtype=object)
        return self
