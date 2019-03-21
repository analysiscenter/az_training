""""Custom Batch class for images.
"""
import numpy as np
from batchflow import ImagesBatch
from batchflow.batch_image import transform_actions
from batchflow.decorators import action, inbatch_parallel


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
        """Cast labels to type int64
        """
        setattr(self, 'labels', np.array(self.labels, dtype=int))
        return self

    def _preprocess_images_(self, image):
        """ Reshape images and cast arrays to type float32.

        Parameters
        ----------
        image : PIL.Image
            Image in the batch.

        Returns
        -------
        image_reshape : np.array
            Reshaped image with channels dimension first.
        """
        image_reshape = np.array(image).reshape(3, 64, 64).astype('float32')
        return image_reshape

    def _to_rgb_(self, image):
        """ Convert image to RGB format.

        Parameters
        ----------
        image : PIL.Image
            Image in grayskale format.

        Returns
        -------
        image_converted : PIL.Image
            Image in RGB format.
        """
        image_converted = image.convert('RGB')
        return image_converted
