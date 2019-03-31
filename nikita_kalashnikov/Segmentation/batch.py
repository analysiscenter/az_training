""" Batch class for MNIST segmentation task.
"""
import PIL
import numpy as np
from batchflow import ImagesBatch, action, inbatch_parallel
from batchflow.batch_image import transform_actions

@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
class MyBatch(ImagesBatch):
    """ Batch class for segmentation task.
    """

    components = 'images', 'masks'
    @action
    @inbatch_parallel(init='indices', post='post_fn')
    def mask(self, ind):
        """ Get the mask out of MNIST image. Mask is not default square
        28x28 but restricted to the digit boundaries.

        Patameters:
        ----------
        ind: int
            Index of the image in the dataset.
        Returns:
        -------
        mask: np.array(dtype='uint8)
            Mask of the digit.
        """
        i = self.get_pos(None, None, ind)
        image = np.array(self.images[i])
        x_min, x_max = np.nonzero(image)[1].min(), np.nonzero(image)[1].max()
        y_min, y_max = np.nonzero(image)[0].min(), np.nonzero(image)[0].max()
        mask = np.zeros_like(image)
        mask[y_min:y_max+1, x_min:x_max+1] = 1
        return mask.astype('uint8')

    def post_fn(self, list_of_res):
        """ Post assmble function for inbatch_parallel decorator.
        """
        setattr(self, 'masks', list_of_res)
        return self

    @action
    @inbatch_parallel(init='indices')
    def custom_rotate(self, ind, angle=0):
        """ Rotate mask and image by the given angle value.

        Parameters:
        ----------
        ind: int
            Index of the element in dataset
        angle: int
            Angle to rotate images
        """
        i = self.get_pos(None, None, ind)
        self.images[i] = self.images[i].rotate(angle=angle)
        mask = PIL.Image.fromarray(self.masks[i])
        self.masks[i] = mask.rotate(angle=angle)

    @action
    @inbatch_parallel(init='indices')
    def background_and_mask(self, ind, bg_shape=(128, 128)):
        """ Place the image and the mask on the black background.

        Parameters:
        ind: int
            Index of the object in the dataset.
        bg_shape: tuple
            Size of the background.
        """
        i = self.get_pos(None, None, ind)
        image = self.images[i]
        background = PIL.Image.fromarray(np.zeros(bg_shape), mode='RGB')
        shape = image.size
        x, y = self._calc_origin(image_shape=shape, origin='random', background_shape=bg_shape)
        background.paste(image, (x, y))
        self.images[i] = background
        background_2 = PIL.Image.fromarray(np.zeros(bg_shape).astype('uint8'), mode='L')
        mask = self.masks[i]
        background_2.paste(mask, (x, y))
        self.masks[i] = background_2

    @action
    @inbatch_parallel(init='indices', target='for')
    def noise(self, ind, n=10):
        """ Add noise to the current image from other images
        in the batch.

        Parameters:
        ind: int
            Index of the object in the dataset.
        n: int
            Number of crops sampled from other images.
        """
        i = self.get_pos(None, None, ind)
        size = self.images[i].width
        for image in self.images:
            for _ in range(n):
                shape = np.random.randint(3, 7, size=2)
                x_to, y_to = np.random.randint(0, size-max(*shape), size=2)
                crop = self._crop_(image, origin='random', shape=shape)
                self.images[i].paste(crop, (x_to, y_to))

    def _custom_to_array_(self, image, mask):
        """ Convert image and mask images from PIL format to np.array.
        Images are reshaped to the format 'channels-first'

        Parameters:
        image: PIL.Image
            Image from the batch.
        mask: PIL.Image
            Mask of the image.
        """
        rev_shape = tuple(reversed(np.array(image).shape))
        return np.array(image).reshape(rev_shape).astype('float32'), \
               np.array(mask).astype('long')
        