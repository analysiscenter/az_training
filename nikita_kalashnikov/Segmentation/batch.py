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

    components = 'images', 'masks', 'ang'
    @action
    @inbatch_parallel(init='indices', post='post_fn')
    def add_mask(self, ind, src, **kwargs):
        """ Get the mask out of MNIST image. Mask is not default square
        28x28 but restricted to the digit boundaries. Returns PIL.Image.

        Patameters
        ----------
        ind: int
            Index of the image in the dataset.

        Returns
        -------
        mask: PIL.Image
            Mask of the digit.
        """
        i = self.get_pos(None, None, ind)
        image = np.array(getattr(self, src)[i])
        boundary = np.nonzero(image)
        x_min, x_max = boundary[1].min(), boundary[1].max()
        y_min, y_max = boundary[0].min(), boundary[0].max()
        mask = np.zeros_like(image)
        mask[y_min:y_max+1, x_min:x_max+1] = 1
        return PIL.Image.fromarray(mask.astype('uint8'))

    def post_fn(self, list_of_res, dst, **kwargs):
        """ Post assmble function for inbatch_parallel decorator.
        """
        setattr(self, dst, list_of_res)
        return self

    @action
    @inbatch_parallel(init='indices')
    def put_on_background(self, ind, bg_shape=(128, 128), shape=(28, 28), src='images', dst='images'):
        """ Place the image and the mask on the black background.
        Input and output are both PIL.Image.

        Parameters
        ----------
        ind: int
            Index of the object in the dataset.
        src: array-like
            The source to get data from.
        bg_shape: tuple
            Size of the background.
        shape: tuple
            Size of the components object.
        """
        if isinstance(src, str):
            src = [src]
            dst = [dst]
        i = self.get_pos(None, None, ind)
        x, y = self._calc_origin(image_shape=shape, origin='random', background_shape=bg_shape)
        for comp_from, comp_to in zip(src, dst):
            mode = 'RGB' if comp_from == 'images' else 'L'
            obj = getattr(self, comp_from)[i]
            background = PIL.Image.fromarray(np.zeros(bg_shape), mode=mode)
            background.paste(obj, (x, y))
            getattr(self, comp_to)[i] = background
        return self

    @action
    @inbatch_parallel(init='indices', target='for')
    def add_noise(self, ind, n=30, src='images', dst='images'):
        """ Add noise to the current image from other images
        in the batch. Input is PIL.Image.

        Parameters
        ----------
        ind: int
            Index of the object in the dataset.
        n: int
            Number of crops sampled from other images.
        src: array-like
            The source to get data from.
        """
        i = self.get_pos(None, None, ind)
        if isinstance(src, str):
            src = [src]
            dst = [dst]
        for comp_from, comp_to in zip(src, dst):
            size = getattr(self, comp_from)[i].size[0]
            for _ in range(n):
                index_from = np.random.choice(self.__len__())
                image_from = getattr(self, comp_from)[index_from]
                shape = np.random.randint(3, 7, size=2)
                x_to, y_to = np.random.randint(0, size-max(*shape), size=2)
                crop = self._crop_(image_from, origin='random', shape=shape)
                getattr(self, comp_to)[i].paste(crop, (x_to, y_to))

    def _make_channels_first_(self, image, mask):
        """ Convert image and mask images from PIL.Image format to np.array.
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
        