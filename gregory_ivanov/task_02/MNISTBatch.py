"""Custom batch class for MNIST images
"""
import numpy as np
from dataset import action, inbatch_parallel, ImagesBatch #pylint: disable=import-error

class AugmentedMNISTBatch(ImagesBatch):
    """ Batch class for MNIST images with colorization and backgrounds
    """

    # @action
    # @inbatch_parallel(init='indices', post='assemble', target='threads')
    # def colorize(self, idx):
    #     """randomly colorize an image with one channel
    #     Args:
    #         idx: index in self.images of an image to be colorized
    #     Returns
    #     -------
    #         colorized image
    #     """
    #     return self.images[idx] * np.random.random(size=3)
    @action
    @inbatch_parallel(init='indices', post='assemble', target='threads')
    def truncate(self, ix, components='images', threshold = 100, min_intencity=0, max_intencity=255):
        """
        """
        image = self.get(ix, components)
        image[image < threshold] = min_intencity
        image[image >= threshold] = max_intencity
        return image.copy()


    @action
    @inbatch_parallel(init='indices', post='assemble', target='threads')
    def reshape(self, ix, components='images'):
        """randomly colorize an image with one channel
        Args:
            idx: index in self.images of an image to be colorized
        Returns
        -------
            colorized image
        """
        # print(ix)
        # # print(self.images.shape)
        image = self.get(ix, components)
        return (image * np.ones(3)).astype(np.uint8).copy()


    @action
    @inbatch_parallel(init='indices', post='assemble', target='threads')
    def colorize(self, ix, components='images'):
        """randomly colorize an image with one channel
        Args:
            idx: index in self.images of an image to be colorized
        Returns
        -------
            colorized image
        """
        # print(ix)
        # # print(self.images.shape)
        image = self.get(ix, components)
        return (image * np.random.random(size=3)).astype(np.uint8).copy()


    def generate_gradient(self, shape):
        x = np.ones(list(shape) + [3])
        x[:, :, 0:3] = np.random.uniform(0, 1, (3,))

        y = np.ones(list(shape) + [3])
        y[:,:,0:3] = np.random.uniform(0, 1, (3,))

        c = np.linspace(0, 1, shape[0])[:, None, None]

        gradient = x + (y - x) * c

        # gradient = self._resize_image(gradient, np.asarray(shape)*1.5)
        # gradient = self._rotate_image(gradient, np.random.uniform(0, 180), preserve_shape=True)
        # gradient = self._crop_image(gradient, 'center', shape)

        return (255*gradient).astype(np.uint8)

    @action
    @inbatch_parallel(init='indices', post='assemble', target='threads')
    def put_on_image(self, ix, components = 'images', background_images_shape = (128,128), **kwargs):
        """put an image on a randomly selected background
        Args:
            idx: index in self.images of an image to bu put
            background_images: np.array of background images
        Returns
        -------
            randomly selected background picture with a mnist image in a random place
        """

        back = self.generate_gradient(background_images_shape)
        # back = self._salt_and_pepper(back, **kwargs)

        # back shape = (n, m, 3)
        # back = np.copy(np.random.choice(background_images))
        n, m = back.shape[:2]

        # image shape = (k, k, 3)
        image = self.get(ix, components).copy()
        k = image.shape[0]

        i_left_upper, j_left_uuper = np.random.randint(n-k+1), np.random.randint(m-k+1)

        non_zero = image > 0
        back[i_left_upper : i_left_upper+k,
             j_left_uuper : j_left_uuper+k][non_zero] = image[non_zero]

        return back
