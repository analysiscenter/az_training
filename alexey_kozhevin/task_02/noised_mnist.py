#pylint:disable=attribute-defined-outside-init

"""
LinkNet implementation as Batch class
"""
import numpy as np

from dataset.dataset import ImagesBatch, action, inbatch_parallel, any_action_failed


class NoisedMnist(ImagesBatch):
    """Batch class for LinkNet."""

    components = 'images', 'labels', 'masks', 'coordinates', 'noise'

    @action
    @inbatch_parallel(init='indices', post='assemble', target='threads')
    def normalize_images(self, ind):
        """Normalize pixel values to (0, 1)"""
        return self.get(ind, 'images') / 255

    @action
    @inbatch_parallel(init='indices', post='post_func_image', target='threads')
    def random_location(self, ind, *args):
        """Put MNIST image in random location"""
        image_size = args[0]
        pure_mnist = self.get(ind, 'images').reshape(28, 28)
        new_x, new_y = np.random.randint(0, image_size-28, 2)
        large_mnist = np.zeros((image_size, image_size))
        large_mnist[new_x:new_x+28, new_y:new_y+28] = pure_mnist
        return large_mnist, new_x, new_y

    def post_func_image(self, list_of_res, *args, **kwargs):
        """Concat outputs from random_location.

        Parameters
        ----------
        list_of_res : list of tuples of np.arrays and two ints
        """
        _ = args, kwargs
        if any_action_failed(list_of_res):
            raise Exception("Something bad happened")
        else:
            images, new_x, new_y = list(zip(*list_of_res))
            self.images = np.stack(images)
            self.coordinates = list(zip(new_x, new_y))
            return self

    @action
    @inbatch_parallel(init='indices', post='assemble', target='threads', component='masks')
    def create_mask(self, ind, **kwargs):
        """Get mask of MNIST image"""
        mask = np.array((self.get(ind, 'images') > 0.1), dtype=np.int32)
        mask = np.stack([1-mask, mask], axis=2)
        return mask

    @action
    @inbatch_parallel(init='indices', post='assemble', target='threads', component='noise')
    def create_noise(self, ind, *args):
        """Create noise at MNIST image"""
        image_size = self.images.shape[1]
        if args[0] == 'random_noise':
            noise = args[1] * np.random.random((image_size, image_size))
        elif args[0] == 'mnist_noise':
            level, n_fragments, size, distr = args[1:]

            ind_for_noise = np.random.choice(len(self.images), n_fragments)
            images = [self.images[i] for i in ind_for_noise]
            coordinates = [self.coordinates[i] for i in ind_for_noise]
            images_for_noise = self.crop_images(images, coordinates)
            fragments = self.create_fragments(images_for_noise, size)
            noise = self.arrange_fragments(image_size, fragments, distr, level)
        else:
            noise = np.zeros_like(self.get(ind, 'images'))
        return noise

    @action
    @inbatch_parallel(init='indices', post='assemble', target='threads')
    def add_noise(self, ind):
        """Add noise at MNIST image."""
        return np.expand_dims(np.max([self.get(ind, 'images'), self.get(ind, 'noise')], axis=0), axis=-1)

    def uniform(self, image_size, fragment_size):
        """Uniform distribution of fragmnents on image."""
        return np.random.randint(0, image_size-fragment_size, 2)


    def normal(self, image_size, fragment_size):
        """Normal distribution of fragmnents on image."""
        return list([int(x) for x in np.random.normal((image_size-fragment_size)/2,
                                                      (image_size-fragment_size)/4, 2)])

    def crop_images(self, images, coordinates):
        """Crop real 28x28 MNIST from large image."""
        images_for_noise = []
        for image, coord in zip(images, coordinates):
            images_for_noise.append(image[coord[0]:coord[0] + 28, coord[1]:coord[1] + 28])
        return images_for_noise

    def create_fragments(self, images, size):
        """Cut fragment from each."""
        fragments = []
        for image in images:
            x, y = np.random.randint(0, 28-size, 2)
            fragment = image[x:x+size, y:y+size]
            fragments.append(fragment)
        return fragments

    def arrange_fragments(self, image_size, fragments, distr, level):
        """Put fragments on image."""
        image = np.zeros((image_size, image_size))
        for fragment in fragments:
            size = fragment.shape[0]
            x_fragment, y_fragment = getattr(self, distr)(image_size, size)
            image_to_change = image[x_fragment:x_fragment+size, y_fragment:y_fragment+size]
            height_to_change, width_to_change = image_to_change.shape
            image_to_change = np.max([level*fragment[:height_to_change, :width_to_change], image_to_change], axis=0)
            image[x_fragment:x_fragment+size, y_fragment:y_fragment+size] = image_to_change
        return image