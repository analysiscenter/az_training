"""
LinkNet implementation as Batch class
"""
import pickle
import numpy as np

from dataset.dataset import Batch, action, inbatch_parallel, any_action_failed


def uniform(image_size, fragment_size):
    """Uniform distribution of fragmnents on image."""
    return np.random.randint(0, image_size-fragment_size, 2)


def normal(image_size, fragment_size):
    """Normal distribution of fragmnents on image."""
    return list([int(x) for x in np.random.normal((image_size-fragment_size)/2,
                                                  (image_size-fragment_size)/4, 2)])


def crop_images(images, coordinates):
    """Crop real 28x28 MNIST from large image."""
    images_for_noise = []
    for image, coord in zip(images, coordinates):
        images_for_noise.append(image[coord[0]:coord[0] + 28, coord[1]:coord[1] + 28])
    return images_for_noise


def create_fragments(images, size):
    """Cut fragment from each."""
    fragments = []
    for image in images:
        x, y = np.random.randint(0, 28-size, 2)
        fragment = image[x:x+size, y:y+size]
        fragments.append(fragment)
    return fragments


def arrange_fragments(image_size, fragments, distr, level):
    """Put fragments on image."""
    image = np.zeros((image_size, image_size))
    for fragment in fragments:
        size = fragment.shape[0]
        x_fragment, y_fragment = globals()[distr](image_size, size)
        image_to_change = image[x_fragment:x_fragment+size, y_fragment:y_fragment+size]
        height_to_change, width_to_change = image_to_change.shape
        image_to_change = np.max([level*fragment[:height_to_change, :width_to_change], image_to_change], axis=0)
        image[x_fragment:x_fragment+size, y_fragment:y_fragment+size] = image_to_change
    return image


class NoisedMnist(Batch):
    """Batch class for LinkNet."""

    def __init__(self, index, *args, **kwargs):
        """Init function."""
        super().__init__(index, *args, **kwargs)
        self.images = None
        self.masks = None
        self.coordinates = None
        self.noise = None
        self.labels = None

    @property
    def components(self):
        """Define components."""
        return 'images', 'masks', 'coordinates', 'noise', 'labels'

    @action
    def load_images(self):
        """Load MNIST images from file."""
        with open('../mnist/mnist_pics.pkl', 'rb') as file:
            self.images = pickle.load(file)[self.indices].reshape(-1, 28, 28)
        with open('../mnist/mnist_labels.pkl', 'rb') as file:
            self.labels = pickle.load(file)[self.indices]
        return self

    def init_func(self, *args, **kwargs): # pylint: disable=unused-argument
        """Create tasks."""
        return [i for i in range(self.images.shape[0])]

    @action
    @inbatch_parallel(init='init_func', post='post_func_image', target='threads')
    def random_location(self, ind, *args):
        """Put MNIST image in random location"""
        image_size = args[0]
        pure_mnist = self.images[ind]
        new_x, new_y = np.random.randint(0, image_size-28, 2)
        large_mnist = np.zeros((image_size, image_size))
        large_mnist[new_x:new_x+28, new_y:new_y+28] = pure_mnist
        return large_mnist, new_x, new_y

    def post_func_image(self, list_of_res, *args, **kwargs): # pylint: disable=unused-argument
        """Concat outputs from random_location.

        Parameters
        ----------
        list_of_res : list of tuples of np.arrays and two ints
        """

        if any_action_failed(list_of_res):
            print(list_of_res)
            raise Exception("Something bad happened")
        else:
            images, new_x, new_y = list(zip(*list_of_res))
            self.images = np.stack(images)
            self.coordinates = list(zip(new_x, new_y))
            return self

    @action
    @inbatch_parallel(init='init_func', post='post_func_mask', target='threads')
    def create_mask(self, ind):
        """Get mask of MNIST image"""
        # new_x, new_y = self.coordinates[ind]
        # image_size = self.images.shape[1]
        # mask = np.zeros((image_size, image_size))
        # mask[new_x:new_x+28, new_y:new_y+28] += 1
        mask = np.array((self.images[ind] > 0.1), dtype=np.int32)
        mask = np.stack([1-mask, mask], axis=2)
        return mask

    def post_func_mask(self, list_of_res, *args, **kwargs): # pylint: disable=unused-argument
        """Concat outputs from random_location.

        Parameters
        ----------
        list_of_res : list of tuples of np.arrays and two ints
        """

        if any_action_failed(list_of_res):
            print(list_of_res)
            raise Exception("Something bad happened")
        else:
            self.masks = np.stack(list_of_res)
            return self

    @action
    @inbatch_parallel(init='init_func', post='post_func_created_noise', target='threads')
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
            images_for_noise = crop_images(images, coordinates)
            fragments = create_fragments(images_for_noise, size)
            noise = arrange_fragments(image_size, fragments, distr, level)
        else:
            noise = np.zeros_like(self.images[ind])
        return noise

    def post_func_created_noise(self, list_of_res, *args, **kwargs): # pylint: disable=unused-argument
        """Concat outputs from add_noise.
        """
        if any_action_failed(list_of_res):
            print(list_of_res)
            raise Exception("Something bad happened")
        else:
            self.noise = np.stack(list_of_res)
            return self

    @action
    @inbatch_parallel(init='init_func', post='post_func_noise', target='threads')
    def add_noise(self, ind):
        """Add noise at MNIST image.
        """
        return np.expand_dims(np.max([self.images[ind], self.noise[ind]], axis=0), axis=-1)

    def post_func_noise(self, list_of_res, *args, **kwargs): # pylint: disable=unused-argument
        """Concat outputs from add_noise.
        """
        if any_action_failed(list_of_res):
            print(list_of_res)
            raise Exception("Something bad happened")
        else:
            self.images = np.stack(list_of_res)
            return self

    @action
    def get_images(self, images):
        """Get images from batch.

        Parameters
        ----------
        images : list of np.array
        """

        images.append(self.images)
        return self

    @action
    def get_masks(self, masks):
        """Get images from batch.

        Parameters
        ----------
        masks : list of np.array
        """

        masks.append(self.masks)
        return self

    @action
    def get_noise(self, noise):
        """Get noise from batch.

        Parameters
        ----------
        masks : list of np.array
        """

        noise.append(self.noise)
        return self
