import sys

import time
import numpy as np
sys.path.append("..//..")
from dataset import ImagesBatch, action, inbatch_parallel

class TimeBatch(ImagesBatch):

    components = 'images', 'labels', 'noise'

    @action
    def update_start_time(self):
    	self.pipeline.update_variable("start_time", time.clock())
    	return self

    @action
    def update_time_history(self):
    	current_interval = time.clock() - self.pipeline.get_variable("start_time")
    	time_history = self.pipeline.get_variable("time_history")
    	time_history.append(current_interval)
    	self.pipeline.update_variable("time_history", time_history) 
    	return self

    @action
    def normalize_images(self):
        """Normalize pixel values to (0, 1)."""
        self.images = self.images / 255
        return self

    def uniform(self, image_size, fragment_size):
        """Uniform distribution of fragmnents on image."""
        return np.random.randint(0, image_size-fragment_size, 2)

    def normal(self, image_size, fragment_size):
        """Normal distribution of fragmnents on image."""
        return list([int(x) for x in np.random.normal((image_size-fragment_size)/2,
                                                      (image_size-fragment_size)/4, 2)])

    @action
    @inbatch_parallel(init='images', post='assemble', components='noise')
    def create_noise(self, image, *args):
        """Create noise at MNIST image."""
        # print('1')
        image_size = self.images.shape[1]
        if args[0] == 'random_noise':
            noise = args[1] * np.random.random((image_size, image_size, 1)) * image.max()
        elif args[0] == 'mnist_noise':
            level, n_fragments, size, distr = args[1:]
            ind_for_noise = np.random.choice(len(self.images), n_fragments)
            images = [self.images[i] for i in ind_for_noise]
            images_for_noise = images
            try:
                fragments = self.create_fragments(images_for_noise, size)
            except Exception as e:
                print('create_fragments failed')
                raise ValueError
            try:
                noise = self.arrange_fragments(image_size, fragments, distr, level)
            except Exception as e:
                print('arrange_fragments failed')
                raise ValueError
        else:
            noise = np.zeros_like(image)
        return noise

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def add_noise(self, ind):
        if self.images.shape[-1] != 1:
            return np.expand_dims(np.max([self.get(ind, 'images'), self.get(ind, 'noise')], axis=0), axis=-1)
        else:
        	return np.max([self.get(ind, 'images'), np.expand_dims(self.get(ind, 'noise'), axis=3)], axis=0)

    def create_fragments(self, images, size):
        """Cut fragment from each."""
        fragments = []
        for image in images:
            image = np.squeeze(image)
            x = np.random.randint(0, image.shape[0] - size)
            y = np.random.randint(0, image.shape[1] - size)
            fragment = image[x : x + size, y : y + size]
            fragments.append(fragment)
        return fragments

    def arrange_fragments(self, image_size, fragments, distr, level):
        """Put fragments on image."""
        image = np.zeros((image_size, image_size))
        for fragment in fragments:
            size = fragment.shape[0]
            x_fragment, y_fragment = getattr(self, distr)(image_size, size)
            image_to_change = image[x_fragment : x_fragment + size, y_fragment : y_fragment + size]
            height_to_change, width_to_change = image_to_change.shape
            image_to_change = np.max([level * fragment[: height_to_change, : width_to_change], image_to_change], axis=0)
            image[x_fragment : x_fragment + size, y_fragment : y_fragment + size] = image_to_change
        return image

