from dataset import Batch, action, inbatch_parallel, ImagesBatch
import numpy as np


class AugmentedMNISTBatch(ImagesBatch):
  @action
  @inbatch_parallel(init='indices', post='assemble', target='threads')
  def colorize(idx):
    """randomly colorize an image with one channel
    Args:
      idx: index in self.images of an image to be colorized
    Returns
    -------
      colorized image
    """
    return self.images[idx] * np.random.random(size = 3)

  @action
  @inbatch_parallel(init='indices', post='assemble', target='threads')
  def put_on_image(self, idx, background_images):
    """put images on a randomly selected backgrounds
    Args:
      background_images: np.array of background images 
    Returns
    -------
      randomly selected background pictures with mnist images in random places
    """
    return self._put_one_on_image(idx, background_images)
  

  def _put_one_on_image(self, idx, background_images):
    """put an image on a randomly selected background
    Args:
      idx: index in self.images of an image to bu put
      background_images: np.array of background images 
    Returns
    -------
      randomly selected background picture with an mnist image in a random place
    """

    # back shape = (n, m, 3)
    back = np.copy(np.random.choice(background_images))
    n, m = back.shape[:2]

    # image shape = (k, k, 3)
    image = self.images[idx]
    k = image.shape[0]
    
    i_left_upper, j_left_uuper = np.random.randint(n - k + 1), np.random.randint(m - k + 1)
    
    non_zero = image > 0
    back[i_left_upper : i_left_upper + k,
         j_left_uuper : j_left_uuper + k][non_zero] = image[non_zero]

    return back