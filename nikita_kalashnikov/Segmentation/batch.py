import PIL
import numpy as np
from batchflow import ImagesBatch
from batchflow import action, inbatch_parallel
from batchflow.batch_image import transform_actions

@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
class MyBatch(ImagesBatch):
    """ Batch class for segmentation task.
    """
    components = 'images', 'masks'
        
    @action
    @inbatch_parallel(init='indices', target='for')
    def noise(self, ind, n=10):
        i = self.get_pos(None, None, ind)
        size = self.images[i].width 
        for image in self.images:
            for _ in range(n):
                shape = np.random.randint(3, 7, size=2)
                x_to, y_to = np.random.randint(0, size-max(*shape), size=2)
                crop = self._crop_(image, origin='random', shape=shape)
                self.images[i].paste(crop, (x_to,y_to))

    @action
    @inbatch_parallel(init='indices', post='post_fn')
    def background_and_mask(self, ind, bg_shape=(128,128)):
        i = self.get_pos(None, None, ind)
        image = self.images[i].convert(mode='RGB')
        background = PIL.Image.fromarray(np.zeros(bg_shape), mode='RGB')
        shape = image.size
        x, y = self._calc_origin(image_shape=shape, origin='random', background_shape=bg_shape)
        background.paste(image, (x,y))
        self.images[i] = background
        mask = np.zeros(bg_shape)
        mask[y:y+shape[0]+1, x:x+shape[1]+1] = 1
        #mask = np.array(background) > 0
        return mask.astype('long')
    
    def post_fn(self, list_of_res, *args):
        self.masks = np.array(list_of_res)
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
        shape = tuple(reversed(np.array(image).shape))
        image_reshape = np.array(image).reshape(shape).astype('float32')
        return image_reshape
                

