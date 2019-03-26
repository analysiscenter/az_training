import PIL
import numpy as np
from batchflow import ImagesBatch
from batchflow import action, inbatch_parallel
from batchflow.batch_image import transform_actions

@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
@transform_actions(prefix='_', suffix='_all', wrapper='apply_transform_all')
class MyBatch(ImagesBatch):
    """ Batch class for segmentation task.
    """
    components = 'images', 'masks'
        
    @action
    @inbatch_parallel(init='indices')
    def noise(self, ind, n=10):
        index = self.get_pos(None, None, ind)
        image_to = self.images[index]
        size = image_to.width
        for image in self.images:
            for _ in range(n):
                shape = np.random.randint(3, 7, size=2)
                x_to, y_to = np.random.randint(0, size-max(*shape), size=2)
                crop = self._crop_(image, origin='random', shape=shape)
                self.images[index].paste(crop, (x_to,y_to))
        


    @action
    @inbatch_parallel(init='images',post='post_fn')
    def mask(self, image):
        return  (np.array(image) > 0)[:,:,0].astype(int)
    
    def post_fn(self, list_of_res):
        self.mask = np.array(list_of_res)
        return self


