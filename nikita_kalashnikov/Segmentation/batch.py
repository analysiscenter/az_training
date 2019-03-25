import PIL
import numpy as np
from batchflow import ImagesBatch
from batchflow.decorators import action, inbatch_parallel


class MyBatch(ImagesBatch):
    """ Batch class for segmentation task.
    """
    components = 'images', 'labels', 'masks'

    @action
    @inbatch_parallel(init='indices')
    def to_rgb(self, ind):
        index = self.get_pos(None, None, ind)
        self.images[index] = self.images[index].convert('RGB')
    
    @action
    @inbatch_parallel(init='indices')
    def digit_on_layout(self, ind):
        layout = PIL.Image.fromarray(np.zeros((128, 128)), mode='RGB')
        index = self.get_pos(None, None, ind)
        image = self.images[index]
        pos_x, pos_y = np.random.randint(0, 100, size=2)
        layout.paste(image, box=(pos_x,pos_y))
        self.images[index] = layout
    
    @action
    @inbatch_parallel(init='indices')
    def noise(self, ind, n=10):
        index = self.get_pos(None, None, ind)
        image_to = self.images[index]
        size = image_to.width
        for image in self.images:
            for _ in range(n):
                shape = np.random.randint(3, 7, size=2)
                x_from, x_to = np.random.randint(low=0, high=size-shape[0],
                                                 size=2)
                y_from, y_to = np.random.randint(low=0, high=size-shape[1],
                                                 size=2)
                crop = image.crop((x_from, y_from,
                                        x_from + shape[0], y_from + shape[1]))
                image_to.paste(crop, (x_to,y_to))
                self.images[index] = image_to

    @action
    @inbatch_parallel(init='images', components='masks')
    def mask(self, image):
        masks = np.array(image) > 0
        return masks
                                    
