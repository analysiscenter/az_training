"""
Generate images with MNIST in random positions
"""
import sys
import pickle
import numpy as np
import time
from scipy.special import expit
import scipy.ndimage
import itertools
from skimage.transform import resize 

sys.path.append('../az_training')

from dataset import action, inbatch_parallel, any_action_failed
from dataset import ImagesBatch

_IOU_LOW = 0.3
_IOU_HIGH = 0.7

class FasterRCNNBatch(ImagesBatch):
    """Batch class for Faster RCNN."""

    components = ('images', 'labels', 'bboxes',
                  'anchors', 'reg', 'clsf', 'anchor_labels', 'anchor_batch')
    
    @action
    def create_anchors(self, image_shape, scales=(4, 8, 16), ratio=2):
        """ Create anchors for image_shape depending on output_map_shape. """
        map_shape = self.pipeline.config['rpn']['output_map_shape']
        ratios = ((np.sqrt(ratio), 1/np.sqrt(ratio)),
                  (1, 1),
                  (1/np.sqrt(ratio), np.sqrt(ratio)))

        self.anchors = []
        for scale in scales:
            for ratio in ratios:
                ih, iw = image_shape
                fh, fw = map_shape
                n = fh * fw

                j = np.array(list(range(fh)))
                j = np.expand_dims(j, 1)
                j = np.tile(j, (1, fw))
                j = j.reshape((-1))

                i = np.array(list(range(fw)))
                i = np.expand_dims(i, 0)
                i = np.tile(i, (fh, 1))
                i = i.reshape((-1))

                s = np.ones((n)) * scale
                r0 = np.ones((n)) * ratio[0]
                r1 = np.ones((n)) * ratio[1]

                h = s * r0
                w = s * r1
                y = (j + 0.5) * ih / fh - h * 0.5
                x = (i + 0.5) * iw / fw - w * 0.5

                y, x = [np.maximum(vector, np.zeros((n))) for vector in [y, x]]
                h = np.minimum(h, ih-y)
                w = np.minimum(w, iw-x)

                anchors = [np.expand_dims(vector, 1) for vector in [y, x, h, w]]
                anchors = np.concatenate(anchors, axis=1)
                self.anchors.append(np.array(anchors, np.int32))

        self.anchors = np.array(self.anchors).transpose(1, 0, 2).reshape(-1, 4)
        return self

    @action
    @inbatch_parallel(init='indices', post='assemble', components=('reg', 'clsf', 'anchor_labels'))
    def create_rpn_inputs(self, ind):
        """ Create reg and clsf targets of RPN. """
        anchors = self.anchors
        bboxes = self.get(ind, 'bboxes')
        labels = self.get(ind, 'labels')

        n = anchors.shape[0]
        k = bboxes.shape[0]

        # Compute the IoUs of the anchors and ground truth boxes
        tiled_anchors = np.tile(np.expand_dims(anchors, 1), (1, k, 1))
        tiled_bboxes = np.tile(np.expand_dims(bboxes, 0), (n, 1, 1))

        tiled_anchors = tiled_anchors.reshape((-1, 4))
        tiled_bboxes = tiled_bboxes.reshape((-1, 4))

        ious = self.iou_bbox(tiled_anchors, tiled_bboxes)[0]
        ious = ious.reshape(n, k)

        # Label each anchor based on its max IoU
        max_ious = np.max(ious, axis=1)
        best_bbox_for_anchor = np.argmax(ious, axis=1)

        reg = bboxes[best_bbox_for_anchor]
        proposal_bboxes_labels = labels[best_bbox_for_anchor].reshape(-1)

        # anchor has at least one gt-bbox with IoU >_IOU_HIGH
        clsf = np.array(max_ious > _IOU_HIGH, dtype=np.int32)

        # anchor intersects with at least one bbox 
        best_anchor_for_bbox = np.argmax(ious, axis=0)
        clsf[best_anchor_for_bbox] = 1

        # max IoU for anchor < _IOU_LOW
        clsf[np.logical_and(max_ious < _IOU_LOW, clsf == 0)] = -1      
        return reg, clsf, proposal_bboxes_labels

    @action
    @inbatch_parallel(init='indices', post='assemble', components='anchor_batch')
    def batch_anchors(self, ind, batch_size=64):
        """ Create batch indices for anchors. """
        clsf = self.get(ind, 'clsf')
        batch_size = min(batch_size, len(clsf))
        positive = clsf == 1
        negative = clsf == -1
        if sum(positive) + sum(negative) < batch_size:
            batch_size = sum(positive) + sum(negative)
        if sum(positive) < batch_size / 2:
            positive_batch_size = sum(positive)
            negative_batch_size = batch_size - sum(positive)
        elif sum(negative) < batch_size / 2:
            positive_batch_size = batch_size - sum(negative)
            negative_batch_size = sum(negative)
        else:
            positive_batch_size = batch_size // 2
            negative_batch_size = batch_size // 2

        p = positive / sum(positive)
        positive_batch = np.random.choice(len(clsf), size=positive_batch_size, replace=False, p=p)
        p = negative / sum(negative)
        negative_batch = np.random.choice(len(clsf), size=negative_batch_size, replace=False, p=p)
        anchor_batch = np.array([False]*len(clsf))
        anchor_batch[positive_batch] = True
        anchor_batch[negative_batch] = True
        return anchor_batch

    @action
    def transform_clsf(self):
        """ Convert three classes of anchors to two classes of positive and others. """
        self.clsf = np.array(self.clsf == 1, dtype=np.int32)
        return self

    @classmethod
    def iou_bbox(cls, bboxes1, bboxes2):
        """ Compute the IoUs between bounding boxes. """
        bboxes1 = np.array(bboxes1, np.float32)
        bboxes2 = np.array(bboxes2, np.float32)

        intersection_min_y = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
        intersection_max_y = np.minimum(bboxes1[:, 0] + bboxes1[:, 2] - 1, bboxes2[:, 0] + bboxes2[:, 2] - 1)
        intersection_height = np.maximum(intersection_max_y - intersection_min_y + 1, np.zeros_like(bboxes1[:, 0]))

        intersection_min_x = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
        intersection_max_x = np.minimum(bboxes1[:, 1] + bboxes1[:, 3] - 1, bboxes2[:, 1] + bboxes2[:, 3] - 1)
        intersection_width = np.maximum(intersection_max_x - intersection_min_x + 1, np.zeros_like(bboxes1[:, 1]))

        area_intersection = intersection_height * intersection_width
        area_first = bboxes1[:, 2] * bboxes1[:, 3]
        area_second = bboxes2[:, 2] * bboxes2[:, 3]
        area_union = area_first + area_second - area_intersection

        iou = area_intersection * 1.0 / area_union
        iof = area_intersection * 1.0 / area_first
        ios = area_intersection * 1.0 / area_second

        return iou, iof, ios


class DetectionMnist(FasterRCNNBatch):
    """Batch class for multiple MNIST."""

    components = ('images', 'labels', 'bboxes',
                  'anchors', 'reg', 'clsf', 'anchor_labels', 'anchor_batch')

    @action
    @inbatch_parallel(init='indices', post='post_func_multi')
    def generate_images(self, ind, *args, **kwargs):
        """ Create image with 'image_shape' and put MNIST digits in random locations resized to 'resize_to'. """
        image_shape = kwargs.get('image_shape', (64, 64))
        n_digits = kwargs.get('n_digits', (10, 20))
        resize_to = kwargs.get('resize_to', (28, 28))

        factor = 1. * np.asarray(resize_to) / np.asarray(self.images.shape[1:3])
        if isinstance(n_digits, (list, tuple)):
            n_digits = np.random.randint(*n_digits)
        elif isinstance(n_digits, int):
            digits = n_digits
        else:
            raise TypeError('n_digits must be int, tuple or list but {} was given'.format(type(n_digits).__name__))
        digits = np.random.choice(len(self.images), min([n_digits, len(self.images)]))
        large_image = np.zeros(image_shape)
        bboxes = []
        labels = []

        for i in digits:
            image = np.squeeze(self.images[i])
            image = scipy.ndimage.interpolation.zoom(image, factor, order=3)
            new_x = np.random.randint(0, image_shape[0]-image.shape[0])
            new_y = np.random.randint(0, image_shape[1]-image.shape[1])
            old_region = large_image[new_x:new_x+image.shape[0], new_y:new_y+image.shape[1]]
            large_image[new_x:new_x+image.shape[0], new_y:new_y+image.shape[1]] = np.max([image, old_region], axis=0)
            bboxes.append((new_x, new_y, image.shape[0], image.shape[1]))
            labels.append(self.labels[i])
        return large_image, np.array(bboxes), np.array(labels)

    def post_func_multi(self, list_of_res, *args, **kwargs): # pylint: disable=unused-argument
        """Post function for generate_multimnist_images."""
        if any_action_failed(list_of_res):
            print(list_of_res)
            raise Exception("Something bad happened")
        else:
            images, bboxes, labels = list(zip(*list_of_res))
            self.images = np.expand_dims(np.stack(images), axis=-1)
            self.labels = labels
            self.bboxes = bboxes
            return self


