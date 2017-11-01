"""
Generate images with MNIST in random positions
"""
import sys
import pickle
import numpy as np
import time

sys.path.append('..')

from dataset import Batch, action, inbatch_parallel, any_action_failed

class DetectionMnist(Batch):
    """Batch class for LinkNet."""

    def __init__(self, index, *args, **kwargs):
        """Init function."""
        super().__init__(index, *args, **kwargs)
        self.images = None
        self.labels = None
        self.bboxes = None
        self.labels_batch = None
        self.bboxes_batch = None
        self.anchors = None
        self.reg = None
        self.clsf = None
        self.true_labels = None

    @property
    def components(self):
        """Define components."""
        return ('images', 'labels', 'bboxes', 'labels_batch', 'bboxes_batch', 
               'anchors', 'reg', 'clsf', 'true_labels')

    @action
    def load_images(self):
        """Load MNIST images from file."""
        with open('../mnist/mnist_pics.pkl', 'rb') as file:
            self.images = pickle.load(file)[self.indices].reshape(-1, 28, 28)
        with open('../mnist/mnist_labels.pkl', 'rb') as file:
            self.labels = np.argmax(pickle.load(file)[self.indices], axis=-1)
        return self

    def init_func(self, *args, **kwargs): # pylint: disable=unused-argument
        """Create tasks."""
        return [i for i in range(self.images.shape[0])]
    
    @action
    @inbatch_parallel(init='init_func', post='post_func_multi', target='threads')
    def generate_multi_mnist(self, ind, *args, **kwargs):
        """Put MNIST images in random location"""
        image_shape = kwargs['image_shape']
        max_digits = kwargs['max_dig']
        n_digits = np.random.randint(1, max_digits+1)
        digits = np.random.choice(len(self.images), min([n_digits, len(self.images)]))
        large_image = np.zeros(image_shape)
        bboxes = []
        labels = []
        
        for i in digits:
            image = self.images[i]
            new_x = np.random.randint(0, image_shape[0]-28)
            new_y = np.random.randint(0, image_shape[1]-28)
            large_image[new_x:new_x+28, new_y:new_y+28] = np.max([image, large_image[new_x:new_x+28, new_y:new_y+28]], axis=0)
            bboxes.append((new_x, new_y, 28, 28))
            labels.append(self.labels[i])
        return large_image, np.array(bboxes), np.array(labels)

    def post_func_multi(self, list_of_res, *args, **kwargs): # pylint: disable=unused-argument
        if any_action_failed(list_of_res):
            print(list_of_res)
            raise Exception("Something bad happened")
        else:
            images, bboxes, labels = list(zip(*list_of_res))
            self.images = np.stack(images)
            self.labels = labels
            self.bboxes = bboxes
            return self


    @action
    @inbatch_parallel(init='init_func', post='post_func_bbox_batch', target='threads')
    def create_bbox_batch(self, ind, *args, **kwargs):
        n_bboxes = kwargs['n_bboxes']
        bboxes = self.bboxes[ind]
        labels = self.labels[ind]
        sample = np.random.choice(len(bboxes), min([len(bboxes), n_bboxes]), replace=False)
        if len(sample) < n_bboxes:
            sample = np.concatenate([sample, np.zeros(n_bboxes - len(sample), dtype=np.int32)])
        return bboxes[sample], labels[sample]

    def post_func_bbox_batch(self, list_of_res, *args, **kwargs): # pylint: disable=unused-argument
        if any_action_failed(list_of_res):
            print(list_of_res)
            raise Exception("Something bad happened")
        else:
            bboxes, labels = list(zip(*list_of_res))
            self.labels_batch = labels
            self.bboxes_batch = bboxes
            return self


    @action
    @inbatch_parallel(init='init_func', post='post_func_reg_cls', target='threads')
    def create_reg_cls(self, ind):
        anchors = self.anchors
        bboxes = self.bboxes_batch[ind]
        labels = self.labels_batch[ind]

        n = anchors.shape[0]
        k = bboxes.shape[0]

        # Compute the IoUs of the anchors and ground truth boxes
        tiled_anchors = np.tile(np.expand_dims(anchors, 1), (1, k, 1))
        tiled_bboxes = np.tile(np.expand_dims(bboxes, 0), (n, 1, 1))

        tiled_anchors = tiled_anchors.reshape((-1, 4))
        tiled_bboxes = tiled_bboxes.reshape((-1, 4))

        ious = iou_bbox(tiled_anchors, tiled_bboxes)[0]
        ious = ious.reshape(n, k)

        # Label each anchor based on its max IoU
        max_ious = np.max(ious, axis=1)
    
        best_gt_bbox_ids = np.argmax(ious, axis=1)

        reg = bboxes[best_gt_bbox_ids]
        #reg = param_bbox(reg, self.anchors)
        true_labels = labels[best_gt_bbox_ids].reshape(-1)
        clsf = max_ious > 0.7
        return reg, clsf, true_labels

    def post_func_reg_cls(self, list_of_res, *args, **kwargs): # pylint: disable=unused-argument
        if any_action_failed(list_of_res):
            print(list_of_res)
            raise Exception("Something bad happened")
        else:
            reg, clsf, true_labels = list(zip(*list_of_res))
            self.reg = np.array(reg)
            self.clsf = np.array(clsf)
            self.true_labels = np.array(true_labels)
            print('clsf:', self.clsf.shape)
            return self

    @action
    def param_reg(self):
        """ Parameterize bounding boxes with respect to anchors. Namely, (y,x,h,w)->(ty,tx,th,tw). """
        
        anchors = self.anchors
        param_bb = []
        for bboxes in self.reg:
            tyx = (bboxes[:, :2] - anchors[:, :2]) / anchors[:, 2:]
            thw = np.log(bboxes[:, 2:] / anchors[:, 2:])
            param_bb.append(np.concatenate((tyx, thw), axis=1))
        param_bb = np.array(param_bb)
        self.reg = param_bb
        return self

    @action
    def unparam_bbox(predictions, max_shape=None):
        """ Unparameterize bounding boxes with respect to anchors. Namely, (ty,tx,th,tw)->(y,x,h,w). """
        predictions = np.array(t, np.float32)
        anchors = self.anchors
        unparam_bb = []

        for t in predictions:
            yx = t[:, :2] * anchors[:, 2:] + anchors[:, :2]
            hw = np.exp(t[:, 2:]) * anchors[:, 2:]

            bboxes = np.concatenate((yx, hw), axis=1)

            if max_shape != None:
                bboxes = rectify_bbox(bboxes, max_shape)

            unparam_bb.append(bboxes)
        unparam_bb = np.array(unparam_bb)
        predictions = unparam_bb
        return self


    @action
    def create_anchors(self, img_shape, map_shape, scales=[8, 16, 32], ratio=3):
        ratios = ((np.sqrt(ratio), 1/np.sqrt(ratio)),
                  (1, 1),
                  (1/np.sqrt(ratio), np.sqrt(ratio)))

        self.anchors = []
        for scale in scales:
            for ratio in ratios:
                ih, iw = img_shape
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

                y = np.maximum(y, np.zeros((n)))
                x = np.maximum(x, np.zeros((n)))
                h = np.minimum(h, ih-y)
                w = np.minimum(w, iw-x)

                y = np.expand_dims(y, 1)
                x = np.expand_dims(x, 1)
                h = np.expand_dims(h, 1)
                w = np.expand_dims(w, 1)
                anchors = np.concatenate((y, x, h, w), axis=1)
                self.anchors.append(np.array(anchors, np.int32))

        self.anchors = np.array(self.anchors).transpose(1, 0, 2).reshape(-1, 4)
        return self

def iou_anchor_bbox(anchor, bbox):
    """ Compute the IoUs between bounding boxes. """
    anchor = np.array(anchor, np.float32)
    bbox = np.array(bbox, np.float32)
    
    intersection_min_y = np.maximum(anchor[0], bbox[0])
    intersection_max_y = np.minimum(anchor[0] + anchor[2] - 1, bbox[0] + bbox[2] - 1)
    intersection_height = np.maximum(intersection_max_y - intersection_min_y + 1, np.zeros_like(bbox[0]))

    intersection_min_x = np.maximum(anchor[1], bbox[1])
    intersection_max_x = np.minimum(anchor[1] + anchor[3] - 1, bbox[1] + bbox[3] - 1)
    intersection_width = np.maximum(intersection_max_x - intersection_min_x + 1, np.zeros_like(bbox[1]))

    area_intersection = intersection_height * intersection_width
    area_first = anchor[2] * anchor[3]
    area_second = bbox[2] * bbox[3]
    area_union = area_first + area_second - area_intersection
    
    iou = area_intersection * 1.0 / area_union
    iof = area_intersection * 1.0 / area_first
    ios = area_intersection * 1.0 / area_second

    return iou, iof, ios

def find_best_bboxes(anchors, bboxes, labels):
    best_bboxes = []
    best_labels = []
    ious = []
    for anchor in anchors:
        iou_max = 0
        best_bbox = None
        for i, bbox in enumerate(bboxes):
            iou = iou_anchor_bbox(anchor, bbox)[0]
            if iou > iou_max:
                iou_max = iou
                best_label = labels[i]
                best_bbox = bbox
        if best_bbox is None:
            best_bboxes.append(bbox)
            best_labels.append(labels[0])
        else:
            best_bboxes.append(best_bbox)
            best_labels.append(best_label)
        ious.append(iou_max)
    return np.array(best_bboxes), np.array(best_labels), np.array(ious)

def iou_bbox(bboxes1, bboxes2):
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

