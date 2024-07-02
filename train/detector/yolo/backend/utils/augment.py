# -*- coding: utf-8 -*-
from imgaug import augmenters as iaa
import cv2
import numpy as np
np.random.seed(1337)
import cv2

class ImgAugment(object):
    def __init__(self, h, w, jitter):
        """
        # Args
            desired_w : int
            desired_h : int
            jitter : bool
        """
        self._jitter = jitter
        self._w = w
        self._h = h
        
    def imread(self, img_file, boxes):
        """
        # Args
            img_file : str
            boxes : array, shape of (N, 4)
        
        # Returns
            image : 3d-array, shape of (h, w, 3)
            boxes_ : array, same shape of boxes
                jittered & resized bounding box
        """
        # 1. read image file
        image = cv2.imread(img_file)

        if image is None:
            print("Image Path: " + img_file)
            raise ValueError
    
        # 2. make jitter on image
        boxes_ = np.copy(boxes)
        if self._jitter:
            image, boxes_ = make_jitter_on_image(image, boxes_)
    
        # 3. resize image            
        image, boxes_ = resize_image(image, boxes_, self._w, self._h)
        return image, boxes_

    def make_jitter(self, img, boxes):
        if self._jitter:
            img, boxes = make_jitter_on_image(img, boxes)
        return img, boxes

def make_jitter_on_image(image, boxes):
    h, w, _ = image.shape

    ### scale the image
    scale = np.random.uniform() / 10. + 1.
    image = cv2.resize(image, (0,0), fx = scale, fy = scale)

    ### translate the image
    max_offx = (scale-1.) * w
    max_offy = (scale-1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    
    image = image[offy : (offy + h), offx : (offx + w)]


    aug_pipe = _create_augment_pipeline()
    image = aug_pipe.augment_image(image)
    
    # fix object's position and size
    new_boxes = []
    for box in boxes:
        x1,y1,x2,y2 = box
        x1 = int(x1 * scale - offx)
        x2 = int(x2 * scale - offx)
        
        y1 = int(y1 * scale - offy)
        y2 = int(y2 * scale - offy)

        new_boxes.append([x1,y1,x2,y2])
    return image, np.array(new_boxes)


def resize_image(image, boxes, desired_w, desired_h):
    h, w, _ = image.shape
    
    # resize the image to standard size
    image = cv2.resize(image, (desired_h, desired_w))
    image = image[:,:,::-1]

    # fix object's position and size
    new_boxes = []
    for box in boxes:
        x1,y1,x2,y2 = box
        x1 = int(x1 * float(desired_w) / w)
        x1 = max(min(x1, desired_w - 1), 0)
        x2 = int(x2 * float(desired_w) / w)
        x2 = max(min(x2, desired_w - 1), 0)
        
        y1 = int(y1 * float(desired_h) / h)
        y1 = max(min(y1, desired_h - 1), 0)
        y2 = int(y2 * float(desired_h) / h)
        y2 = max(min(y2, desired_h - 1), 0)

        new_boxes.append([x1,y1,x2,y2])
    return image, np.array(new_boxes)


def _create_augment_pipeline():
    
    ### augmentors by https://github.com/aleju/imgaug
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    aug_pipe = iaa.Sequential(
        [
            iaa.SomeOf((0, 5),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    ]),
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                   
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return aug_pipe
