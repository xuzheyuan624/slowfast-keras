import cv2
import random
import numpy as np
import collections
import numbers

class Compose(object):
    def __init__(self, transforms=[]):
        self.transforms = transforms
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class Normalize(object):
    def __init__(self, mean=[128, 128, 128], std=[128, 128, 128]):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = (img - np.array([[self.mean]])) / np.array([[self.std]])
        return img

    def randomize_parameters(self):
        pass

class Scale(object):
    def __init__(self, size, interpolation=cv2.INTER_AREA):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size)==2)

        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            h, w = img.shape[:2]
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return cv2.resize(img, (ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return cv2.resize(img, (ow, oh), self.interpolation)
        else:
            return cv2.resize(img, self.size, self.interpolation)

    def randomize_parameters(self):
        pass

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]
        th, tw = self.size
        y1 = int(round((h - th)/2.))
        x1 = int(round((w - tw)/2.))
        return img[y1:y1+th, x1:x1+tw, :]


class CornerCrop(object):
    def __init__(self, size, crop_position=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']
        self.crop_position = crop_position

    def __call__(self, img):
        h, w = img.shape[:2]

        if self.crop_position == 'c':
            th, tw = self.size
            y1 = int(round((h - th)/2.))
            x1 = int(round((w - tw)/2.))
            y2 = y1+th
            x2 = x1+tw
        elif self.crop_position == 'tl':
            y1 = 0
            x1 = 0
            y2 = th
            x2 = tw
        elif self.crop_position == 'tr':
            y1 = 0
            x1 = w - tw
            y2 = th
            x2 = w
        elif self.crop_position == 'bl':
            y1 = h - th
            x1 = 0
            y2 = h
            x2 = tw
        elif self.crop_position == 'br':
            y1 = h - th
            x1 = w - tw
            y2 = h
            x2 = w

        return img[y1:y2, x1:x2, :]

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = random.choice(self.crop_positions)

class RandomHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, img):
        if self.p < 0.5:
            return img[:, ::-1, :]
        else:
            return img
    
    def randomize_parameters(self):
        self.p = random.random()

class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img):
        h, w = img.shape[:2]
        th, tw = self.size
        x1 = int(self.x * (w - tw))
        y1 = int(self.y * (h - th))
        x2 = x1 + tw
        y2 = y1 + th
        return img[y1:y2, x1:x2, :]

    def randomize_parameters(self):
        self.x = random.random()
        self.y = random.random()

