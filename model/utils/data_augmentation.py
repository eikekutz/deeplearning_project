import math
import numbers
import random

from PIL import Image, ImageOps
import numpy as np
#from __future__ import division
import torch
import math
import sys
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

#import torch.nn.functional as F
import torchvision.transforms.functional as F
def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        #print(img[0].size(),mask[0].size())
        #print(img.size,mask.size)
        #assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask
class RandomHorizontalFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return F.vflip(img),F.vflip(mask)
        return img, mask
class RandomVerticalFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return F.hflip(img),F.hflip(mask)
        return img, mask
    def __repr__(self):
        return self.__class__.__name__ + '()'
class ToTensor(object):
    def __call__(self, img,mask):
        return F.to_tensor(img),F.to_tensor(mask)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToPILImage(object):
    def __init__(self, mode=None):
        self.mode = mode
    def __call__(self, img,mask):
        return F.to_pil_image(img, self.mode),F.to_pil_image(mask, self.mode)
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string
class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img,mask):
        
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)
        _,_, h_m, w_m = self.get_params(img, self.size)
        assert h==h_m and w==w_m
        return F.crop(img, i, j, h, w),F.crop(mask, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor