#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from scipy import ndimage

# 这个函数没用到，仅供查询插值参数
def resize(src, factor=8, interpolation="INTER_CUBIC"):
        '''
        shrink to the same area
        Calculate mean feature of input features
        Arg:
            src: numpy.ndarray. ndim is 2.
            factor: integer. resize factor
            interpolation: string. interpolation algorithm
        Retrun:
            dst: numpy.ndarray. ndim is 2.
        '''
        if interpolation == 'INTER_NEAREST ': 
            interpolation = cv2.INTER_NEAREST
        elif interpolation == 'INTER_LINEAR ':
            interpolation = cv2.INTER_LINEAR #default
        elif interpolation == 'INTER_CUBIC':
            interpolation = cv2.INTER_CUBIC
        elif interpolation == 'INTER_AREA':
            interpolation = cv2.INTER_AREA
        elif interpolation == 'INTER_LANCZOS4':
            interpolation = cv2.INTER_LANCZOS4
        dst = cv2.resize(src, (0, 0), fx=factor, fy=factor,
                         interpolation=interpolation)
        return dst


def channel_mean(features ,resize=False,factor=8,interpolation=cv2.INTER_CUBIC):
    '''
    Calculate the mean of input features
    Arg:
        features: numpy array, channel last, ndim is 4.
    Retrun:
        feature: numpy array, ndim is 2
        i: integer; channels
    '''
    feature = np.zeros(features.shape[1:3])
    if resize == False:
        feature = np.zeros(features.shape[1:3])
        for i in np.arange(features.shape[3]):
            feature += features[0, :, :, i]
    else:
        feature = np.zeros([_*factor for _ in features.shape[1:3]])
        for i in np.arange(features.shape[3]):
            feature += cv2.resize(features[0, :, :, i], (0, 0), fx=factor, 
                        fy=factor, interpolation=interpolation)
    feature /= features.shape[3]
    return feature


def gaussian_kernel1d(sigma, order=0, truncate=4.0):
    """One-dimensional Gaussian filter.
    Parameters
    ----------
    %(input)s
    sigma : scalar
        standard deviation for Gaussian kernel
    %(axis)s
    order : int, optional
        An order of 0 corresponds to convolution with a Gaussian
        kernel. A positive order corresponds to convolution with
        that derivative of a Gaussian.
    ----------
    Return: w_filter: shape:(1,width,1,1)
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    # Since we are calling correlate, not convolve, revert the kernel
    w_filter = ndimage.filters._gaussian_kernel1d(sigma, order, lw)[::-1]
    # h_filter = w_filter[np.newaxis, ...].T[..., np.newaxis, np.newaxis]
    w_filter = w_filter[np.newaxis, ..., np.newaxis, np.newaxis]
    return  w_filter


def size_norm(src, cut_ratio=8, strategy='area', norm_value=None):
    '''resize images to a small size with the shorter edge equals to short_edge
    Arg:
        src: list of 3D numpy array, the dim is HWC of a image, C=3
        cut_ratio: int or float, >1. maximum aspect ratio
        strategy: str. 
            default: 'area', the erea should equal to norm_value
            if 'short_edge', the shorter edge should equal to norm_value,
            if 'long_edge', the longer edge should equal to norm_value(smaller image size)
        norm_value: int.
    Return:
        dst: list of 3D numpy array, the dim is HWC of a image, C=3
        src.shape: if the src has been cut, the shape will be different and it will be used to resize the coordinates.
    '''
    try:
        if not isinstance(src,np.ndarray):
            print('Load Image Error! src type:%s'%type(src))
            return None

        # if the aspect ratio is bigger than cut_ratio, cut the image, leave the left or top part
        if src.shape[0]*cut_ratio < src.shape[1]:
            src = src[0:src.shape[0], 0:src.shape[0]*cut_ratio]
        elif src.shape[0] > src.shape[1]*cut_ratio:
            src = src[0:src.shape[1]*cut_ratio, 0:src.shape[1]]
        
        def area_norm(norm_value):
            ratio = np.sqrt(norm_value/(src.shape[1]*src.shape[0]))
            width = int(ratio*src.shape[1])
            height = int(ratio*src.shape[0])
            dst = cv2.resize(src, (width, height), interpolation=cv2.INTER_CUBIC)
            return dst

        def short_edge_norm(norm_value):
            # the shorter edge should equal to norm_value
            if src.shape[0] < src.shape[1]:
                # height < width, height = norm_value
                width = int(src.shape[1]/src.shape[0] * norm_value)
                dst = cv2.resize(src, (width, norm_value), interpolation=cv2.INTER_CUBIC)
            elif src.shape[0] > src.shape[1]:
                # height > width, width = norm_value
                height = int(src.shape[0] / src.shape[1] * norm_value)
                dst = cv2.resize(src, (norm_value, height), interpolation=cv2.INTER_CUBIC)
            else:
                dst = cv2.resize(src, (norm_value, norm_value), interpolation=cv2.INTER_CUBIC)
            return dst

        def long_edge_norm(norm_value):
            # the longer edge should equal to norm_value
            if src.shape[0] < src.shape[1]:
                # height < width, width = norm_value
                height = int(src.shape[0]/src.shape[1] * norm_value)
                dst = cv2.resize(src, (norm_value, height), interpolation=cv2.INTER_CUBIC)
            elif src.shape[0] > src.shape[1]:
                # height > width, height = norm_value
                width = int(src.shape[1]/src.shape[0] * norm_value)
                dst = cv2.resize(src, (width, norm_value), interpolation=cv2.INTER_CUBIC)
            else:
                dst = cv2.resize(src, (norm_value, norm_value), interpolation=cv2.INTER_CUBIC)
            return dst

        if strategy == 'area':
            if not norm_value:
                norm_value = 480*460
            dst = area_norm(norm_value)
        elif strategy=='short_edge':
            if not norm_value:
                norm_value = 480
            dst = short_edge_norm(norm_value)
        elif strategy=='long_edge':
            if not norm_value:
                norm_value = 640
            dst = long_edge_norm(norm_value)
        else:
            dst = area_norm(480*640)
    except Exception as e:
        print(e)
    return dst, src.shape

def resize_coordinate(y, x, h, w, src_shape, dst_shape):
    '''
    y, x, h, w: int
    src_shape, dst_shape: HWC
    '''
    ratio = dst_shape[0] / src_shape[0]
    if x==0:
        w = dst_shape[1]
        h = int(h*ratio)
        y = int(y*ratio)
    elif y==0:
        h = dst_shape[0]
        w = int(w*ratio)
        x = int(x*ratio)
    else:
        # print('Not 1d crop coordinate')
        w = int(w*ratio)
        x = int(x*ratio)
        h = int(h*ratio)
        y = int(y*ratio)
    return y, x, h, w

if __name__ == "__main__":
    # w_filter = gaussian_kernel1d(19)
    # print(w_filter.shape)

    # test size_norm
    # import sys
    # sys.path.append("D:\\PySpace\\SACI")
    # import matplotlib.pyplot as plt
    # from utils.io import imread
    # src = imread(r'test_data/cat.jpg')
    # dst = size_norm(src)
    # plt.imshow(dst)
    # plt.show()

    # test resize_coordinate
    ny=0;nx=100;nh=600;nw=200
    src_shape=(400,600)
    dst_shape=(800,1200)
    y, x, h, w  = resize_coordinate(ny, nx, nh, nw, src_shape, dst_shape)
    print(y,x,h,w)