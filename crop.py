# -*- coding: utf-8 -*-
'''
use the window to find a position in the image where the window contains the most salient points.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import matplotlib.pyplot as plt

'''======================================== crop_1d head ========================================'''


def get_central_x(x_list, x_range):
    '''get the most central x in x_range
    Arg:
        x_list: list of int; item range is [0, x_range]
        x_range: int
    Return:
        x: int; the most central x
    '''
    center = x_range / 2
    x = x_list[0]
    dist = abs(x - center)
    centeral_x = [dist, x]  # initial
    for x in x_list:
        dist = abs(x - center)
        if dist < centeral_x[0]:
            centeral_x = [dist, x]
    return centeral_x[1]


def pick_x(sum_list, central_bias=True):
    '''Pick the max summation x using center bias
    Arg:
        sum_list: list of int
        central_bias: boolean; use central bias
    Return:
        x: int; x range is [0, len(sum_list)]
    '''
    max_list = [sum_list[0], 0]  # [summation, *index]
    for i, sum_item in enumerate(sum_list):
        if sum_item < max_list[0]:
            continue
        elif sum_item > max_list[0]:
            max_list = [sum_item, i]
        else:  # sum_item == max_list[0]
            max_list.append(i)
    if central_bias == True:
        return get_central_x(max_list[1:], len(sum_list))
    else:
        return max_list[1]


# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum() 

def zero_norm(x):
    """Translation the min of x to zero"""
    return x - np.min(x)


def linear_weight_1d(length, max_pos, max_weight=1, min_weight=0.5):
    '''         0    max_pos↴
       weights: +----------+++-------------+
                |←left_len→| |← right_len →|
    Arg:
        length: int
        max_pos: position of max_weight form 0
    Return:
        weights: list, dtype float
    '''

    weight_range = float(max_weight - min_weight)
    left_len = int(max_pos)
    right_len = int(length - max_pos - 1)

    def generate_weight(reverse=False):
        weights = [min_weight for _ in np.arange(left_len + 1)]  # initialize weights with min_weight
        for i in np.arange(left_len + 1):
            weights[i] = weights[i] + weight_range * i / (left_len)  # calculate linear weights
        weights.extend(list(reversed(weights))[1:right_len + 1])
        if reverse:
            weights = list(reversed(weights))
        return weights

    if left_len > right_len:
        weights = generate_weight(reverse=False)
    else:
        left_len, right_len = right_len, left_len
        max_pos = right_len
        weights = generate_weight(reverse=True)
    return weights


# print(linear_weight_1d(5, 2, max_weight=1, min_weight=0.5))    # test


def linear_nonmax_restrain(x, max_weight=1, min_weight=0.5):
    """add linear non-max restrain on x
    Arg:
        x: numpy array, ndim=1
        min_weight: float
    """
    max_pos = np.argmax(x)
    length = x.size
    weights = linear_weight_1d(length, max_pos, max_weight, min_weight)
    x = zero_norm(x)
    return x * weights


def linear_restrain(x, max_weight=1, min_weight=0.7):
    """add linear restrain on x
    Arg:
        x: numpy array, ndim=1
        min_weight: float. minimum weight of restrain.
    """
    length = x.size
    weights = linear_weight_1d(length, int(length / 2), max_weight, min_weight)
    x = zero_norm(x)
    return x * weights


src_1d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
restrain = linear_restrain(src_1d, max_weight=1, min_weight=0.5)  # test


def arc_restrain(x):
    """add arc restrain on x
    Arg:
        x: numpy array, ndim=1
    """
    pass


def slide_window(src_1d, win_l, src_restrain=0.5):
    u'''Get the position of the slide window, which has max summation and maximum inside
                  x ↴
        src_1d: +---+---→--+---------+
                   →|window|←
    Arg:
        src_1d: np.ndarray, ndim: 1, dtype float64
        win_l: int; length of the 1D slide window
    Return:
        x: int; left vertex coordinate of the slide window
    '''
    max_index = np.argmax(src_1d)  # src_1d 最大值索引
    src_1d_l = src_1d.shape[0]  # src_1d 长度
    # 用 src_1d 最大值锚定滑窗最大滑动范围，两倍滑窗窗宽，也是 src_1d 需要抑制的范围
    slide_left_i = max_index - win_l + 1 if max_index - win_l + 1 > 0 else 0  # 滑窗滑动最左索引
    slide_right_i = max_index + win_l - 1 if max_index + win_l - 1 < src_1d_l - 1 else src_1d_l - 1  # 滑窗滑动最右索引
    slide_l = slide_right_i - slide_left_i + 1  # 滑窗滑动范围

    slide_src = src_1d[slide_left_i:slide_right_i + 1]  # 抠出滑窗最大滑动范围内的图像
    if src_restrain is not None and src_restrain >= 0 and src_restrain <= 1:
        slide_src = linear_nonmax_restrain(slide_src, max_weight=1, min_weight=src_restrain)

    sum_list = []  # list of win_sum
    win_sum = np.sum(slide_src[0:win_l])  # initial; sum of all elements in window
    sum_list.append(win_sum)

    for x in np.arange(slide_l - win_l):
        win_sum = win_sum - slide_src[x] + slide_src[x + win_l]
        sum_list.append([win_sum])
    x = pick_x(sum_list, central_bias=False)  # 0~len(sum_list)
    return x + slide_left_i


# src_1d=np.array([1,2,1,13,100,1,14,8,9,0])
# print(slide_window(src_1d, win_l=3, src_restrain=0.5))


def crop_1d(src, d_size, src_restrain=0.5):
    u'''Crop image in 1 dimension by the max summation slide window.
    Annotation: 
        sliding direction for crop window
        0: width direction;         1: height direction   ↓
           +---+-------+---            +-----------+--------  
           |   |       | ↑             |     |     | ↑ win_h
           | ----→     |src_h, win_h   +-----|-----+src_h---
           |   |       | ↓             |     ↓     | ↓    ↑
           +---+-------+---            +-----------+---
           |← →|win_w  |               |←  src_w  →|
           |←  src_w  →|               |   win_w   |
    Arg:
        src: np.ndarray, ndim: 2, dtype float64; src image
        d_size: tuple of (height, width), dtype int; dst size; NOTE: 注意格式是高宽！
        src_restrain: float, 0~1. if 1: no restrain; if 0: max restrain 
    Return: 
        [win_x, win_y, win_w, win_h]: list, dtype int; 
            win_x, win_y: Left vertex coordinates of crop window
            win_w, win_h: width and height of crop window
    '''
    # reader-friendly
    src_h = src.shape[0]  # height of src image
    src_w = src.shape[1]  # width of src image
    crop_h = d_size[0]  # height of cropped image
    crop_w = d_size[1]  # width of cropped image

    src_ratio = src_h / src_w
    crop_ratio = crop_h / crop_w
    direction = 0 if crop_ratio > src_ratio else 1  # see Annotation
    if direction == 0:  # width direction
        win_h = src_h
        win_w = int(src_h * crop_w / crop_h)
        win_y = 0
        src_1d = np.max(src, axis=0)  # 2D→1D, get max over height axis.
        win_x = slide_window(src_1d, win_w, src_restrain)
    else:  # height direction
        win_w = src_w
        win_h = int(src_w * crop_h / crop_w)
        win_x = 0
        src_1d = np.max(src, axis=1)  # 2D→1D, get max over width axis.
        win_y = slide_window(src_1d, win_h, src_restrain)
    return win_y, win_x, win_h, win_w


'''======================================== crop_1d end ========================================'''


def visualize_crop(src, saliency_map, d_size=(392, 697)):
    u'''可视化展示'''
    re_saliency_map = cv2.resize(saliency_map, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)
    # print('saliency_map:%s, re_saliency_map:%s'%(saliency_map.shape, re_saliency_map.shape))

    y, x, h, w = crop_1d(re_saliency_map, d_size, src_restrain=0.5)

    gray_map = cv2.normalize(re_saliency_map, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    heat_map = cv2.applyColorMap(255 - gray_map, cv2.COLORMAP_JET)
    fusion = cv2.addWeighted(src, 0.4, heat_map, 0.6, 0)
    # 融合带窗
    fusion_win = np.copy(fusion)
    cv2.rectangle(fusion_win, (x, y), (x + w, y + h), (255, 0, 0), 8)
    # 裁图结果
    src_win = np.copy(src)
    cv2.rectangle(src_win, (x, y), (x + w, y + h), (255, 0, 0), 8)
    return y, x, h, w, heat_map, fusion_win, src_win


'''======================================== crop_mask head ========================================'''
centerbias = np.load('fitbias.npy')


def mask(src_shape, centerbias):
    pass


'''======================================== crop_mask end ========================================'''


if __name__ == "__main__":
    from utils.io import *


    def plot(img1, img2, img3, img4):
        plt.figure(figsize=(10, 5))
        plt.subplot(141)
        plt.title('src')
        plt.axis('off')
        plt.imshow(img1)
        plt.subplot(142)
        plt.axis('off')
        plt.imshow(img2)
        plt.subplot(143)
        plt.axis('off')
        plt.imshow(img3)
        plt.subplot(144)
        plt.axis('off')
        plt.imshow(img4)
        # plt.get_current_fig_manager().window.showMaximized()
        plt.show()


    def crop(sample_num=3):
        '''Crop image using corp_1d()'''
        # batch crop use crop_1d()
        file_dir = 'D:\dataset\crop_image\deepgazeII'
        img_list = get_image_list(file_dir)
        file_list, co_img_list = get_cofile_list(file_dir, img_list, ext='npy')
        # save_dir = make_save_dir(file_dir)
        for i in np.arange(min(len(file_list), sample_num)):
            src_path = co_img_list[i]
            src = imread(src_path)

            saliency_map = np.load(file_list[i])

            y, x, h, w, heat_map, fusion_win, src_win = visualize_crop(src, saliency_map)
            print(y, x, h, w, )
            y, x, h, w, heat_map, fusion_win, src_win = visualize_crop(src, saliency_map)
            print(y, x, h, w, )
            yr, xr, hr, wr, heat_mapr, fusion_winr, src_winr = visualize_crop(src, saliency_map)
            print(yr, xr, hr, wr)
            plot(fusion_win, fusion_winr, src[y:(y + h), x:(x + w)], src[yr:(yr + hr), xr:(xr + wr)])

            # plot(src, heat_map, fusion_win, src[y:(y+h), x:(x+w)])
            # imwrite(fusion_winr, save_dir, src_path, '_restrain')
            # imwrite(src[yr:(yr+hr), xr:(xr+wr)], save_dir, src_path, '_restrain')

    # crop()
