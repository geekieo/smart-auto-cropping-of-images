'''
裁图结果展示
6幅图像为1组。  
左上：原图           中上：热力图+裁图框    右上：热力图+非极大值约束裁图框
左下：传统算法结果    中下：新算法结果       右下：新算法+非极大值约束结果
'''
import sys
sys.path.append("D:\\PySpace\\SACI")

import os
import cv2
import numpy as np

from utils.io import get_image_list, get_cofile_list, imread, make_save_dir, imwrite
import matplotlib.pyplot as plt
from utils.process import size_norm

data_dir= r'D:\dataset\crop_image\crop_images\ori_images'
src_list = get_image_list(r'D:\dataset\crop_image\crop_images\ori_images')
# old_crop_dir = r'D:\dataset\crop_image\crop_images\ori_images\legacy_Result_20180916124943'
old_crop_dir = r'D:\dataset\crop_image\crop_images\ori_images\edge_640_Result_20180913111931'
new_crop_dir = r'D:\dataset\crop_image\crop_images\ori_images\edge_480_Result_20180915001155'


crop_list=[]
cropflat_list=[]

for img in src_list:
    name, ext = os.path.splitext(img)
    crop_path = name+'_crop'+ext
    crop_list.append(crop_path)
    cropflat_path = name+'_cropflat'+ext
    cropflat_list.append(cropflat_path)

old_crop_list,_ = get_cofile_list(old_crop_dir, crop_list, ext='png')
old_cropflat_list,_ = get_cofile_list(old_crop_dir, cropflat_list, ext='png')
new_crop_list,_ = get_cofile_list(new_crop_dir, crop_list, ext='png')
new_cropflat_list,_ = get_cofile_list(new_crop_dir, cropflat_list, ext='png')


def splice_3img(src, cropold, cropnew):
    # 单对结果竖向排版
    src_height = src.shape[0]
    src_width = src.shape[1]
    crop_shape = cropold.shape
    crop_width = int(src_width/2)
    crop_height = int(crop_shape[0]/crop_shape[1]*crop_width)
    resize_cropold = cv2.resize(cropold, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
    resize_cropnew = cv2.resize(cropnew, ((src_width-crop_width), crop_height), interpolation=cv2.INTER_CUBIC)
    dst_height = src_height+crop_height
    dst_width = src_width
    dst = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
    np.copyto(dst[0:src_height, 0:dst_width], src)
    np.copyto(dst[src_height:dst_height, 0:crop_width], resize_cropold)
    np.copyto(dst[src_height:dst_height, crop_width:dst_width], resize_cropnew)
    print(src.shape, dst.shape)
    return dst

def splice_5img(src, crop1old, crop1new, crop2old, crop2new):
    # 两对结果竖向排版
    sh = src.shape[0]       #sh：src_height
    sw = src.shape[1]       #sw：src_width
    ch1= crop1old.shape[0]  #ch: crop1old_height
    cw1= crop1old.shape[1]  #cw: crop1old_width
    ch2= crop2old.shape[0]
    cw2= crop2old.shape[1]

    rw = int(sw/2)          #rw: resize_crop1old_width, resize_crop2old_width
    rh1 = int(ch1/cw1*rw)   #rh1:resize_crop1old_height
    rh2 = int(ch2/cw2*rw)   #rh2:resize_crop2old_height

    resize_crop1old = cv2.resize(crop1old, (rw, rh1), interpolation=cv2.INTER_CUBIC)
    resize_crop1new = cv2.resize(crop1new, ((sw-rw), rh1), interpolation=cv2.INTER_CUBIC)
    print(resize_crop1old.shape, resize_crop1new.shape)
    resize_crop2old = cv2.resize(crop2old, (rw, rh2), interpolation=cv2.INTER_CUBIC)
    resize_crop2new = cv2.resize(crop2new, ((sw-rw), rh2), interpolation=cv2.INTER_CUBIC)
    
    
    dh = sh+rh1+rh2
    dw = sw
    dst = np.zeros((dh, dw, 3), dtype=np.uint8)
    np.copyto(dst[0:sh, 0:dw], src)
    np.copyto(dst[sh:sh+rh1, 0:rw], resize_crop1old)
    np.copyto(dst[sh:sh+rh1, rw:dw], resize_crop1new)
    np.copyto(dst[sh+rh1:dh, 0:rw], resize_crop2old)
    np.copyto(dst[sh+rh1:dh, rw:dw], resize_crop2new)
    return dst

def splice_7img(src, crop1old, crop1new, crop2old, crop2new, crop3old, crop3new):
    # 三对结果横向排版
    sh = src.shape[0]
    sw = src.shape[1]
    ch1= crop1old.shape[0]
    cw1= crop1old.shape[1]
    ch2= crop2old.shape[0]
    cw2= crop2old.shape[1]
    ch3= crop3old.shape[0]
    cw3= crop3old.shape[1]
    rw =  int(sh / (ch1/cw1+ch2/cw2+ch3/cw3))
    rh1 = int(ch1/cw1*rw)
    rh2 = int(ch2/cw2*rw)
    rh3 = sh - rh1 -rh2
    dst = np.zeros((sh, (sw+rw+rw), 3), dtype=np.uint8)
    np.copyto(dst[0:sh, 0:dw], src)

    return dst

save_dir = make_save_dir(data_dir)#savedir
for i in np.arange(len(src_list)):
    src = imread(src_list[i])
    old_crop = imread(old_crop_list[i])
    old_cropflat = imread(old_cropflat_list[i])
    new_crop = imread(new_crop_list[i])
    new_cropflat = imread(new_cropflat_list[i])
    #resize to compress image
    src = size_norm(src)
    #splice image
    compare = splice_5img(src, old_cropflat, new_cropflat, old_crop, new_crop)

    # plt.figure(figsize=(6,9))
    # plt.imshow(compare)
    # plt.show()

    #save image
    imwrite(compare, save_dir, src_list[i], postfix='_compare', ext='.jpg')
