#-*- coding: utf-8 -*-
'''
use os, opencv to i/o image
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2

from datetime import datetime

def get_image_list(img_dir, prefix='', postfix='', contain_str='',debug=False):
    '''get images in the <img_dir>
    Arg:
        img_dir: str; img directory
    Return:
        img_path_list: list of img_path
    '''
    assert(os.path.exists(img_dir))
    imgs = os.listdir(img_dir)
    img_path_list = []
    for img_name in imgs:
        name, ext = os.path.splitext(img_name)
        if not name.startswith(str(prefix)) and prefix is not '':
            continue
        elif not name.endswith(str(postfix)) and postfix is not '':
            continue
        elif not str(contain_str) in name and contain_str is not '':
            continue
        elif ext.lower() in ['.jpg', '.png', '.bmp', '.jpeg']:
            img_path_list.append(os.path.join(img_dir, img_name))
    if debug:
        print('Get %d pathes of image. ' % (len(img_path_list)))
    return img_path_list


def get_cofile_list(file_dir, img_path_list=[], ext='npy'):
    '''
    get correlation file by img_path_list with a specify extension
    Arg:
        file_dir: str; file directory
        img_path_list: list of string
        ext: string; extension of files
    Return:
        file_list: list of new extension file pathes
        co_img_list: list of corresponding image pathes
    '''
    file_list = []
    co_img_list = []
    for img_path in img_path_list:
        basename = os.path.basename(img_path)
        filename = os.path.splitext(basename)[0]
        file_path = os.path.join(file_dir, filename +'.' + ext) 
        
        if os.path.exists(file_path):
            file_list.append(file_path)
            co_img_list.append(img_path)
        else:
            print('File not exist: "%s"' % file_path)
    print('Get %d pathes in list. Return %d sets of "%s" file pathes.' % (
        len(img_path_list), len(file_list), ext))
    return file_list, co_img_list


def gen_save_dir(father_path):
    save_dir = extend_path(father_path, postfix='\Result_', time=True)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


def extend_path(path, postfix='', time=False):
    if isinstance(postfix, str) and not postfix == '':
        path += postfix
    if time:
        path += datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    return path


def imread(src_path, BGR2RGB=True):
    src = cv2.imread(src_path,1)
    if src is not None:
        if BGR2RGB==True:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        return src
    else:
        print('imread is None, path: %s'%src_path)
        return None

def make_save_dir(root_path):
    save_dir = root_path+'\Result_%s'%datetime.now().strftime('%Y%m%d%H%M%S')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    return save_dir

def imwrite(dst, save_dir, src_path, postfix='', ext='.png'):
    basename = os.path.basename(src_path)
    filename = os.path.splitext(basename)[0]
    save_path = os.path.join(save_dir, filename+postfix+ext)
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, dst, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    data_dir = r'D:\dataset\crop_image\deepgazeII'
    img_list = get_image_list(data_dir)
    get_cofile_list(data_dir, img_list, ext='npy')

    data_dir = r'D:\PySpace\SACI\test_data\images'
    img_list = get_image_list(data_dir, prefix='COCO_val')
    print(len(img_list))
    img_list = get_image_list(data_dir, prefix='COCO_train')
    print(len(img_list))
    img_list = get_image_list(data_dir, contain_str='00000018')
    print(len(img_list))
    img_list = get_image_list(data_dir, postfix='198447')
    print(len(img_list))
    img_list = get_image_list(data_dir)
    print(len(img_list))