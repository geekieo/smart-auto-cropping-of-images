#-*- coding: utf-8 -*-
# 以载入模型的形式使用模型（推荐）
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import cv2
from crop import crop_1d
from utils.process import size_norm, resize_coordinate


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # ignore memory warning
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # 指定 GPU
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True

class DeepGaze():
    def __init__(self, init_img=r'test_data/cat.jpg'):    
        '''
        init_img: str. a path of a local image.
        '''                 
        checkpoint = r'checkpoints/deepgaze.ckpt-29999'  # DeepGaze II
        new_saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint))

        self.sess = tf.Session(config=tfconfig)
        new_saver.restore(self.sess, checkpoint)
        self.sess.graph.finalize() # graph 只读
        graph = tf.get_default_graph()
        self.input_image = graph.get_operation_by_name(r'input_images').outputs[0]
        self.gauss = graph.get_operation_by_name(r'Readout/blur/Conv2D_1').outputs[0]
        print('Deepgaze model construct!')
        self._init_predict(init_img)

    def _init_predict(self, init_img):
        try:
            src = imread(init_img)
            gauss, src_shape = self.predict(src)
            print('DeepGaze prediction initialized!')
        except Exception as e:
            print(str(e) if isinstance(str(e),str) else type(e)) #tensorflow 抛出的图异常不能转换成str

    def predict(self, src):
        if src is None:
            print('src is None')
            return
        src_norm, src_shape = size_norm(src, cut_ratio=7, strategy='area') # short_edge/area
        src_4D = np.expand_dims(src_norm, 0)
        gauss = self.sess.run(self.gauss, {self.input_image: src_4D})
        gauss_norm = cv2.resize(gauss[0], (src_norm.shape[1], src_norm.shape[0]), interpolation=cv2.INTER_LINEAR) #微调至src尺寸
        return gauss_norm, src_shape
        
    def crop(self, gauss_norm, src_shape, d_size=(392, 698), debug=False):
        ny, nx, nh, nw = crop_1d(gauss_norm, d_size=d_size, src_restrain=0.6)
        y, x, h, w  = resize_coordinate(ny, nx, nh, nw, gauss_norm.shape, src_shape)
        if debug:
            print('Deepgaze OK! gauss_norm shape',gauss_norm.shape)
            print('src_norm crop (y,x,h,w)', ny, nx, nh, nw)
            print('src shape ', src_shape)
            print('src crop (y,x,h,w)', y, x, h, w )
        return y, x, h, w 


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import time
    from utils.io import imread
    from crop import visualize_crop


    deepgaze = DeepGaze()

    src_list = [
        'D:/dataset/crop_image/badcase_deepgaze_v1/ABFB402DC6980083AC1B80439693A9F40EF0D8FB_size220_w690_h969.jpeg',
        'D:/dataset/crop_image/badcase_deepgaze_v1/0CEAA584BFC4D38266DCF18B86E06C389EBDAB55_size112_w300_h534.jpeg',
    ]

    for i, src in enumerate(tqdm(src_list)):
        begin = time.time()
        src = imread(src)
        if src is None:
            print('WARNING: src is None')
            continue
        fig = plt.figure('figure_'+str(i), figsize=(19.0, 10.0))  # in inches!
        plt.subplot(141)
        plt.imshow(src)
        gauss, src_shape = deepgaze.predict(src)
        plt.subplot(142)
        plt.imshow(gauss)
        y, x, h, w = deepgaze.crop(gauss, src_shape,(392, 698),True)   # (312, 698)
        plt.subplot(143)
        plt.imshow(src[y:y+h, x:x+w])
        y, x, h, w = deepgaze.crop(gauss, src_shape,(392, 698),True)
        plt.subplot(143)
        plt.imshow(src[y:y+h, x:x+w])
        print('留白crop测试')
        ny, nx, nh, nw = deepgaze.crop(gauss, src_shape,(312, 698),True)
        ny -= int((h-nh)/2)
        nh = h
        plt.subplot(144)
        plt.imshow(src[ny:ny+nh, nx:nx+nw])
        print(y,ny)
        # plt.savefig('D:/experiment/' + str(i)+'_test')
        # plt.close()
        print('time cost:', time.time() - begin)
        plt.show()
    print('Done.')