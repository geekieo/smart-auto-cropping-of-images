from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
from scipy.ndimage import zoom
from scipy.misc import logsumexp
import tensorflow as tf

from gaze_vgg_layers import *
from utils.process import gaussian_kernel1d

VGG_MEAN = [103.939, 116.779, 123.68]


class GAZEVGG:

    def __init__(self, vgg16_npy_path=None, use_centerbias=True, gauss_factor=1, pred_h=480, pred_w=640):
        path = sys.modules[self.__class__.__module__].__file__
        path = os.path.abspath(os.path.join(path, os.pardir))

        if vgg16_npy_path is not None:
            if not os.path.isfile(vgg16_npy_path):
                # logging.error(("File '%s' not found. Download it from "
                #                "ftp://mi.eng.cam.ac.uk/pub/mttt2/"
                #                "model/vgg16.npy"), vgg16_npy_path)
                print(("File '%s' not found. Download it from ftp://mi.eng.cam.ac.uk/pub/mttt2/"
                        "model/vgg16.npy"), vgg16_npy_path)
                sys.exit(1)
        else:
            vgg16_npy_path = os.path.join(path, "../../model/vgg16.npy")
            print("Load ```npy file from '%s'.", vgg16_npy_path)
            # logging.info("Load ```npy file from '%s'.", vgg16_npy_path)
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.wd = 5e-4
        print("npy file loaded")

        if use_centerbias is True:
            centerbias_path = os.path.join(path, "centerbias.npy")
            if os.path.isfile(centerbias_path):
                # load precomputed log density over a 1024x1024 image
                centerbias_template = np.load(centerbias_path)  
                # rescale to match image size
                centerbias = zoom(centerbias_template, (pred_h/1024, pred_w/1024), order=0, mode='nearest')
                # renormalize log density
                centerbias -= logsumexp(centerbias)
                centerbias = centerbias[np.newaxis, :, :, np.newaxis]  # BHWC, 1 channel (log density)
                self.centerbias = tf.constant(centerbias, tf.float32)
                print("centerbias file loaded, shape", centerbias.shape)
        else:
            self.centerbias = None
            print("no centerbias")

        w_filter = gaussian_kernel1d(sigma=19)
        self.pad = int(w_filter.shape[1]/gauss_factor/2)
        print('gauss padding:',self.pad)
        self.h_filter, self.w_filter = filter_pool(w_filter, gauss_factor)
        print('gauss filter:',self.h_filter )

    def _VGG16(self, rgb, debug=False, trainable=False, norm=False, training=False):
        with tf.name_scope('VGG16'):
            # self.rgb = tf.cast(rgb, tf.uint8) # for summary
            self.rgb = tf.cast(rgb, tf.float32)
            # Convert RGB to BGR
            red, green, blue = tf.split(self.rgb, 3, 3)
            # assert red.get_shape().as_list()[1:] == [224, 224, 1]
            # assert green.get_shape().as_list()[1:] == [224, 224, 1]
            # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], 3)

            if debug:
                bgr = tf.Print(bgr, [tf.shape(bgr)],
                               message='Shape of input image: ',
                               summarize=4, first_n=1)
            # restore VGG16
            self.conv1_1 = conv_layer(bgr, "conv1_1", self.data_dict, self.wd, trainable, norm, training)
            self.conv1_2 = conv_layer(self.conv1_1, "conv1_2", self.data_dict, self.wd, trainable, norm, training)
            self.pool1 = max_pool(self.conv1_2, 'pool1', debug)

            self.conv2_1 = conv_layer(self.pool1, "conv2_1", self.data_dict, self.wd, trainable, norm, training)
            self.conv2_2 = conv_layer(self.conv2_1, "conv2_2", self.data_dict, self.wd, trainable, norm, training)
            self.pool2 = max_pool(self.conv2_2, 'pool2', debug)

            self.conv3_1 = conv_layer(self.pool2, "conv3_1", self.data_dict, self.wd, trainable, norm, training)
            self.conv3_2 = conv_layer(self.conv3_1, "conv3_2", self.data_dict, self.wd, trainable, norm, training)
            self.conv3_3 = conv_layer(self.conv3_2, "conv3_3", self.data_dict, self.wd, trainable, norm, training)
            self.pool3 = max_pool(self.conv3_3, 'pool3', debug)

            self.conv4_1 = conv_layer(self.pool3, "conv4_1", self.data_dict, self.wd, trainable, norm, training)
            self.conv4_2 = conv_layer(self.conv4_1, "conv4_2", self.data_dict, self.wd, trainable, norm, training)
            self.conv4_3 = conv_layer(self.conv4_2, "conv4_3", self.data_dict, self.wd, trainable, norm, training)
            self.pool4 = max_pool(self.conv4_3, 'pool4', debug)

            self.conv5_1 = conv_layer(self.pool4, "conv5_1", self.data_dict, self.wd, trainable, norm, training)
            self.conv5_2 = conv_layer(self.conv5_1, "conv5_2", self.data_dict, self.wd, trainable, norm, training)
            self.conv5_3 = conv_layer(self.conv5_2, "conv5_3", self.data_dict, self.wd, trainable, norm, training)


    def build_deepgaze(self, rgb, label=None, concat=False, debug=False, readout_norm=False, vgg_train=False, vgg_norm=False, training=True):
        """
        Build the VGG model using loaded weights
        Parameters
        ----------
        rgb: image batch tensor
            Image in rgb shap. Scaled to Intervall [0, 255]
        training: bool
            Whether to build train or inference graph
        debug: bool
            Whether to print additional Debug Information.
        """
        if label != None:
            self.label = tf.cast(label, tf.uint8)
        # rgb = avg_pool(rgb, "downsample_by_2", debug)
        self._VGG16(rgb, debug=debug, trainable=vgg_train, norm=vgg_norm, training=training)
        with tf.name_scope('Readout'):
            # readout network
            if concat:
                with tf.name_scope('concat'):
                    self.features = tf.concat([self.pool4, self.conv5_1, self.conv5_2, self.conv5_3], 3, "concat")  
                    f_channels = 2048
            else:
                self.features = self.conv5_3 
                f_channels = 512
            self.conv1 = conv_1x1_layer(self.features, [1, 1, f_channels, 32], "conv1", debug,'relu', readout_norm, training)
            self.conv2 = conv_1x1_layer(self.conv1, [1, 1, 32, 16], "conv2", debug, 'relu', readout_norm, training)
            self.conv3 = conv_1x1_layer(self.conv2, [1, 1, 16, 8], "conv3", debug, 'relu', readout_norm, training)
            self.conv4 = conv_1x1_layer(self.conv3, [1, 1, 8, 1], "conv4", debug, 'relu', readout_norm, training)
            self.deconv = deconv_layer(self.conv4, [32,32,1,1], 16, 'deconv', debug, readout_norm, training)
            self.gauss = gauss_layer(self.deconv, self.w_filter, self.h_filter, self.pad, 'blur', debug)
            if self.centerbias is not None:
                multiples=tf.concat([[tf.shape(self.gauss)[0]], [1], [1], [1]], 0, name='bias_shape')
                centerbias = tf.tile(self.centerbias, multiples, name='batch_centerbias')
                self.gauss = tf.add(self.gauss, centerbias,'add_centerbias')

            # self.lse = logsumexp_normalize(self.gauss, 'salmap_logsumexp', debug=debug)
            # self.softmax = softmax(self.lse, maxnorm=True, lineshape=False, summary=True, name='softmax')
        

    