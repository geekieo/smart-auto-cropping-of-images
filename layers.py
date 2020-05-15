# -*- coding: utf-8 -*-
'''
@Description: 
@Date: 2018-07-18 16:48:09
@Author: Weng Jingyu
'''
import tensorflow as tf

# vgg16


def avg_pool(bottom, name, debug=False):
    '''down scale by 2'''
    pool = tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=name)

    if debug:
        pool = tf.Print(pool, [tf.shape(pool)],
                        message='Shape of %s' % name,
                        summarize=4, first_n=1)
    return pool


def max_pool(bottom, name, debug=False):
    pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=name)

    if debug:
        pool = tf.Print(pool, [tf.shape(pool)],
                        message='Shape of %s' % name,
                        summarize=4, first_n=1)
    return pool


def conv_layer(bottom, name, data_dict, wd, trainable=False, norm=False, training=False):
    with tf.variable_scope(name) as scope:
        filters = get_conv_filter(name, data_dict, wd, trainable)
        x = tf.nn.conv2d(bottom, filters, [1, 1, 1, 1], padding='SAME')
        
        biases = get_bias(name, data_dict, trainable)
        x = tf.nn.bias_add(x, biases)
        
        x = tf.nn.relu(x)

        if norm == True:
            x = tf.layers.batch_normalization(x, training=training, name='batch_norm')

        _activation_summary(filters, range=True)
        _activation_summary(biases, mean=True)
        _activation_summary(x, range=True)
        _mean_feature_summary(x)
        return x

# def upscore_layer(bottom, shape,
#                     out_c, name, debug,
#                     ksize=4, stride=2,):
#     '''deconv layer, up scale
#     out_c: out channels
#     '''
#     strides = [1, stride, stride, 1]
#     with tf.variable_scope(name):
#         in_channels = bottom.get_shape()[3].value

#         if shape is None:
#             # Compute shape out of Bottom
#             in_shape = tf.shape(bottom)

#             h = (in_shape[1] * stride)
#             w = (in_shape[2] * stride)
#             new_shape = [in_shape[0], h, w, out_c]
#         else:
#             new_shape = [shape[0], shape[1], shape[2], out_c]
#         output_shape = tf.stack(new_shape)
#         logging.debug("Layer: %s, Fan-in: %d" % (name, in_channels))
#         f_shape = [ksize, ksize, out_c, in_channels]

#         # create
#         num_input = ksize * ksize * in_channels / stride
#         stddev = (2 / num_input)**0.5

#         weights = get_deconv_filter(f_shape)
#         deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
#                                         strides=strides, padding='SAME')

#         if debug:
#             deconv = tf.Print(deconv, [tf.shape(deconv)],
#                                 message='Shape of %s' % name,
#                                 summarize=4, first_n=1)

#         _activation_summary(deconv)
#     return deconv

# # sub-modules
# def get_deconv_filter(f_shape):
#     width = f_shape[0]
#     height = f_shape[1]
#     f = np.ceil(width/2.0)
#     c = (2 * f - 1 - f % 2) / (2.0 * f)
#     bilinear = np.zeros([f_shape[0], f_shape[1]])
#     for x in range(width):
#         for y in range(height):
#             value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
#             bilinear[x, y] = value
#     weights = np.zeros(f_shape)
#     for i in range(f_shape[2]):
#         weights[:, :, i, i] = bilinear

#     init = tf.constant_initializer(value=weights,
#                                     dtype=tf.float32)
#     return tf.get_variable(name="up_filter", initializer=init,
#                             shape=weights.shape)


def get_conv_filter(name, data_dict, wd, trainable):
    init = tf.constant_initializer(value=data_dict[name][0],
                                   dtype=tf.float32)
    shape = data_dict[name][0].shape
    print('Layer name: %s' % name, ' shape: %s' % str(shape))
    var = tf.get_variable(name="filters", initializer=init, trainable=trainable, shape=shape)
    if not tf.get_variable_scope().reuse:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                   name='weight_loss')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             weight_decay)
    return var


def get_bias(name, data_dict, trainable):
    bias_wights = data_dict[name][1]
    shape = data_dict[name][1].shape
    init = tf.constant_initializer(value=bias_wights, dtype=tf.float32)
    return tf.get_variable(name="biases", initializer=init, trainable=trainable, shape=shape)


def _bias_reshape(bweight, num_orig, num_new):
    """ Build bias weights for filter produces with `_summary_reshape`

    """
    n_averaged_elements = num_orig//num_new
    avg_bweight = np.zeros(num_new)
    for i in range(0, num_orig, n_averaged_elements):
        start_idx = i
        end_idx = start_idx + n_averaged_elements
        avg_idx = start_idx//n_averaged_elements
        if avg_idx == num_new:
            break
        avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
    return avg_bweight

# readout
def conv_1x1_layer(bottom, f_shape, name, debug=False, activation='conv_1x1', norm=True, training=False, bias=True):
    '''
    f_shape: [height, width, in_channels, out_channels]
    '''
    with tf.variable_scope(name) as scope:
        filters = tf.get_variable('filters',
                                  shape=f_shape,
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        x = tf.nn.conv2d(bottom, filters, strides=[1, 1, 1, 1], padding='SAME')
        
        if bias == True:
            biases = tf.get_variable('biases',
                                    shape=[f_shape[3]],  # out_channels
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            x = tf.nn.bias_add(x, biases)
            _activation_summary(biases, mean=True)  

        if activation == 'relu':
            x = tf.nn.relu(x, name='relu')
        elif activation == 'elu':
            x = tf.nn.elu(x, name='elu')
        elif activation == 'selu':
            x = tf.nn.selu(x, name='selu')

        if norm == True:
            x = tf.layers.batch_normalization(x, training=training, name='batch_norm')
        
        if debug:
            x = tf.Print(x, [tf.shape(x), tf.reduce_min(x), tf.reduce_max(x)],
                         message='Shape, min, max of %s' % (name),
                         summarize=4, first_n=1)

        _activation_summary(filters, range=True)
        _activation_summary(x,range=True)
        _mean_feature_summary(x)
        return x


def deconv_layer(bottom, f_shape, stride, name='deconv', debug=False, norm=True, training=False):
    '''
    bottom: 4D tensor; shape: [batch_size, in_height, in_width, in_channels], NHWC
    f_shape: [f_size, f_size, out_channels, in_channels]
        output_shape =[in_channels, stride*in_height, stride*in_width, out_channels]
    stride: int; strides = [1, stride, stride, 1]
    '''
    with tf.variable_scope(name):
        in_shape = tf.shape(bottom)
        # fan_in = f_shape[0] * f_shape[1] * f_shape[2]
        # stddev = tf.sqrt(2.0 / fan_in)
        filters = tf.get_variable('filters',
                                  shape=f_shape,
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=f_shape[2],  # out_channels
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        out_height = in_shape[1] * stride
        out_width = in_shape[2] * stride
        output_shape = tf.concat(
            [[in_shape[0]], [out_height], [out_width], [f_shape[2]]], 0, name='output_shape')
        strides = [1, stride, stride, 1]
        deconv = tf.nn.conv2d_transpose(
            bottom, filters, output_shape, strides, padding='SAME')
        deconv = tf.nn.bias_add(deconv, biases)

        if norm == True:    
            deconv = tf.layers.batch_normalization(deconv, training=training, name='batch_norm')

        if debug:
            deconv = tf.Print(deconv, [tf.shape(deconv), tf.reduce_max(deconv), tf.reduce_min(deconv)],
                         message='Shape, min, max of %s' % (name),
                         summarize=4, first_n=1)
            output_shape = tf.Print(output_shape, [output_shape],
                         message='output_shape:',
                         summarize=4, first_n=1)
            strides = tf.Print(strides, [strides],
                         message='strides:',
                         summarize=4, first_n=1)

        _activation_summary(filters, range=True)
        _activation_summary(biases, mean=True)
        _activation_summary(deconv, range=True)
        # sub-sample by 2
        deconv4summ = avg_pool(deconv, 'downsample', debug=False)
        _mean_feature_summary(deconv4summ)
        return deconv

# constant operator
def logsumexp_normalize(bottom, name='logsumexp_norm', debug=False):
    '''logsumexp normalize in image'''
    with tf.variable_scope(name) as scope:
        shape = tf.shape(bottom)
        line_shape = tf.stack([shape[0], -1])
        bottom = tf.reshape(bottom, line_shape) # reshape to line_shape
        line_max = tf.reduce_max(bottom, -1)
        line_max_ed = tf.expand_dims(line_max, axis=-1) 
        line_max_tl = tf.tile(line_max_ed, tf.concat([[1],[shape[1]*shape[2]*shape[3]]],0))
        lse = line_max + tf.reduce_logsumexp(bottom-line_max_tl, axis=-1) # line_shape
        lse_ed = tf.expand_dims(lse, axis=-1) 
        lse_tl = tf.tile(lse_ed, tf.concat([[1],[shape[1]*shape[2]*shape[3]]],0))
        bottom -= lse_ed
        bottom = tf.reshape(bottom, shape)
        if debug:
            bottom = tf.Print(bottom, [tf.shape(bottom), tf.reduce_max(bottom), tf.reduce_min(bottom)],\
                         message='Shape, min, max of %s' % (name),\
                         summarize=4, first_n=1)
        return bottom

def softmax(bottom, maxnorm=True, lineshape=False, summary=False, name='softmax'):
    '''softmax in image'''
    with tf.variable_scope(name) as scope:
        shape = tf.shape(bottom)
        line_shape = tf.stack([shape[0], -1])
        line_bottom = tf.reshape(bottom, line_shape)
        if maxnorm:
            line_max = tf.reduce_max(line_bottom, axis=-1)
            line_max_ed = tf.expand_dims(line_max, axis=-1) 
            line_max_tl = tf.tile(line_max_ed, tf.concat([[1],[shape[1]*shape[2]*shape[3]]],0))
            softmax = tf.nn.softmax(line_bottom - line_max_tl)
        else:
            softmax = tf.nn.softmax(line_bottom)
        if not lineshape:
            softmax = tf.reshape(softmax, shape)
        if summary:
            if not lineshape:
                _mean_feature_summary(softmax)
            if lineshape:
                _mean_feature_summary(tf.reshape(softmax, shape))
        return softmax


def gauss_layer(bottom, h_filter, w_filter, pad, name='gauss', debug=False, maxminnorm=False):
    '''
    h_filter: [size,1,1,1]
    w_filter: [1,size,1,1]
    pad: int; int(filter_width/2)
    '''
    with tf.variable_scope(name) as scope:
        paddings = tf.constant([[0,0], [pad, pad], [pad, pad], [0,0]])
        gauss1d = tf.nn.conv2d(tf.pad(bottom, paddings,"SYMMETRIC"), h_filter, strides=[
                               1, 1, 1, 1], padding='VALID')
        gauss2d = tf.nn.conv2d(gauss1d, w_filter, strides=[
                               1, 1, 1, 1], padding='VALID')
        if debug:
            gauss2d = tf.Print(gauss2d, [tf.shape(gauss2d), tf.reduce_max(gauss2d), tf.reduce_min(gauss2d)],
                                message='Shape, min, max of %s' % (name),
                                summarize=4, first_n=1)

        _activation_summary(gauss2d, range=True)
        _mean_feature_summary(gauss2d)
        return gauss2d


def filter_pool(w_filter, factor=2, name='filter_pool'):
    '''
    w_filter: numpy array, [1,width,1,1]
        tf.nn.avg_pool can not pool h_filter with shape (width,1,1,1)
    '''
    w_filter = tf.constant(w_filter, tf.float32)
    with tf.variable_scope(name) as scope:
        w_filter = tf.nn.avg_pool(w_filter, ksize=[1, factor,1,1], strides=[
                                  1,factor,1,1], padding='SAME')
        h_filter = tf.transpose(w_filter, [1,0,2,3])
    return h_filter, w_filter


def maxmin_norm(bottom, name='maxmin_norm', debug=False, domain=1, shift=None,summary=False):
    '''maxmin normalize in H and W dim
    Arg:
        bottom: 4D tensor;
        domain: int; output value domain
        shift: str; 
            None(LeftCenter):  the left point of output is zero point;
            ZeroCenter: the center of output is zero point;
            NoShift: the center of output stick on the original place
            
    '''
    with tf.variable_scope(name) as scope:
        bottom = tf.cast(bottom, tf.float32)
        shape = tf.shape(bottom)
        line_shape = tf.concat([[shape[0]], [shape[1]*shape[2]], [shape[3]]],0) # only H and W dims
        bottom = tf.reshape(bottom, line_shape) # reshape to line_shape
        line_mean = tf.reduce_mean(bottom, axis=1, keepdims=True)
        #1 shift
        line_min = tf.reduce_min(bottom, axis=1, keepdims=True)
        line_min_tl = tf.tile(line_min, tf.concat([[1],[shape[1]*shape[2]],[1]],0))
        shift_norm = tf.add(bottom, -line_min_tl)
        #2 Scale
        line_max = tf.reduce_max(shift_norm, axis=1, keepdims=True) / domain
        line_max_tl = tf.tile(line_max, tf.concat([[1],[shape[1]*shape[2]],[1]],0))
        norm = shift_norm/line_max_tl
        #3 center shift
        if shift == 'ZeroCenter':
            domain_max = tf.ones(line_shape, dtype=tf.float32) * domain / 2
            norm -= domain_max
        elif shift == 'NoShift':
            line_mid = line_max - line_min
            norm += line_mid
        elif shift == 'MeanCenter':
            new_line_mean = tf.reduce_mean(norm, axis=1, keepdims=True)
            norm += line_mean - new_line_mean
        #else zero left
        #4 reshape
        norm = tf.reshape(norm, shape)

        if debug:
            norm = tf.Print(norm, [tf.shape(norm), tf.reduce_max(norm), tf.reduce_min(norm)],
                         message='Shape, min, max of %s' % (name),
                         summarize=4, first_n=1)
        if summary:
            _mean_feature_summary(norm)
        return norm

def zscore_norm(bottom, name, debug=False, stddev=1, shift='',summary=False):
    '''maxmin normalize in H and W dim
    Arg:
        bottom: 4D tensor;
        stddev: int; output value standard deviation
        shift: str; 
            (Default LeftCenter):  the left point of output is zero point;
            ZeroCenter: the center of output is zero point;
            NoShift: the center of output stick on the original place    
    '''
    # use tf.nn.moments() to find mean and variance
    pass


def _activation_summary(x, sparsity=False, mean=False, range=False):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
        x: Tensor
    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    if sparsity:
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    if mean:
        tf.summary.scalar(tensor_name + '/mean', tf.reduce_mean(x))
    if range:
        tf.summary.scalar(tensor_name + '/min', tf.reduce_min(x))
        tf.summary.scalar(tensor_name + '/max', tf.reduce_max(x))

def _mean_feature_summary(features):
    tensor_name = features.op.name
    feature = tf.reduce_mean(features, axis=-1)
    feature = tf.expand_dims(feature, axis=-1)
    feature = maxmin_norm(feature, 'mean_feature_maxmin_norm',debug=False, domain=1,shift='ZeroLeft',summary=False)
    # return feature 
    tf.summary.image(tensor_name, feature)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from utils.io import imread

    def deconv_test():
        # (480, 640, 3)
        src = imread('test_data/images/COCO_train2014_000000123273.jpg')
        src_c = np.expand_dims(src, 0)  # (1, 480, 640, 3)
        print(src_c.shape)
        src_c = tf.cast(tf.constant(src_c), tf.float32)
        print(src_c)
        # test parameter here
        deconv = deconv_layer(src_c, [4, 4, 1, 3], stride=2, name='upscale')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(deconv)
            print(out)
            plt.figure()
            plt.subplot(121)
            plt.imshow(src)
            plt.subplot(122)
            plt.imshow(np.squeeze(out))
            plt.show()

    t1 = np.array([[[[ 3, 2, 3],
                     [ 5, 3, 4]], 
                    [[ 7, 4, 5],
                     [ 9, 5, 6]],
                    [[11, 6, 7],
                     [13, 7, 8]]]], dtype=float)
    t2 = np.array([[[[ 3, 2, 3],
                     [ 5, 3, 4]],
                    [[ 7, 4, 5],
                     [ 9, 5, 6]],
                    [[11, 6, 7],
                     [13, 7, 8]]],
                   [[[ 5, 2, 3],
                     [ 6, 3, 4]],
                    [[ 7, 0, 5],
                     [ 0, 5, 6]],
                    [[14, 6, 7],
                     [ 8, 7,10]]]], dtype=float)
    t3 = np.array([[[[ 0.3, 0.2, 0.3],
                     [ 0.5, 0.3, 0.4]],
                    [[ 0.7, 0.4, 0.5],
                     [ 0.9, 0.5, 0.6]],
                    [[ 1.1, 0.6, 0.7],
                     [ 1.3, 0.7, 0.8]]],
                   [[[ 0.5, 0.2, 0.3],
                     [ 0.6, 0.3, 0.4]],
                    [[ 0.7, 0.0, 0.5],
                     [ 0.0, 0.5, 0.6]],
                    [[ 1.4, 0.6, 0.7],
                     [ 0.8, 0.7, 1.0]]]], dtype=float)
    
    t1 = np.array([[[[ 3, 2, 3],
                     [ 5, 3, 4]], 
                    [[ 7, 4, 5],
                     [ 9, 5, 6]],
                    [[11, 6, 7],
                     [13, 7, 8]]]], dtype=float)
    
    def shape_test():
        tt = tf.constant(t3, tf.float32)
        shape = tf.shape(tt)
        shape_line = tf.concat([[shape[0]*shape[1]*shape[2]]],0) # all dims
        shape_len = shape_line[0]
        with tf.Session() as sess:
            out=sess.run([shape,shape_line, shape_len])
            print(out)
    shape_test()


    def softmax_test():
        tt = tf.constant(t3, tf.float32)
        t_sm = softmax(tt, maxnorm=True, lineshape=False, summary=False, name='softmax')
        with tf.Session() as sess:
            out = sess.run(t_sm)
            plt.imshow(out[0,:])
            plt.show()
            print(np.sum(out))
    softmax_test()

    def lse_test():
        tt = tf.constant(t1, tf.float32)
        t_lse = logsumexp_normalize(tt, 'lse', True)
        with tf.Session() as sess:
            out = sess.run(t_lse)
            print(out)
            # print(out[0],out[1],out[2], sep='\n')
    # lse_test()

    def lse_softmax_test():
        tt = tf.constant(t3, tf.float32)
        t_lse = logsumexp_normalize(tt, 'lse', False)
        t_sm = softmax(t_lse, maxnorm=True, lineshape=True, summary=False, name='softmax')
        with tf.Session() as sess:
            out = sess.run(t_sm)
            print(out)
            print(np.sum(out))
    # lse_softmax_test() # the same as softmax_test()

    def maxmin_norm_test():
        with tf.Session() as sess:
            out = sess.run(maxmin_norm(t1,'norm',domain=1, shift='MeanCenter',debug=True)) 
            print(out,out.shape, sep='\n')
    # maxmin_norm_test()

    def mean_feature_summary_test():
        with tf.Session() as sess:
            out = sess.run(_mean_feature_summary(t2))
            print(out)   
    # mean_feature_summary_test()

    def zscore_norm_test():
        with tf.Session() as sess:
            out = sess.run(_mean_feature_summary(t2))
            print(out)
    # zscore_norm_test()

    def moments_test():
        tt=tf.constant(t3, tf.float32)
        a_mean, a_var = tf.nn.moments(tt, [0])
        with tf.Session() as sess:
            out = sess.run([a_mean, a_var])
            print(out[0])
