# coding=UTF-8
# The aim of this project is to use TensorFlow to process our own data.
import os
import math

import tensorflow as tf

from salicon.salicon import SALICON


class SaliconInputs:
    def __init__(self, data_dir, json_postfix):
        ''' Reads and parses samples from SALICON data files.
        Gets image ids, shuffles it and then splits them to train set and validation set
        Args:
            data_dir: str; data directory
            json_postfix: str; postfix of json data
        '''
        self.data_dir = data_dir
        self.json_postfix = json_postfix
        annFile = '%s/annotations/fixations_%s.json' % (data_dir, json_postfix)
        self.salicon = SALICON(annFile)
    
    def get_fix_map(self, img_id):
        '''get fixation map by img_id'''
        annIds = self.salicon.getAnnIds(imgIds=img_id)
        anns = self.salicon.loadAnns(annIds)
        sal_map = self.salicon.buildFixMap(anns)
        return sal_map

    def get_img_batch(self, batch_size, epoch):
        ''' Parses samples
        Args:
            ~
        Returns:
            one_batch: tuple, dtype=(ndarray of 'image', ndarray of 'img_id')
                image: 3-dimensional ndarray
                img_id: numpy.int32
        '''
        img_ids = self.salicon.getImgIds()   # list of image ids
        img_info_list = self.salicon.loadImgs(img_ids)  # list of image info
        img_pathes = ['%s/images/%s' %
                      (self.data_dir, img['file_name']) for img in img_info_list]

        # make graph
        # convert to tensor
        img_pathes = tf.constant(img_pathes)
        img_ids = tf.constant(img_ids)

        # map接收一个函数，Dataset中的每个元素都会被当作这个函数的输入，
        # 并将函数返回值作为新的Dataset，如我们可以对dataset中每个元素的值加1：
        # initial tensor_slices
        dataset = tf.data.Dataset.from_tensor_slices((img_pathes, img_ids))
        # element in dataset is (images, labels)
        dataset = dataset.map(self._parse_function)
        # shuffle, batch_size, epoch
        dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).repeat(epoch)

        # get data
        iterator = dataset.make_one_shot_iterator()
        one_batch = iterator.get_next()  # (images, img_ids)
        return one_batch

    @staticmethod
    def _parse_function(img_path, img_id, standardize=False, dst_h=None, dst_w=None):
        '''
        read image path and id in tfGraph
        Arg:
            img_path: tf.string
            img_id: tf.int32
            standardize: boolean; 
            dst_h: int; height of dst images
            dst_w: int; width of dst images 
        '''
        image_string = tf.read_file(img_path)
        image = tf.image.decode_image(image_string, channels=3)
        # # tf data augmentation should code here
        # # preprocess can code here
        if standardize:
            image = tf.image.per_image_standardization(image)
        if dst_h and dst_w:
            image = tf.image.resize_image_with_crop_or_pad(image, dst_h, dst_w)
        return image, img_id
        # return image, img_id, img_path

# tfrecord input
def read_and_decode(tfrecords_file, 
                    epoch, 
                    batch_size, 
                    capacity=32,
                    num_threads=1,
                    min_after_dequeue=10, 
                    shuffle_batch=True):
        '''read and decode tfrecord file, generate (image, label) batches 
        Args: 
            tfrecords_file: str; the directory of tfrecord file 
            epoch: int; number of dataset copies
            batch_size: int; number of images in each batch 
            capacity: int; maximum number in queue
            num_threads: int; suggestion: num_threads>1
            min_after_dequeue:int; Minimum number elements in the queue after a  
                dequeue, used to ensure a level of mixing of elements.
            shuffle_batch: boolen
        Returns: 
            image: 4D tensor - [batch_size, width, height, channel]
            label: 3D tensor - [batch_size, width, height]
        '''
        # make an input queue from the tfrecord file
        filename_queue = tf.train.string_input_producer([tfrecords_file],
                                                        num_epochs=epoch)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={
                        'image':  tf.FixedLenFeature([], tf.string),
                        'fix_map':  tf.FixedLenFeature([], tf.string),
                        })
        # decode
        image_raw = tf.decode_raw(features['image'], tf.uint8)
        fix_map_raw = tf.decode_raw(features['fix_map'], tf.uint8)
        # input
        image = tf.reshape(image_raw, [480, 640, 3])
        fix_map = tf.reshape(fix_map_raw, [480, 640])

        ##########################################################
        # you can put data augmentation here
        ##########################################################

        if shuffle_batch:
            img_batch, fix_batch = tf.train.shuffle_batch(
                                                [image, fix_map],
                                                batch_size=batch_size,
                                                capacity=capacity,
                                                num_threads=num_threads,
                                                min_after_dequeue=min_after_dequeue)
        else:
            img_batch, fix_batch= tf.train.batch(
                                                [image, fix_map],
                                                batch_size= batch_size,
                                                num_threads= num_threads, 
                                                capacity = capacity)
        return img_batch, fix_batch


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage
    import config

    def test_get_img_batch1():
        inputs = SaliconInputs(config.data_dir, config.json_postfix[0])
        one_batch = inputs.get_img_batch(config.BATCH_SIZE, config.EPOCH)
        with tf.Session() as sess:
            # batch_i=0
            try:
                iter_num = 1
                while True and iter_num > 0:
                    iter_num -= 1
                    # (ndarray list of image, ndarray list of img_id)
                    images, img_ids = sess.run(one_batch)
                    # batch_i += 1
                    # print('Batch %d'%batch_i)
                    for i in range(config.BATCH_SIZE):     # batch_size；len(batch[1]）
                        img = images[i]             # ndarray
                        img_id = img_ids[i].item()  # int
                        # print(type(batch[1][i]))
                        annIds = inputs.salicon.getAnnIds(imgIds=img_id)
                        anns = inputs.salicon.loadAnns(annIds)
                        sal_map = inputs.salicon.buildFixMap(anns)
                        # print(sal_map.shape)
                        plt.figure()
                        plt.subplot(121)
                        plt.imshow(img)
                        plt.subplot(122)
                        plt.imshow(sal_map)
                        plt.show()
            except tf.errors.OutOfRangeError:
                print("end!")
    
    # test_get_img_batch1()

    def test_get_img_batch2():
        inputs = SaliconInputs(config.data_dir, config.json_postfix[0])
        one_batch = inputs.get_img_batch(config.BATCH_SIZE, config.EPOCH)
        with tf.Session() as sess:
            try:
                # (ndarray list of image, ndarray list of img_id)
                images, img_ids = sess.run(one_batch)
                print(type(images),type(img_ids))
                print(images.shape, img_ids.shape)
                sal_maps = np.array([inputs.get_fix_map(
                    img_id) for img_id in img_ids])
                print(sal_maps.shape)

                images, img_ids = sess.run(one_batch)
                print(type(images),type(img_ids))
                print(images.shape, img_ids.shape)
                
                images, img_ids = sess.run(one_batch)
                print(type(images),type(img_ids))
                print(images.shape, img_ids.shape)
            except tf.errors.OutOfRangeError:
                print("end!")
                
        with tf.Session() as sess:
            try:        
                images, img_ids = sess.run(one_batch)
                print(type(images),type(img_ids))
                print(images.shape, img_ids.shape)
                
                images, img_ids = sess.run(one_batch)
                print(type(images),type(img_ids))
                print(images.shape, img_ids.shape)
            except tf.errors.OutOfRangeError:
                print("end!")
        print('done!')
    
    # test_get_img_batch2()
    
    def plot_samples(imgs, labs):
            print("shapes:",imgs.shape, labs.shape)
            batch_num = imgs.shape[0]
            try:
                for i in range(batch_num):
                    # # fix to salmap
                    # fxmp = np.zeros((480, 640))
                    # for y,x in fixs[i]:
                    #     if y<0 or x<0:
                    #         continue
                    #     fxmp[y-1][x-1] = 1
                    # if 0:
                    #     fxmp = ndimage.filters.gaussian_filter(fxmp, sigma=19)
                    #     fxmp -= np.min(fxmp)
                    #     fxmp /= np.max(fxmp)
                    # fxmp = np.float32(fxmp)

                    plt.figure()
                    plt.subplot(121)
                    plt.imshow(imgs[i])
                    plt.subplot(122)
                    plt.imshow(labs[i])
                    plt.show()
            except IndexError:
                print('Index %s is out of the range of images or labels!'%i)

    def test_read_and_decode():
       
        tfrecords_file = os.path.join(config.data_dir, 'train_data.tfrecords')
        # tfrecords_file = os.path.join(config.data_dir, 'val_data.tfrecords')

        img_batch, fix_batch = read_and_decode(tfrecords_file, 
                                               config.EPOCH, 
                                               config.BATCH_SIZE, 
                                               config.CAPACITY,
                                               config.NUM_THREADS,
                                               config.MIN_AFTER_DEQUEUE,
                                               shuffle_batch=True)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        x = tf.placeholder(tf.uint8,[config.BATCH_SIZE,480,640], name='x')
        tx = tf.cast(x, tf.bool)
        bool_x = tf.boolean_mask(x, tx)
       
        with tf.Session(config=config.tf_config) as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                # for step in range(config.MAX_STEP):
                for step in range(1):
                    print('Step: ',step)
                    if coord.should_stop():
                        break
                    imgs, fixs = sess.run([img_batch, fix_batch])
                    print(imgs.dtype, fixs.dtype)
                    # plot_samples(imgs, fixs)
                    print(sess.run(bool_x, feed_dict={x:fixs}))
                    
            except tf.errors.OutOfRangeError:
                print('done!')
            finally:
                coord.request_stop()
            coord.join(threads)

    # test_read_and_decode()

    def max_fix_num_in_label():
        tfrecords_file = os.path.join(config.data_dir, 'train_data.tfrecords')
        # tfrecords_file = os.path.join(config.data_dir, 'val_data.tfrecords')

        img_batch, fix_batch = read_and_decode(tfrecords_file, 
                                               epoch=2, 
                                               batch_size=8, 
                                               capacity=10,
                                               num_threads=4,
                                               min_after_dequeue=1,
                                               shuffle_batch=False)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        x = tf.placeholder(tf.uint8,[8,480,640], name='x')
        max_num = tf.reduce_max(x)

        with tf.Session(config=config.tf_config) as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            max_fix_num = 0
            try:
                for step in range(2000):
                    print('Step: ',step)
                    if coord.should_stop():
                        break
                    in_fixs = sess.run(fix_batch)
                    batch_max_num = sess.run(max_num, feed_dict={x:in_fixs})
                    if max_fix_num < batch_max_num:
                        max_fix_num = batch_max_num
                    print('max_fix_num:%d, batch_max_num:%d'%(max_fix_num,batch_max_num))
                print('max_fix_num:',max_fix_num)
            except tf.errors.OutOfRangeError:
                print('done!')
            finally:
                coord.request_stop()
            coord.join(threads)

    max_fix_num_in_label() #max_fix_num:9
