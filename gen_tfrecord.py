# coding: utf-8

import sys
import os

# In[1]:

from salicon.salicon import SALICON
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from scipy import ndimage
from tqdm import tqdm

from utils.io import imread
from utils.io import get_image_list
from utils.io import get_cofile_list
import config

# # In[2]:
# def decode_salicon(data_dir, json_postfix, save=True):
#     '''
#     data_dir, json_postfix: str
#     '''
    
#     save_dir = os.path.join(data_dir, 'fix_maps')

#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)

#     annFile = '%s/annotations/fixations_%s.json'%(data_dir,json_postfix)

#     # initialize COCO api for instance annotations
#     salicon=SALICON(annFile)

#     # get all images 
#     imgIds = salicon.getImgIds()
#     imgs = salicon.loadImgs(imgIds)
#     # img = salicon.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
#     # '%s/images/%s'%(dataDir,img['file_name'])


#     # save fix_map
#     for img in tqdm(imgs):
#         annIds = salicon.getAnnIds(imgIds=img['id'])
#         anns = salicon.loadAnns(annIds)
#         fix_map = salicon.buildFixMap(anns, doBlur=False)
#         fix_map = np.int8(fix_map)
#         filename, ext = os.path.splitext(img['file_name'])
#         if save:
#             np.save('%s/%s.npy'%(save_dir, filename),fix_map)

# # test fix_maps
# def read_fix_map(fix_map_dir):
#     assert(os.path.exists(fix_map_dir))
#     files = os.listdir(fix_map_dir)
#     for file in tqdm(files):
#         _,ext = os.path.splitext(file)
#         if ext.lower() in ['.npy']:
#             file_path = os.path.join(fix_map_dir, file)
#             fix_map = np.load(file_path)
#             plt.imshow(fix_map)
#             plt.show()

# # test fix_maps
# def read_train_data(data_dir):
#     img_dir = data_dir+'/images'
#     fix_dir = data_dir+'/fix_maps'
#     assert(os.path.exists(img_dir))
#     assert(os.path.exists(fix_dir))
#     img_list = get_image_list(img_dir)
#     fix_list, co_img_list = get_cofile_list(fix_dir, img_list, ext='npy') 
    
#     for i in tqdm(np.arange(len(fix_list))):
#         img = imread(co_img_list[i])
#         fix_map = np.load(fix_list[i])
#         plt.figure()
#         plt.subplot(121)
#         plt.imshow(img) 
#         plt.subplot(122)
#         plt.imshow(fix_map)
#         plt.show()

# # In[3]:
# def convert_to_tfrecord(images, labels, save_dir, name):
#     '''convert all images and labels to one tfrecord file. 
#     Args: 
#         images: list of image directories, string type 
#         labels: list of saliency map directories, string type 
#         save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/' 
#         name: the name of tfrecord file, string type, e.g.: 'train' 
#     Return: no return 
#     Note: converting needs some time, be patient... 
#     '''

#     filename = os.path.join(save_dir, name + '.tfrecords')
#     n_samples = len(labels)

#     # wait some time here, transforming need some time based on the size of your data.
#     writer = tf.python_io.TFRecordWriter(filename)
#     print('\nTransform start......')
#     for i in tqdm(np.arange(0, n_samples)):
#         try:
#             image = imread(images[i])  # type(image) must be array!
#             label = np.load(labels[i])
#             if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
#                 print('image: "%s" shape %s; label: "%s" shape %s.\nheight and width of the image and the label need to be equal' % (
#                     images[i],image.shape, labels[i], label.shape))
#                 continue
#             # plt.subplot(121)
#             # plt.imshow(image)
#             # plt.subplot(122)
#             # plt.imshow(label)
#             # plt.show()
#             example = tf.train.Example(features=tf.train.Features(feature={
#                             'image': _bytes_feature(image.tostring()),
#                             'fix_map': _bytes_feature(label.tostring())
#                             }))
#             writer.write(example.SerializeToString()) 
#         except IOError as e:
#             print('Could not read %s or %s'%(images[i],labels[i]))
#             print('Warning: %s' %e)
#     writer.close()
#     print('Transform done!')

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def _int64_feature(value):
#     return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

# def gen_tfrecords_from_file(data_dir, data_type, save_name):
#     '''
#     data_dir: str; SALICON dataset directory
#     data_type: int; 0 for train data, 1 for validation data
#     save_name: str
#     '''
#     if data_type == 'train':
#         prefix='COCO_train'
#     elif data_type == 'val':
#         prefix='COCO_val'
#     else:
#         return
#     img_dir = os.path.join(data_dir, 'images')
#     img_list = get_image_list(img_dir, prefix)
#     fix_dir = os.path.join(data_dir, 'fix_maps')
#     fix_list, co_img_list = get_cofile_list(fix_dir, img_list, ext='npy')
    
#     convert_to_tfrecord(co_img_list, fix_list, data_dir, save_name)

def gen_tfrecords_fix_map(data_dir, json_postfix, save_dir, save_name, require_num=None):
    '''
    data_dir, json_postfix: str
    '''
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, save_name + '.tfrecords')

    annFile = '%s/annotations/fixations_%s.json'%(data_dir,json_postfix)

    # initialize COCO api for instance annotations
    salicon=SALICON(annFile)

    # get all image info
    imgIds = salicon.getImgIds()
    imgs = salicon.loadImgs(imgIds)

    writer = tf.python_io.TFRecordWriter(save_path)
    print('\nTransform start......')
    trans_num = 0
    if require_num == None:
        require_num = len(imgs)
    for trans_num, imginfo in enumerate(tqdm(imgs)):
        if trans_num >= require_num:
            trans_num -= 1
            break
        img_path = '%s/images/%s' % (data_dir, imginfo['file_name'])
        img = imread(img_path)
        if img is None:
            print('Fail to read. Error image path:%s'%img_path)
            continue
        annIds = salicon.getAnnIds(imgIds=imginfo['id'])
        anns = salicon.loadAnns(annIds)
        ## salicon.buildFixMap
        if len(anns) == 0:
            return 0
        #build saliency map based on annotations
        #check whether all annotations are for the same image
        assert(len(set([ann['image_id'] for ann in anns])) == 1)
        image_id = list(set([ann['image_id'] for ann in anns]))[0]
        all_fixations = [ann['fixations'] for ann in anns] # all_fixations from several workers
        
        merged_fixations = [item for sublist in all_fixations for item in sublist] #merge
        # print(type(merged_fixations), len(merged_fixations), imginfo['height'])
        fix_map = np.zeros((imginfo['height'],imginfo['width']))
        #NOTE: the coordinate in fixations is (y,x) !!! Notice the order of y and x
        for y,x in merged_fixations: 
            # fix_map[y-1][x-1] = 1
            fix_map[y-1][x-1] += 1
        fix_map = np.uint8(fix_map)

        try:
            example = tf.train.Example(features=tf.train.Features(feature={
                                'image': _bytes_feature(img.tostring()),
                                'fix_map': _bytes_feature(fix_map.tostring())
                                }))
            writer.write(example.SerializeToString()) 
        except IOError as e:
            print('Warning: %s' %e)
    writer.close()
    trans_num += 1
    if trans_num < require_num:
        print('%d samples transform done! Require num is %d, Maximum number is %d'%(trans_num,require_num,trans_num))
    else:
        print('%d samples transform done!'%require_num)


# In[image + merge_fixations + salmap]:
# def gen_tfrecords(data_dir, train_or_val, json_postfix, save_dir, save_name):

#     '''
#     data_dir, json_postfix: str
#     '''
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#     save_path = os.path.join(save_dir, save_name + '.tfrecords')


#     annFile = '%s/annotations/fixations_%s.json'%(data_dir,json_postfix)

#     # initialize COCO api for instance annotations
#     salicon=SALICON(annFile)

#     # get all image info
#     imgIds = salicon.getImgIds()
#     imgs = salicon.loadImgs(imgIds)
#     # imginfo = salicon.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
#     # '%s/images/%s'%(data_dir,imginfo['file_name'])
#     # imginfo = imgs[0]
#     # print(imginfo)
#     ''' imginfo example
#     {'license': 4, 
#     'url': 'http://farm7.staticflickr.com/6120/6330820978_d44a63af02_z.jpg', 
#     'file_name': 'COCO_train2014_000000123855.jpg', 
#     'height': 480, 
#     'width': 640, 
#     'date_captured': '2013-11-14 16:50:09', 
#     'id': 123855}
#     '''

#     writer = tf.python_io.TFRecordWriter(save_path)
#     print('\nTransform start......')
#     # save sal_map
#     for imginfo in tqdm(imgs):
#         img_path = '%s/images/%s' % (data_dir, imginfo['file_name'])
#         img = imread(img_path)
#         if img is None:
#             print('Fail to read. Error image path:%s'%img_path)
#             continue
#         annIds = salicon.getAnnIds(imgIds=imginfo['id'])
#         anns = salicon.loadAnns(annIds)
#         ## salicon.buildFixMap
#         if len(anns) == 0:
#             return 0
#         #build saliency map based on annotations
#         #check whether all annotations are for the same image
#         assert(len(set([ann['image_id'] for ann in anns])) == 1)
#         image_id = list(set([ann['image_id'] for ann in anns]))[0]
#         all_fixations = [ann['fixations'] for ann in anns] # all_fixations from several workers
        
#         merged_fixations = [item for sublist in all_fixations for item in sublist] #merge
#         # print(type(merged_fixations), len(merged_fixations), imginfo['height'])
#         sal_map = np.zeros((imginfo['height'],imginfo['width']))
#         #NOTE: the coordinate in fixations is (y,x) !!! Notice the order of y and x
#         for y,x in merged_fixations: 
#             sal_map[y-1][x-1] = 1
#         # sal_map = ndimage.filters.gaussian_filter(sal_map, sigma=19)
#         # sal_map -= np.min(sal_map)
#         # sal_map /= np.max(sal_map)
#         sal_map = np.float32(sal_map)
#         # print(sal_map.shape, np.min(sal_map), np.max(sal_map))

#         merged_fixations = np.array(merged_fixations, dtype=np.int32)
#         # padding to the same length (12000)
#         fixations = np.zeros([12000,2], dtype=np.int32)
#         fixations[:merged_fixations.shape[0],:merged_fixations.shape[1]] = merged_fixations
#         fixations -= 1

#         # print(merged_fixations.shape, np.min(merged_fixations))
#         try:
#             example = tf.train.Example(features=tf.train.Features(feature={
#                                 'image': _bytes_feature(img.tostring()),
#                                 'fixations': _bytes_feature(fixations.tostring()),
#                                 'sal_map': _bytes_feature(sal_map.tostring())
#                                 }))
#             writer.write(example.SerializeToString()) 
#         except IOError as e:
#             print('Warning: %s' %e)
#     writer.close()
#     print('Transform done!')

# gen_tfrecords(config.data_dir, 'train', config.json_postfix[0], config.data_dir, 'train_data')

# In[4] test:
def read_tfrecords(tfrecords_file):
    print(tfrecords_file)
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep size down
    filename_queue = tf.train.string_input_producer([tfrecords_file], num_epochs=1)
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic, 即所做的操作不会立即执行
    reader = tf.TFRecordReader()
    keys, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
                     'image':  tf.FixedLenFeature([], tf.string),
                     'fix_map':  tf.FixedLenFeature([], tf.string),
                    })
    # decode
    image = tf.decode_raw(features['image'], tf.uint8)
    fix_map = tf.decode_raw(features['fix_map'], tf.uint8)
    # input
    image = tf.reshape(image,[480, 640, 3])
    fix_map = tf.reshape(fix_map,[480, 640])

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        tf.train.start_queue_runners()
        for i in range(10):
            img, sal = sess.run([image, fix_map])
            print(img.shape, np.min(img), np.max(img))
            print(sal.shape, np.min(sal), np.max(sal))
            plt.figure()
            plt.subplot(121)
            plt.imshow(img) 
            plt.subplot(122)
            plt.imshow(sal)
            plt.show()

# def read_tfrecords_fix(tfrecords_file):
#     print(tfrecords_file)
#     # first construct a queue containing a list of filenames.
#     # this lets a user split up there dataset in multiple files to keep size down
#     filename_queue = tf.train.string_input_producer([tfrecords_file], num_epochs=1)
#     # Unlike the TFRecordWriter, the TFRecordReader is symbolic, 即所做的操作不会立即执行
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)

#     features = tf.parse_single_example(serialized_example, features={
#                      'image':  tf.FixedLenFeature([], tf.string),
#                      'fixations': tf.FixedLenFeature([], tf.string),
#                      'sal_map':  tf.FixedLenFeature([], tf.string),
#                     })
#     # decode
#     image = tf.decode_raw(features['image'], tf.uint8)
#     sal_map = tf.decode_raw(features['sal_map'], tf.float32)
#     fixations = tf.decode_raw(features['fixations'], tf.int32)
#     # input
#     image = tf.reshape(image,[480, 640, 3])
#     sal_map = tf.reshape(sal_map,[480, 640])
#     fixations = tf.reshape(fixations,[12000, 2])

#     init_op = tf.group(tf.global_variables_initializer(),
#                        tf.local_variables_initializer())
#     with tf.Session() as sess:
#         sess.run(init_op)
#         tf.train.start_queue_runners()
#         for i in range(2):
#             img, sal, fix = sess.run([image, sal_map, fixations])
#             print(fix.shape)
#             print(img.shape, np.min(img), np.max(img))
#             print(sal.shape, np.min(sal), np.max(sal))

            
if __name__ == '__main__':

    # # ## fix_map tfrecords
    # gen_tfrecords_fix_map(config.data_dir, config.json_postfix[0], config.data_dir, 'train_data')
    # read_tfrecords('test_data/train_data.tfrecords')  # test

    ## fix_map tfrecords
    gen_tfrecords_fix_map(config.data_dir, config.json_postfix[1], config.data_dir, 'val_data',400)
