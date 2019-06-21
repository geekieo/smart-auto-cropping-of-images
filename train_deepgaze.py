import tensorflow as tf
import numpy as np
import os

from inputs import read_and_decode
import gaze_vgg
import config
from gaze_vgg_layers import avg_pool, softmax, _mean_feature_summary,logsumexp_normalize, maxmin_norm
# debug
from utils.process import channel_mean
import matplotlib.pyplot as plt

def pre_loss(images, preds, labels):
    """pre precess before calculate loss or info gain, and write images into summary
    Args:
        preds: 4D tensor, float32 - [batch_size, width, height, 1].
            Use gaze.out as preds.
        labels: 3D tensor, int32 - [batch_size, width, height].
            The ground truth of your data.
    Returns:
        preds: 4D tensor, float32 - [batch_size, width, height, 1].
    """
    with tf.name_scope('pre_loss'):
        images = avg_pool(images, 'images', debug=False)
        tf.summary.image('images', images)

        labels = tf.expand_dims(labels, -1)     # NHW to NHWC
        labels = tf.cast(labels, tf.float32)
        tf.summary.image('ori-labels', labels)

        preds = avg_pool(preds, 'preds', debug=False)
        tf.summary.image('preds', preds)

        lse = logsumexp_normalize(preds, 'salmap_logsumexp', debug=False)
        lse_softmax = softmax(preds, maxnorm=True, lineshape=False, summary=False, name='softmax')
        tf.summary.image('softmax', lse_softmax)
        return  labels

def calc_loss(preds, labels, debug=False):
    """Calculate the loss from the fixation map logits and the labels.
    Args:
    preds: 4D tensor, float32 - [batch_size, width, height, 1].
        Use gaze.out as preds.
    labels: 4D tensor, float32 - [batch_size, width, height, 1].
        The ground truth of your data.
    Returns:
    loss: Loss tensor of type float.
    """
    with tf.name_scope('Loss'):
        preds = tf.sigmoid(preds)
        labels_bool = tf.cast(labels, tf.bool)  # [[0,1],[0,3]] → [[False, True], [False, True]]
        
        # Find cross_entropy
        mask_preds = tf.boolean_mask(preds, mask=labels_bool) # [[6,7],[8,9]] + labels_bool → [7,9]
        mask_labels= tf.boolean_mask(labels,mask=labels_bool) # [[0,1],[0,3]] + labels_bool → [1,3]
        
        if debug:
            # # example:[2 480 640 1][0.476581037][0.505854249]
            # cross_entropy 要写在 tf.Print 后面，才会执行打印
            mask_preds = tf.Print(mask_preds, [tf.shape(mask_preds), tf.reduce_min(mask_preds), tf.reduce_max(mask_preds)],
                             message='\nShape of mask_preds, min, max', summarize=4, first_n=1)
            mask_labels = tf.Print(mask_labels, [tf.shape(mask_labels), tf.reduce_min(mask_labels), tf.reduce_max(mask_labels)],
                             message='\nShape of mask_labels, min, max', summarize=4, first_n=1)
        
        # cross_entropy = -tf.multiply(tf.log(mask_preds+1e-10), mask_labels)
        cross_entropy = -tf.multiply(tf.log(mask_preds), mask_labels)
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        # loss = tf.add_to_collection('losses', loss)
        # loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss

def training(loss, optimizer='adam'):
    with tf.name_scope('optimizer'):
        if optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=config.learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step=global_step)
        elif optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=0.0025, momentum=0.9)
            train_op = optimizer.minimize(loss)
    return train_op

def validate():
    tfrecords_file = os.path.join(config.data_dir, 'val_data.tfrecords')
    img_batch, fix_batch = read_and_decode(tra_data,
                                        epoch=1,
                                        batch_size=400,
                                        capacity=400,
                                        num_threads=config.NUM_THREADS,
                                        min_after_dequeue=config.MIN_AFTER_DEQUEUE,
                                        shuffle_batch=True)
    with tf.name_scope('validate'):
        val_images, val_fixs = sess.run([img_batch, fix_batch])
        feed_dict = {input_images: val_images, input_labels: val_fixs}
        val_loss = sess.run(tra_loss, feed_dict=feed_dict)
    return val_loss

def train():
    # input
    tra_data = os.path.join(config.data_dir, 'train_data.tfrecords')

    img_batch, fix_batch = read_and_decode(tra_data,
                                        config.EPOCH,
                                        config.BATCH_SIZE,
                                        config.CAPACITY,
                                        config.NUM_THREADS,
                                        config.MIN_AFTER_DEQUEUE,
                                        shuffle_batch=True)

    # feed_dict
    input_images = tf.placeholder(tf.uint8, name='input_images')  # dims = 4
    input_labels = tf.placeholder(tf.uint8, shape=[config.BATCH_SIZE,480,640], name='input_labels') 
    # graph
    gaze = gaze_vgg.GAZEVGG(use_centerbias=False, gauss_factor=1, pred_h=480, pred_w=640)
    gaze.build_deepgaze(input_images, input_labels, debug=config.debug, concat=False, readout_norm=True,
                        vgg_train=True, vgg_norm=False, training=True)
    print('Finished building Network.')

    label = pre_loss(gaze.rgb, gaze.gauss, gaze.label)
    tra_loss = calc_loss(gaze.gauss, label, debug=config.debug)
    tf.summary.scalar('loss', tra_loss)

    # tra_op = training(tra_loss, optimizer='momentum')
    tra_op = training(tra_loss, optimizer='adam') #set saver after adam. adam has dynamic learning rate

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    # Start the queue runners.
    coord = tf.train.Coordinator()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    def train_part(step, max_step, summary_step):
        if step % summary_step != 0  and step != max_step:
            _, train_loss = sess.run([tra_op, tra_loss], feed_dict=feed_dict)
        else:
            # summary and eval
            _, train_loss, summary_str = sess.run([tra_op, tra_loss, 
                summary_op ], feed_dict)
            print('Step %d, train loss = %f (summary)' %
                  (step, train_loss))
            train_writer.add_summary(summary_str, step)
        if step % 2000 == 0 or step == max_step:
            checkpoint_path = os.path.join(config.ckpt_dir, 'deepgaze.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
    

    def debug_readout_part(case=1):
        if case == 0:
            train_loss = sess.run([tra_loss], feed_dict=feed_dict)
            # print(train_loss[0].shape, np.min(train_loss[0]),np.max(train_loss[0]))
            print('train loss = %f' % train_loss[0])
        elif case == 1:
            out = sess.run([gaze.label,
                            gaze.features,
                            gaze.conv1,
                            gaze.conv2,
                            gaze.conv3,
                            gaze.conv4,
                            gaze.deconv,
                            gaze.gauss], feed_dict=feed_dict)
            out[0]=np.expand_dims(out[0], -1)
            plt.figure()
            for i, item in enumerate(out):
                print(str(step)+'_'+str(i), np.min(item), np.max(item))
                img = channel_mean(item)
                plt.subplot(2, 4, i+1)
                plt.imshow(img)
                plt.colorbar()
            plt.show()
        elif case == 2 :
            out = sess.run([gaze.rgb, gaze.gauss, gaze.label], feed_dict=feed_dict)
            print('rgb',out[0].dtype, out[0].shape)
            print('gauss',out[1].dtype, out[1].shape)
            print('label',out[2].dtype, out[2].shape)

    print('Running the Network')
    ckpt = tf.train.get_checkpoint_state(config.ckpt_dir)
    with tf.Session(config=config.tf_config) as sess:
        sess.run(init_op)
        sess.graph.finalize()  # prevent operations to be added to the graph
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)#自动恢复model_checkpoint_path保存模型一般是最新
            print("Model restored...")
        train_writer = tf.summary.FileWriter(config.tb_dir, sess.graph)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for step in np.arange(1, config.MAX_STEP+1):
                if coord.should_stop():
                    break
                tra_images, tra_fixs = sess.run([img_batch, fix_batch])
                feed_dict = {input_images: tra_images, input_labels: tra_fixs}
                train_part(step, config.MAX_STEP, config.summary_step)
                
                # debug_readout_part()
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()
