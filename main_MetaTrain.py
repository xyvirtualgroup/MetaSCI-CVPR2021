"""
@author : Hao

"""

import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_eager_execution()

import numpy as np
import os
import random
import scipy.io as sci
from utils import generate_masks_MAML
import time
from tqdm import tqdm
from MetaFunc import construct_weights_modulation, MAML_modulation

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# data file
filename = "./data/train_multi_mask2/"

# saving path
path = './Result'

# setting global parameters
batch_size = 1
Total_batch_size = batch_size*2
num_frame = 8
image_dim = 256
Epoch = 100
sigmaInit = 0.01
step = 1
update_lr = 1e-5
num_updates = 5
num_task = 3

weights, weights_m = construct_weights_modulation(sigmaInit)

mask = tf.placeholder('float32', [num_task, image_dim, image_dim, num_frame])
X_meas_re = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, 1])
X_gt = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, num_frame])
Y_meas_re = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, 1])
Y_gt = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, num_frame])

final_output = MAML_modulation(mask, X_meas_re, X_gt, Y_meas_re, Y_gt, weights, weights_m, batch_size, num_frame, update_lr, num_updates)

optimizer = tf.train.AdamOptimizer(learning_rate=0.00025).minimize(final_output['Loss'])
#
nameList = os.listdir(filename + 'gt/')
mask_sample, mask_s_sample = generate_masks_MAML(filename, num_task)

if not os.path.exists(path):
    os.mkdir(path)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(Epoch):
        random.shuffle(nameList)
        epoch_loss = 0
        begin = time.time()

        for iter in tqdm(range(int(len(nameList)/Total_batch_size))):
            sample_name = nameList[iter*Total_batch_size: (iter+1)*Total_batch_size]
            X_gt_sample = np.zeros([num_task, batch_size, image_dim, image_dim, num_frame])
            X_meas_sample = np.zeros([num_task, batch_size, image_dim, image_dim])
            Y_gt_sample = np.zeros([num_task, batch_size, image_dim, image_dim, num_frame])
            Y_meas_sample = np.zeros([num_task, batch_size, image_dim, image_dim])

            for task_index in range(num_task):
                for index in range(len(sample_name)):
                    gt_tmp = sci.loadmat(filename + 'gt/' + sample_name[index])
                    meas_tmp = sci.loadmat(filename + 'measurement' + str(task_index+1) + '/' + sample_name[index])

                    if index < batch_size:
                        if "patch_save" in gt_tmp:
                            X_gt_sample[task_index, index, :, :] = gt_tmp['patch_save'] / 255
                        elif "p1" in gt_tmp:
                            X_gt_sample[task_index, index, :, :] = gt_tmp['p1'] / 255
                        elif "p2" in gt_tmp:
                            X_gt_sample[task_index, index, :, :] = gt_tmp['p2'] / 255
                        elif "p3" in gt_tmp:
                            X_gt_sample[task_index, index, :, :] = gt_tmp['p3'] / 255

                        X_meas_sample[task_index, index, :, :] = meas_tmp['meas'] / 255

                    else:
                        if "patch_save" in gt_tmp:
                            Y_gt_sample[task_index, index-batch_size, :, :] = gt_tmp['patch_save'] / 255
                        elif "p1" in gt_tmp:
                            Y_gt_sample[task_index, index-batch_size, :, :] = gt_tmp['p1'] / 255
                        elif "p2" in gt_tmp:
                            Y_gt_sample[task_index, index-batch_size, :, :] = gt_tmp['p2'] / 255
                        elif "p3" in gt_tmp:
                            Y_gt_sample[task_index, index-batch_size, :, :] = gt_tmp['p3'] / 255

                        Y_meas_sample[task_index, index-batch_size, :, :] = meas_tmp['meas'] / 255

            X_meas_re_sample = X_meas_sample / np.expand_dims(mask_s_sample, axis=1)
            X_meas_re_sample = np.expand_dims(X_meas_re_sample, axis=-1)

            Y_meas_re_sample = Y_meas_sample / np.expand_dims(mask_s_sample, axis=1)
            Y_meas_re_sample = np.expand_dims(Y_meas_re_sample, axis=-1)

            _, Loss = sess.run([optimizer, final_output['Loss']],
                               feed_dict={mask: mask_sample,
                                          X_meas_re: X_meas_re_sample,
                                          X_gt: X_gt_sample,
                                          Y_meas_re: Y_meas_re_sample,
                                          Y_gt: Y_gt_sample})

            epoch_loss += Loss

        end = time.time()

        print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / int(len(nameList)/batch_size)),
              "  time: {:.2f}".format(end - begin))

        if (epoch+1) % step == 0:
            saver.save(sess, path + '/model_{}.ckpt'.format(epoch))

