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
from MetaFunc import construct_weights_modulation, forward_modulation

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
gpus=[1,2,3]

# data file
filename = "../data/train_multi_mask2/"

# saving path
path = './Result/task'

# setting global parameters
batch_size = 2
Total_batch_size = batch_size*2
num_frame = 8
image_dim = 256
Epoch = 100
sigmaInit = 0.01
step = 1
update_lr = 1e-5
num_updates = 5
num_task = 3

nameList = os.listdir(filename + 'gt/')
mask_sample, mask_s_sample = generate_masks_MAML(filename, num_task)

if not os.path.exists(path):
    os.mkdir(path)

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

weights, weights_m = construct_weights_modulation(sigmaInit)

tower_grads = []
tower_loss = []
reuse_vars = False

mask = tf.placeholder('float32', [num_task, image_dim, image_dim, num_frame])
X_meas_re = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, 1])
X_gt = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, num_frame])
Y_meas_re = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, 1])
Y_gt = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, num_frame])

for i in range(len(gpus)):
    with tf.device('/gpu:%d' % i):
        with tf.variable_scope('forward', reuse=reuse_vars):
            xtask_output = forward_modulation(mask[i], X_meas_re[i], X_gt[i], weights, weights_m, batch_size, num_frame)
            maml_grads = tf.gradients(xtask_output['loss'], list(weights_m.values()))
            gradients = dict(zip(weights_m.keys(), maml_grads))
            fast_weights = dict(zip(weights_m.keys(), [weights_m[key] - update_lr * gradients[key] for key in weights_m.keys()]))

            for j in range(num_updates - 1):
                xtask_output = forward_modulation(mask[i], X_meas_re[i], X_gt[i], weights, fast_weights, batch_size, num_frame)
                maml_grads = tf.gradients(xtask_output['loss'], list(fast_weights.values()))
                gradients = dict(zip(fast_weights.keys(), maml_grads))
                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - update_lr * gradients[key] for key in fast_weights.keys()]))

            ytask_output = forward_modulation(mask[i], Y_meas_re[i], Y_gt[i], weights, fast_weights, batch_size, num_frame)

            loss_op = ytask_output['loss']

            optimizer = tf.train.AdamOptimizer(learning_rate=0.00025)
            grads = optimizer.compute_gradients(loss_op)

            reuse_vars = True
            tower_grads.append(grads)
            tower_loss.append(loss_op)

tower_grads = average_gradients(tower_grads)

update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update):
    train_op = optimizer.apply_gradients(tower_grads)
    tower_loss = tf.reduce_sum(tower_loss)

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

            _, Loss = sess.run([train_op, tower_loss],
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

