#!/usr/bin/python3
"""Training and Validation On Classification Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import os
import sys
import math
import random
import shutil
import argparse
import importlib
import data_utils
import numpy as np
import pointfly as pf
import tensorflow as tf
from datetime import datetime
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-t', help='Path to data', required=True)
    parser.add_argument('--path_val', '-v', help='Path to validation data')
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', help='Path to folder for saving check points and summary', required=True)
    parser.add_argument('--model', '-m', help='Model to use', required=True)
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)
    parser.add_argument('--gpu', '-gpu', help='Setting to use', required='0')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.save_folder, '%s_%s_%s_%d' % (args.model, args.setting, time_string, os.getpid()))
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    sys.stdout = open(os.path.join(root_folder, 'log.txt'), 'w')

    print('PID:', os.getpid())

    print(args)

    model = importlib.import_module(args.model)
    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    num_epochs = setting.num_epochs
    batch_size = setting.batch_size
    sample_num = setting.sample_num
    point_num = 2048
    rotation_range = setting.rotation_range
    scaling_range = setting.scaling_range
    jitter = setting.jitter
    pool_setting_train = None if not hasattr(setting, 'pool_setting_train') else setting.pool_setting_train

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    sys.stdout.flush()
    data_train, label_train, weight_train, box_sizes, len = data_utils.load_file(args.path)

    num_train = len

    print('{}-{:d} training samples.'.format(datetime.now(), len))
    sys.stdout.flush()

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(None, None, 2), name="indices")
    xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
    rotations = tf.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
    jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training = tf.placeholder(tf.bool, name='is_training')

    weight_train_placeholder = tf.placeholder(tf.float32, shape=(batch_size), name="weight")
    ### add weight
    data_train_placeholder = tf.placeholder(tf.float32, shape=(batch_size, point_num, 6), name='data_train')
    label_train_placeholder = tf.placeholder(tf.int64, shape=(batch_size), name='label_train')
    size_train_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 1, 3), name="weight")
    ########################################################################
    batch_num_per_epoch = math.floor(num_train / batch_size)

    print('{}-{:d} training batches per_epoch.'.format(datetime.now(), batch_num_per_epoch))
    sys.stdout.flush()

    pts_fts_sampled = tf.gather_nd(data_train_placeholder, indices=indices, name='pts_fts_sampled')
    features_augmented = None
    if setting.data_dim > 3:
        points_sampled, features_sampled = tf.split(pts_fts_sampled,
                                                    [3, setting.data_dim - 3],
                                                    axis=-1,
                                                    name='split_points_features')
        if setting.use_extra_features:
            if setting.with_normal_feature:
                if setting.data_dim < 6:
                    print('Only 3D normals are supported!')
                    exit()
                elif setting.data_dim == 6:
                    features_augmented = pf.augment(features_sampled, rotations)
                else:
                    normals, rest = tf.split(features_sampled, [3, setting.data_dim - 6])
                    normals_augmented = pf.augment(normals, rotations)
                    features_augmented = tf.concat([normals_augmented, rest], axis=-1)
            else:
                features_augmented = features_sampled
    else:
        points_sampled = pts_fts_sampled
    points_augmented = pf.augment(points_sampled, xforms, jitter_range)

    net = model.Net(points=points_augmented, features=features_augmented, is_training=is_training, setting=setting)
    #logits = net.logits
    feature = net.fc_layers[-1]

    ####
    box_size = size_train_placeholder
    #box_size = tf.expand_dims(size_train_placeholder, axis=1, name='box_size')
    box_feature = tf.layers.dense(inputs=box_size, units=20)
    feature_concat = tf.concat((feature, box_feature), 2)
    output = tf.layers.dense(inputs=feature_concat, units=256)
    logits = tf.layers.dense(inputs=output, units=100)
    ####

    probs = tf.nn.softmax(logits, name='probs')
    predictions = tf.argmax(probs, axis=-1, name='predictions',output_type=tf.int32)
    predictions = tf.squeeze(predictions)

    labels_2d = tf.expand_dims(label_train_placeholder, axis=-1, name='labels_2d')
    labels_tile = tf.tile(labels_2d, (1, tf.shape(logits)[1]), name='labels_tile')
    # loss_op = tf.losses.sparse_softmax_cross_entropy(labels=labels_tile, logits=logits)
    weights_2d = tf.expand_dims(weight_train_placeholder, axis=-1, name='weights_2d')
    loss_op = tf.losses.sparse_softmax_cross_entropy(labels=labels_tile, logits=logits, weights=weights_2d)

    lr_exp_op = tf.train.exponential_decay(setting.learning_rate_base, global_step, setting.decay_steps,
                                           setting.decay_rate, staircase=True)

    lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)

    _ = tf.summary.scalar('learning_rate', tensor=lr_clip_op)

    reg_loss = setting.weight_decay * tf.losses.get_regularization_loss()

    if setting.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_clip_op, epsilon=setting.epsilon)
    elif setting.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr_clip_op, momentum=setting.momentum, use_nesterov=True)

    train_op = optimizer.minimize(loss_op + reg_loss, global_step=global_step)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=None)

    folder_ckpt = os.path.join(root_folder, 'ckpts')
    if not os.path.exists(folder_ckpt):
        os.makedirs(folder_ckpt)

    folder_summary = os.path.join(root_folder, 'summary')
    if not os.path.exists(folder_summary):
        os.makedirs(folder_summary)

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))
    sys.stdout.flush()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(folder_summary, sess.graph)
        sess.run(init_op)
        # Load the model
        if args.load_ckpt is not None:
            saver.restore(sess, args.load_ckpt)
            print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))
        print('total-[Train]-Iter: ', num_epochs)
        sys.stdout.flush()

        num_epochs = 1 # test mode
        dataset = 'ScanNet'
        if dataset == 'S3DIS':
            categories = [6, 8, 9, 14, 99]  # chair,board,table,sofa
        elif dataset == 'Matterport':
            categories = [3, 5, 7, 8, 11, 15, 18, 22, 25, 28]
        elif dataset == 'ScanNet':
            categories = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        categories = np.array(categories)

        TP = np.zeros(categories.shape[0])
        FP = np.zeros(categories.shape[0])
        FN = np.zeros(categories.shape[0])
        TN = np.zeros(categories.shape[0])
        recall = np.zeros(categories.shape[0])
        precision = np.zeros(categories.shape[0])

        for epoch_idx_train in range(num_epochs):
            print('xxxx')
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            if epoch_idx_train == num_epochs - 1:
                confidences = []
                cloud_features = []
                for batch_idx_train in range(batch_num_per_epoch):
                    print('batch_idx_train',batch_idx_train)
                    index_ch = np.arange(len)
                    # do not shuttle
                    label = []
                    weight = []
                    size = []
                    dataset_train = []
                    for i in range(batch_size):
                        #print('i',i)
                        k = batch_idx_train * batch_size + i
                        label.append(label_train[index_ch[k]])
                        weight.append(weight_train[index_ch[k]])
                        size.append(box_sizes[index_ch[k]])
                        data = []
                        count = 0
                        with open(data_train[index_ch[k]]) as fpts:
                            while 1:
                                line = fpts.readline()
                                if not line:
                                    break;
                                L = line.split(' ')
                                L = [float(i) for i in L]
                                data.append(np.array(L))
                                count = count + 1
                            data = np.array(data)
                            data = data[:,:6]
                            trans_x = (min(data[:, 0])+max(data[:, 0]))/2
                            trans_y = (min(data[:, 1])+max(data[:, 1]))/2
                            trans_z = (min(data[:, 2])+max(data[:, 2]))/2
                            data = data - [trans_x, trans_y, trans_z, 0.5, 0.5, 0.5]
                            if (count >= 2048):
                                index = np.random.choice(count, size=2048, replace=False)
                                # index = random.sample(range(0, count), 2048)
                                dataset = data[index, :]
                            else:
                                # k = random.sample(range(0, count), count)
                                index = np.random.choice(count, size=2048, replace=True)
                                dataset = data[index, :]
                            dataset_train.append(dataset)
                    data_batch = np.array(dataset_train)
                    label_batch = np.array(label)
                    weight_batch = np.array(weight)
                    size_batch = np.array(size)
                    ######################################################################
                    # TESting
                    offset = int(random.gauss(0, sample_num * setting.sample_num_variance))
                    offset = max(offset, -sample_num * setting.sample_num_clip)
                    offset = min(offset, sample_num * setting.sample_num_clip)
                    sample_num_train = sample_num + offset
                    xforms_np, rotations_np = pf.get_xforms(batch_size,
                                                            rotation_range=rotation_range,
                                                            scaling_range=scaling_range,
                                                            order=setting.rotation_order)
                    loss, prediction, confidence, cloud_feature = sess.run([loss_op, predictions, probs, feature],
                                                feed_dict={
                                                             data_train_placeholder: data_batch,
                                                             label_train_placeholder: label_batch,
                                                             indices: pf.get_indices(batch_size,
                                                                                     sample_num_train,
                                                                                     point_num,
                                                                                     pool_setting_train),
                                                             xforms: xforms_np,
                                                             rotations: rotations_np,
                                                             jitter_range: np.array([jitter]),
                                                             is_training: True,
                                                             weight_train_placeholder: weight_batch,
                                                             size_train_placeholder: size_batch,
                                                                 })
                    print('confidence.shape',confidence.shape)
                    confidences.append(confidence)
                    cloud_features.append(cloud_feature)
                    correct = np.sum(prediction == label_batch)
                    total_correct += correct
                    total_seen += batch_size
                    loss_sum += loss

                    for i in range(categories.shape[0]):
                        for j in range(label_batch.shape[0]):
                            pred = prediction[j]
                            label = label_batch[j]
                            cat = categories[i]

                            if label == cat and pred == cat:
                                TP[i] += 1
                            elif label == cat and pred != cat:
                                FN[i] += 1
                            elif label != cat and pred == cat:
                                FP[i] += 1
                            elif label != cat and pred != cat:
                                TN[i] += 1

                    for i in range(categories.shape[0]):
                        recall[i] = TP[i] / (TP[i] + FN[i])
                        precision[i] = TP[i] / (TP[i] + FP[i])
                    print('precision', precision)
                    print('recall', recall)
                
                for i in range(categories.shape[0]):
                    recall[i] = TP[i] / (TP[i] + FN[i])
                    precision[i] = TP[i] / (TP[i] + FP[i])
                print('precision', precision)
                print('recall', recall)

                confidences = np.array(confidences).reshape((-1, 101))
                cloud_features = np.array(cloud_features)
                cloud_features = cloud_features.reshape((-1, cloud_features.shape[-1]))

                # class num :101
                np.savetxt(os.path.join(folder_summary, 'confidence.txt'), confidences)
                np.savetxt(os.path.join(folder_summary, 'feature.txt'), cloud_features)
                print('confidences and features saved to {}!'.format(folder_summary))
                print('confidences shape is {}!'.format(confidences.shape))
                filename_ckpt = os.path.join(folder_ckpt, 'iter')
                saver.save(sess, filename_ckpt, global_step=global_step)
                print('{}-Checkpoint saved to {}!'.format(datetime.now(), filename_ckpt))
                print('{}-[test]-done: {:06d}  Loss: {:.4f}   Acc: {:.4f}  lr:{:.4f}'
                          .format(datetime.now(), epoch_idx_train, loss_sum, (total_correct / float(total_seen)),
                                  learningrate))
                sys.stdout.flush()
            else:
                for batch_idx_train in range(batch_num_per_epoch):
                    ########################################################################
                    #sample
                    index_ch = np.arange(len)
                    np.random.shuffle(index_ch)
                    label = []
                    weight = []
                    dataset_train = []
                    size = []
                    for i in range(batch_size):
                        k = batch_idx_train * batch_size + i
                        label.append(label_train[index_ch[k]])
                        #weight.append(pow(weight_train[index_ch[k]], 2))
                        weight.append(weight_train[index_ch[k]])
                        size.append(box_sizes[index_ch[k]])
                        data = []
                        count = 0
                        with open(data_train[index_ch[k]]) as fpts:
                            while 1:
                                line = fpts.readline()
                                if not line:
                                    break;
                                L = line.split(' ')
                                L = [float(i) for i in L]
                                data.append(np.array(L))
                                count = count + 1
                            data = np.array(data)
                            data = data[:, :6]
                            trans_x = (min(data[:, 0])+max(data[:, 0]))/2
                            trans_y = (min(data[:, 1])+max(data[:, 1]))/2
                            trans_z = (min(data[:, 2])+max(data[:, 2]))/2
                            data = data - [trans_x, trans_y, trans_z, 0.5, 0.5, 0.5]
                            ######################################

                            if (count >= 2048):
                                index = np.random.choice(count, size=2048, replace=False)
                                dataset=data[index, :]
                            else:
                                # k = random.sample(range(0, count), count)
                                index = np.random.choice(count, size=2048, replace=True)
                                dataset =data[index, :]
                            dataset_train.append(dataset)
                    data_batch =np.array(dataset_train)
                    label_batch = np.array(label)
                    weight_batch = np.array(weight)
                    size_batch = np.array(size)
                    ######################################################################
                    # Training
                    offset = int(random.gauss(0, sample_num * setting.sample_num_variance))
                    offset = max(offset, -sample_num * setting.sample_num_clip)
                    offset = min(offset, sample_num * setting.sample_num_clip)
                    sample_num_train = sample_num + offset
                    xforms_np, rotations_np = pf.get_xforms(batch_size,
                                                            rotation_range=rotation_range,
                                                            scaling_range=scaling_range,
                                                            order=setting.rotation_order)
                    _, loss, prediction, learningrate,bs,bf = sess.run([train_op,loss_op,predictions,lr_clip_op,box_size,box_feature] ,
                             feed_dict={
                                 data_train_placeholder: data_batch,
                                 label_train_placeholder: label_batch,
                                 indices: pf.get_indices(batch_size, sample_num_train, point_num, pool_setting_train),
                                 xforms: xforms_np,
                                 rotations: rotations_np,
                                 jitter_range: np.array([jitter]),
                                 is_training: True,
                                 weight_train_placeholder: weight_batch,
                                 size_train_placeholder: size_batch,
                             })
                    correct = np.sum(prediction == label_batch)
                    total_correct += correct
                    total_seen += batch_size
                    loss_sum += loss
                    if batch_idx_train % 50 == 0 or 1:
                        print('{}-[Train]-Iter:{:06d}   batch_idx:{:06d}  Loss: {:.4f}   Acc: {:.4f}  lr:{:.4f}'
                              .format(datetime.now(), epoch_idx_train, batch_idx_train, loss, (total_correct / float(total_seen)),
                                      learningrate))
                        sys.stdout.flush()
                print('{}-[Train]-Iter: {:06d}  Loss: {:.4f}   Acc: {:.4f}  lr:{:.4f}'
                          .format(datetime.now(), epoch_idx_train, loss_sum, (total_correct / float(total_seen)),learningrate))
                filename_ckpt = os.path.join(folder_ckpt, 'iter')
                saver.save(sess, filename_ckpt, global_step=global_step)
                print('{}-Checkpoint saved to {}!'.format(datetime.now(), filename_ckpt))
                sys.stdout.flush()

                ####################################################################

        print('{}-Done!'.format(datetime.now()))


if __name__ == '__main__':
    main()
