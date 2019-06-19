import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import time
import os
import sys
import tf_util
import random
from sklearn import preprocessing
from show_samples import plotit
import show_pc
from show_pc import PointCloud
from matplotlib import pyplot as plt
from transform_nets import homo_transform_net, input_transform_net, feature_transform_net
from plot4experiment import vis_first_layer , plot_embedding_3d
from read_data import get_local_eig_np

from mayavi import mlab
LOG_FOUT = open('log_train.txt', 'w')

# pc = np.loadtxt('monsterbladecaranya.txt')  # 4*1024 x 3
# pc /= 2
# # show_pc.show_custom(pc)
# ran_trans = -200+400*np.random.random([4, 3])
# print(ran_trans)
# pc[0:1024] += ran_trans[0, :]
# pc[1024:2048] += ran_trans[1, :]
# pc[2048:3072] += ran_trans[2, :]
# pc[3072:4096] += ran_trans[3, :]
# pc = np.reshape(pc, (4, 1024, 3))

init_learning_rate = float(10 ** -np.random.randint(-1, 8) * np.random.random(1))
init_learning_rate = 0.0002
decay_rate = float(np.random.random(1) * 10 ** -np.random.randint(0, 2))
decay_rate = 0.999
decay_step = int(np.random.randint(1000, 1001))
decay_step = 1000
batchsize = 100
max_epoch = 100  # 200
nb_classes = 8
nb_points = 1024
key_pts_percentage = 0.1
pc_scale_factor = 100
# tile_size = 256   # total

readh5 = h5py.File('/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/8object0.04noise/mykeyptssimuN_data.h5')  # file path

pc_tile = readh5['train_set'][:]  # 20000 * 1024 * 3
pc_local_eigs = readh5['train_set_local'][:]  # 20000 * 102 * 9
pc_label = readh5['train_labels'][:]

pc_test = pc_tile[0, :, :]
pc_test = PointCloud(pc_test)
quaternion_range = [0, 0.5]
translation_range = [-100, 100]

pc_tile *= pc_scale_factor   # for scale

# <editor-fold desc="use this snipet to make">
# for i in range(pc_tile.shape[0]):
#     pc = PointCloud(pc_tile[i, :, :])
#     pc.normalize()
#     pc_tile[i, :, :] = np.expand_dims(pc.position, axis=0)
#
# readh5["train_set"][...] = pc_tile
# readh5.close()
# </editor-fold>

# pc_tile += -5 + 10*np.random.random(size=(20000, 1, 3))  # 20000 * 1024 * 3



light = np.array([[1,  0,  0],
                 [0,  0,  1],
                 [1,  1,  0],
                 [0,  1,  0]])
#dark = np.random.random((4, 3)) * 0.8 + 0.2
print('light color:', light)
shade = light * 0.7

light1 = light[0, :].tolist()
shade1 = shade[0, :].tolist()
light2 = light[1, :].tolist()
shade2 = shade[1, :].tolist()
light3 = light[2, :].tolist()
shade3 = shade[2, :].tolist()
light4 = light[3, :].tolist()
shade4 = shade[3, :].tolist()

colorset = [shade1, light1, shade2, light2, shade3, light3, shade4, light4]

colorset = [[plt.cm.Set1(i)[:3], tuple(np.asarray(plt.cm.Set1(i)[:3])*0.7)] for i in range(8)]

print('initial learning rate :', init_learning_rate, '\n', 'decay_rate is :', decay_rate, '\n',
      'decay_step is: ', decay_step, 'max_epoch is:', max_epoch, '\n', 'batchsize is:', batchsize)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def get_pts_nei(idx, batch, nb_points):
    """

    :param idx:  ? x 3
    :param batch:  integer
    :param nb_points: interger
    :return: batch x nb_points x max_nb_pts_neighbors points index
    """

    """
    get max_number of points neighbors
    """
    b = idx[:, 0]  # shape: ?
    pt = idx[:, 0:2]  # ? x 2

    _, mid_idx, counts = np.unique(b, return_index=True, return_counts=True)
    max_nb_pts_neighbors = 0
    # print('max_nb_pts_neighbors:', max_nb_pts_neighbors)
    for i in range(batch):
        if i == 0:
            this_batch, remain = pt[:counts[i], :], pt[counts[i]:, :]
        else:
            this_batch, remain = remain[:counts[i], :], remain[counts[i]:, :]
        # print('this batch:', this_batch)
        _,  current_max = np.unique(this_batch[:, 1], return_counts=True)
        current_max = np.float32(np.max(current_max))
        # print('currentmax:', current_max)
        max_nb_pts_neighbors = np.where(current_max > max_nb_pts_neighbors,
                                        current_max, max_nb_pts_neighbors,)

    point_rneighbors = np.empty((np.int(batch), int(nb_points), int(max_nb_pts_neighbors)))
    point_rneighbors[:] = np.nan

    b = idx[:, 0]
    _, mid_idx, counts = np.unique(b, return_index=True, return_counts=True)

    for i in range(batch):
        if i == 0:
            this_batch, remain = idx[:counts[i], :], idx[counts[i]:, :]
        else:
            this_batch, remain = remain[:counts[i], :], remain[counts[i]:, :]
        # print('this batch:', this_batch)
        pts_idx,  nb_neighbor = np.unique(this_batch[:, 1],  return_counts=True)  # ? x 1
        # print('nb_neighbor: ', nb_neighbor)
        start = 0

        for j in range(np.shape(nb_neighbor)[0]):
            neighbor_idx = this_batch[start: (start + nb_neighbor[j]), 2]
            start += nb_neighbor[j]
            # print('neighbor_idx:', neighbor_idx)
            point_rneighbors[i, pts_idx[j], :nb_neighbor[j]] = neighbor_idx

    return np.int32(point_rneighbors)


def get_pts_cov(pc, pts_r_neirhbor_idx):
    """

    :param pc:  bxnx3
    :param pts_r_neirhbor_idx: bxmxk   m can be less than n, only compute key points cov
    :return: b x m x 3 x 3
    """
    batch = pc.shape[0]
    nb_key_pts = pts_r_neirhbor_idx.shape[1]

    cov = np.empty((batch, nb_key_pts, 3, 3))
    cov[:] = np.nan

    for i in range(batch):
        for j in range(nb_key_pts):
            this_pt_nei_idx = pts_r_neirhbor_idx[i, j, :][pts_r_neirhbor_idx[i, j, :] >= 0]
            neighbor_pts = pc[i, this_pt_nei_idx, :]  #
            # print('neighbor_pts shape:', neighbor_pts.shape)
            if neighbor_pts.size == 0:
                cov[i, j, :, :] = np.eye(3)
            else:
                cov[i, j, :, :] = np.cov(neighbor_pts, rowvar=False)

    return np.float32(cov)


def train(model_name, use_local=False):
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            pointclouds_pl, labels_pl = placeholder_inputs(batchsize, nb_points)  # batch, num_points
            pt_local_eigs_pl = tf.placeholder(tf.float32, shape=(batchsize, int(nb_points*0.1), 9))
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model_prediction and loss here end_points stores the transformation
            pred, end_points = get_model(pointclouds_pl, pt_local_eigs_pl, is_training_pl, bn_decay=bn_decay, use_local=use_local)

            loss = get_loss(pred, labels_pl, end_points)

            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(batchsize)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)  # write this in to tensorboard

            optimizer = tf.train.AdamOptimizer(learning_rate)
            optimizer_fast = tf.train.AdamOptimizer(learning_rate * 10)

            main_vars_set = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main_net')
            homo_var_set = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='homo_transform_net')
            # c=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='input_transform_net')
            # d=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='feature_transform_net')
            all_var_set = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)  # all variables
            other_var_list = list(set(all_var_set).difference(set(homo_var_set)))
            input_fea_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='input_transform_net')

            assert set(all_var_set) == set(homo_var_set) | set(other_var_list)  # assure that all variables are included.

            print('length of :other var list,homo_var_set, all_var_set', len(other_var_list), len(homo_var_set),
                  len(all_var_set))
            #train_op_other = optimizer.minimize(loss, global_step=batch, var_list=other_var_list)
            #train_op_homo = optimizer_fast.minimize(loss, global_step=batch, var_list=homo_var_set)
            train_op_all = optimizer.minimize(loss, global_step=batch, var_list=all_var_set)
            # train_op_input_fea = optimizer_fast.minimize(loss, global_step= batch, var_list=input_fea_var_list )
            #train_op = tf.group(train_op_other, train_op_homo)
            train_op = train_op_all

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        # <editor-fold desc="Create a session and some configuration about that session">
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # saver.restore(sess, "/log/model.ckpt")  # if you want to load model

        # </editor-fold>
        # Add summary writers
        # merged = tf.merge_all_summaries()
        train_summary_merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join('log', 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join('log', 'test'), sess.graph)
        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'pt_local_eigs_pl': pt_local_eigs_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,   # prediction from network
               'loss': loss,
               'train_op': train_op,
               'train_summary_merged': train_summary_merged,
               'step': batch,
               'trans_dis': end_points['trans_dis'],
               'compare': end_points['compare'],
               'original_pc': end_points['test_layer1'],
               'recovered_pc': end_points['test_layer2']
               }

        # tvars = tf.all_variables()
        # trianable = tf.trainable_variables()
        # tvars_vals = sess.run(tvars)
        # Fack = tvars[1]
        # nodes = [n for n in tf.get_default_graph().as_graph_def().node]
        # print('length of all_variables:', len(tvars_vals))
        # print('length of nodes:', len(nodes), type(nodes[0]))
        #
        # for var, val in zip(tvars, tvars_vals):
        #     print('ALL VARIABLE:::look   ', var.shape, var.name)  # Prints the name of the variable alongside its value.

        for epoch in range(max_epoch):   #how many epoch epoches  you want to train max epoch
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer, epoch)  # trainning and evaluation is starting simultaneous
            eval_one_epoch(sess, ops, test_writer, epoch)  # after training, evaluation is begin.

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join('tmp', model_name))
                log_string("Model saved in file: %s" % save_path)


@timeit
def inference(model_path, pcpath='test_dataset.h5', show_result=False, use_local=False, times=1,
              vis_feature=False, vis_tsne=False, test_batchsize=1):
    """
    After trainning , restore the saved pre-trained model, and test for inference.
    :param model_path:
    :param show_result:
    :param use_local:
    :param times:  how many times to test
    :param vis_feature:
    :param vis_tsne:
    :param test_batchsize:
    :return:
    """

    try:
        readh5 = h5py.File(pcpath)  # file path
        pc_tile = readh5['test_set'][:]  # m * 1024 * 3
        pc_local_eigs = readh5['test_set_local'][:]  # m * 102 * 9
        pc_label = readh5['test_labels'][:]
        pc_tile *= pc_scale_factor
    except:
        log_string('TEST SET NOT FOUND, USING TRAINING SET TO PLOT NOW!!!')
        global readh5
        global pc_tile
        global pc_local_eigs
        global pc_label

    tf.reset_default_graph()

    print('test_batchsize is :', test_batchsize)
    with tf.device('/gpu:0'):
        pointclouds_pl, labels_pl = placeholder_inputs(test_batchsize, nb_points)
        pt_local_eigs_pl = tf.placeholder(tf.float32, shape=(test_batchsize, int(nb_points * 0.1), 9))
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = get_model(pointclouds_pl, pt_local_eigs_pl, is_training_pl, apply_rand=True, use_local=use_local)

        loss = get_loss(pred, labels_pl, end_points)

        # Add ops to save and restore all the variables.

        saver = tf.train.Saver()
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    inference_summary_merged = tf.summary.merge_all()
    sess = tf.Session(config=config)
    inference_writer = tf.summary.FileWriter(os.path.join('log', 'inference'), sess.graph)
    # Restore variables from disk.
    saver.restore(sess, model_path)
    log_string("Model restored.")

    # sess.run(init, {is_training_pl: False})

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'pt_local_eigs_pl': pt_local_eigs_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,  # prediction from network
           'loss': loss,
           # 'train_op': train_op,
           'inference_summary_merged': inference_summary_merged,
           'step': tf.Variable(initial_value=np.ones(shape=1)),   # TODO
           'trans_dis': end_points['trans_dis'],
           'compare':  end_points['compare'],
           'original_pc': end_points['test_layer1'],
           'recovered_pc': end_points['test_layer2'],
           'moved_pc': end_points['moved_pc'],
           'random_pos': end_points['random_pos'],
           'predict_pos': end_points['predict_pos'],
           'first_layer_output': end_points['first_layer_output'],
           'classification_output': end_points['classification_output']
           }
    classification_output4tsne = np.zeros((times, test_batchsize, 256))  # times x b x 256
    tsne_label = np.zeros((times, test_batchsize))  # times x b
    point_clouds = np.zeros((times, test_batchsize, nb_points, 3))  # times x b  x nb_points x 3
    for j in range(times):  # how many times you want to test

        if vis_feature:
            x = np.linspace(0, 1, 16)
            y = np.linspace(0, 1, 16)
            z = np.linspace(0, 1, 16)
            xi, yi, zi = np.meshgrid(x, y, z)
            points = np.concatenate([np.reshape(xi, [-1, 1]), np.reshape(yi, [-1, 1]), np.reshape(zi, [-1, 1])], axis=1)
            np.random.shuffle(points)
            points = points[np.newaxis, 0:1024, :]
            # pc_tile = points   # todo once define pc_tile here, pc_tile becomes local variable
            # pc_local_eigs = get_local_eig_np(points)   # todo once define pc_local_eigs here,pc_tile becomes local variable

        rand_idx = np.random.choice(pc_tile.shape[0], test_batchsize)   # randomly choose a batchsize of data

        tsne_label[j] = pc_label[rand_idx]  # (b, )
        point_clouds[j] = pc_tile[rand_idx, :, :]  # (b x nb_points x 3)
        feed_dict = {ops['pointclouds_pl']: pc_tile[rand_idx, :, :],   # one point cloud at a time
                     ops['pt_local_eigs_pl']: pc_local_eigs[rand_idx, :, :],
                     ops['labels_pl']: pc_label[rand_idx],
                     ops['is_training_pl']: False}

        start = time.time()
        [pred_class, total_loss, ran_pos, predict_pos, trans_dis, opc, rpc, mpc, summary, cls_out, first_ly] = \
            sess.run([ops['pred'], ops['loss'], ops['random_pos'],
                      ops['predict_pos'], ops['trans_dis'],
                      ops['original_pc'], ops['recovered_pc'],
                      ops['moved_pc'], ops['inference_summary_merged'],
                      ops['classification_output'],
                      ops['first_layer_output']],
                      feed_dict=feed_dict)

        classification_output4tsne[j] = cls_out  # b x 256

        tf.summary.scalar('total_loss', total_loss)
        inference_writer.add_summary(summary, j)
        end = time.time()
        print('inference time cost:{} s'.format(end-start))
        print('trans dis is :{}'.format(trans_dis))
        print('total loss is:{}'.format(total_loss))

        pred_class = np.argmax(pred_class, 1)  # asy anotationed
        #print('pred_label:', pred_val)  # asy anotationed
        #print('truth_label:', current_label[start_idx:end_idx])  # asy anotationed
        correct = np.sum(pred_class == pc_label[rand_idx])   # asy anotationed

        print('predicted class is :{}, which {} are right'.format(pred_class, correct))
        print('***********')
        # print('pred_class:', pred_class, 'total loss: ', total_loss, 'random_pos:', ran_pos,
        #       'predicted pos:', predict_pos, 'pose distance:', trans_dis)

    if show_result:
        rand_trans = np.random.random([test_batchsize, 3])*0   # todo manually ajust the translation range
        rand_trans = np.expand_dims(rand_trans, axis=1)
        rand_trans = np.tile(rand_trans, [1, 1024, 1])

        opc += rand_trans
        rpc += rand_trans

        fig = show_pc.show_trans(mpc, rpc, colorset=colorset, scale=100, returnfig=True)  # simulate the ramdon

        filename1='before_alignment1.png'
        while(True):
            if os.path.exists(filename1):
                filename1 = filename1.split('.')[0][:-1] + str(int(filename1.split('.')[0][-1])+1) + '.png'
                continue
            break
        f = mlab.gcf()  # this two line for mlab.screenshot to work
        f.scene._lift()
        mlab.savefig(filename=filename1)
        mlab.close()

        fig = show_pc.show_trans(opc, rpc, colorset=colorset, scale=100, returnfig=True)  # after recover

        filename1='after_alignment1.png'
        while(True):
            if os.path.exists(filename1):
                filename1 = filename1.split('.')[0][:-1] + str(int(filename1.split('.')[0][-1])+1) + '.png'
                continue
            break
        f = mlab.gcf()  # this two line for mlab.screenshot to work
        f.scene._lift()
        mlab.savefig(filename=filename1)
        mlab.close()


    if vis_feature:
        print('points and values:', points, 'shapes', np.shape(points))
        print('points and values:', first_ly, 'shapes', np.shape(first_ly))
        np.save('points.npy', points)
        np.save('first_ly.npy', first_ly)
        vis_first_layer(np.squeeze(points, axis=0), np.squeeze(first_ly, axis=0))

    if vis_tsne:
        classification_output4tsne = np.reshape(classification_output4tsne, (-1, 256))
        tsne_label = np.reshape(tsne_label, (-1, ))
        point_clouds = np.reshape(point_clouds, (-1, nb_points, 3))
        np.save('classification_output4tsne.npy', classification_output4tsne)
        np.save('tsne_label.npy', tsne_label)
        np.save('point_clouds.npy', point_clouds)
        # plot_embedding_3d(classification_output4tsne, tsne_label, point_clouds=point_clouds)

def get_bn_decay(batch):
    # ofr batch decay
    bn_momentum = tf.train.exponential_decay(
        0.5,  # initial batch normalization decay rate
        batch * 4,
        1000,  # Decay step.
        0.5,     # Decay rate.
        staircase=True)
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)
    return bn_decay


def get_model(point_cloud, point_cloud_local, is_training, bn_decay=None, apply_rand=True, use_local=True):
    """
    Classification PointNet, input is BxNx3, output Bx4
    :param point_cloud:
    :param point_cloud_local:
    :param is_training:
    :param bn_decay:
    :param apply_rand: whether to apply random transformation for data augmentation
    :param use_local:
    :return:
    """

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    # point_cloud=zero_center_and_norm(point_cloud)  #zero_center and then normalize the input features,
    # need to fix: normalize all the axis with same coifficent
    # generate random_pose once here
    ran_pos = tf.concat([tf.random_uniform([batch_size, 1], minval=0.99, maxval=1),
                         tf.random_uniform([batch_size, 1], minval=quaternion_range[0], maxval=quaternion_range[1]),
                         tf.random_uniform([batch_size, 1], minval=quaternion_range[0], maxval=quaternion_range[1]),
                         tf.random_uniform([batch_size, 1], minval=quaternion_range[0], maxval=quaternion_range[1]),
                         tf.random_uniform([batch_size, 3], minval=translation_range[0], maxval=translation_range[1])], axis=1)  # random_ROTATION_and POSITION, batch x 7

    ran_pos = tf.concat([tf.divide(tf.slice(ran_pos, [0, 0], [batch_size, 4]),
                         tf.norm(tf.slice(ran_pos, [0, 0], [batch_size, 4]), axis=1, keep_dims=True)),
                         tf.slice(ran_pos, [0, 4], [batch_size, 3])], axis=1)       # normalize random pose

    end_points['random_pos'] = ran_pos
    ran_homo = tf_quat_pos_2_homo(ran_pos)  # Bx4x4
    if apply_rand:
        point_cloud_jitterd = apply_homo_to_pc(point_cloud, ran_homo)  # add this to input batch for data_augmentation
        end_points['moved_pc'] = point_cloud_jitterd
    else:
        point_cloud_jitterd = point_cloud

    with tf.variable_scope('homo_transform_net') as sc:
        transformation_7, test_layer1, test_layer2 = homo_transform_net(point_cloud_jitterd, point_cloud_local,
                                                                        is_training, bn_decay=bn_decay,
                                                                        use_local=use_local)  # B x 7 predicted transformation

    #transformation_7 = tf.concat([tf.slice(ran_pos, [0, 0], [batch_size, 1]), -1*tf.slice(ran_pos, [0, 1], [batch_size, 3]),
    #                              tf.slice(transformation_7, [0, 4], [batch_size, 3])], axis=1)  # leave the rotation unchanged
    end_points['predict_pos'] = transformation_7
    transformation = tf_quat_pos_2_homo(transformation_7)  # Bx4x4

    end_points['compare'] = [transformation_7, ran_pos]  # compare the predicted transformation and ground true transformation
    end_points['trans_dis'] = compute_pos_distance(transformation_7, ran_pos)  # add by asy, to compute the distance between predict and ground truth

    tf.summary.scalar('ang_dis', end_points['trans_dis'][0])
    tf.summary.scalar('pos_dis', end_points['trans_dis'][1])

    point_cloud_transformed = apply_homo_to_pc(point_cloud_jitterd, transformation)  # apply this for shape transformation.
    # point_cloud_transformed = point_cloud_jitterd                                    # if you don't apply this

    end_points['test_layer1'] = point_cloud              # original point cloud
    end_points['test_layer2'] = point_cloud_transformed  # point_cloud_jitterd for original,point_cloud_transformed for compare

    with tf.variable_scope('input_transform_net') as sc:
        transform3 = input_transform_net(point_cloud_transformed, is_training, bn_decay, K=3)
        # origin Bx3x3 transform

    point_cloud_transformed = tf.matmul(point_cloud_transformed, transform3)   # applay or not apply this original input transform net
    # B x n x3 mul Bx3x3 = B x n x 3

    input_image = tf.expand_dims(point_cloud_transformed, -1)  # become Bxnx3x1

    with tf.variable_scope('main_net') as sc:
        net = tf_util.conv2d(input_image, 64, [1, 3],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)      # Bxnx1x64

        net = tf_util.conv2d(net, 128, [1, 1],
                         padding = 'VALID', stride=[1, 1],
                         bn = True, is_training=is_training,
                         scope = 'conv2', bn_decay=bn_decay)       # Bx1024x1x128

        net = tf_util.conv2d(net, 256, [1, 1],
                         padding = 'VALID', stride=[1, 1],
                         bn = True, is_training=is_training,
                         scope = 'conv2_copy', bn_decay=bn_decay)       #   Bx1024x1x256

    with tf.variable_scope('feature_transform_net') as sc:
        transformation_feature = feature_transform_net(net, is_training, bn_decay, K=64)  # B X 64 X 64

    end_points['transform'] = transformation_feature  #note here the transform is dimension of 64

    #net_transformed = tf.matmul(tf.squeeze(net), transformation_feature) #Bx1024x64  matmul Bx64x64 equals Bx1024x64

    net_transformed = net            #not apply feature transform

    #net_transformed = tf.expand_dims(net_transformed, [2])  #Bx1024x1x64

    #  classification network
    with tf.variable_scope('main_net') as sc:
        net = tf_util.conv2d(net_transformed, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)      #Bx1024x1x64

        end_points['first_layer_output'] = tf.reshape(net, shape=[batch_size, 1024, 64])  # B X 1024 X 64

        net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope = 'conv4', bn_decay=bn_decay)      #Bx1024x1x128
        net = tf_util.conv2d(net, 256, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4_copy', bn_decay=bn_decay)      #Bx1024x1x128
        net = tf_util.conv2d(net, 512, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv5_copy', bn_decay=bn_decay)      #Bx1024x1x128
        net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)      #B x 1024 x 1 x 1024

    # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='maxpool') #B x1x1x1024

        net = tf.reshape(net, [batch_size, -1])  # Bx1024

        net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)  #Bx512
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)  #Bx256
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc3', bn_decay=bn_decay)  #Bx256
        # net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training,
        #                           scope='fc4', bn_decay=bn_decay)  #Bx256
        # net = tf_util.fully_connected(net, 64, bn=True, is_training=is_training,
        #                           scope='fc5', bn_decay=bn_decay)  #Bx256
        # net = tf_util.fully_connected(net, 32, bn=True, is_training=is_training,
        #                           scope='fc6', bn_decay=bn_decay)  #Bx256
        # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
        #                   scope='dp2')
        end_points['classification_output'] = tf.reshape(net, shape=[batch_size, -1])  #B X 256
        if use_local:
            # b x nb_key_pts x 9 x 1 , 9 because multi-scale
            point_cloud_local = tf.expand_dims(point_cloud_local, axis=-1)
            point_cloud_local = tf.reshape(point_cloud_local, [batch_size, int(1024*0.1), 9, 1])
            point_cloud_local = tf.layers.conv2d(inputs=point_cloud_local, filters=64,
                                                 kernel_size=[1, 9])  # b x nb_key_pts x 1 x 64
            point_cloud_local = tf.layers.conv2d(inputs=point_cloud_local, filters=256,
                                                 kernel_size=[1, 1])  # b x nb_key_pts x 1 x 256
            point_cloud_local = tf.layers.max_pooling2d(point_cloud_local,
                                                        pool_size=(int(nb_points * key_pts_percentage), 1),
                                                        strides=1)  # b x 1 x 1 x 256
            point_cloud_local = tf.reshape(point_cloud_local, [batch_size, -1])  # b x 256

            net = tf.concat([net, point_cloud_local], axis=-1)   # Bx256
            prediction = tf_util.fully_connected(net, nb_classes, activation_fn=None, scope='fc7')# B x 4
        else:
            prediction = tf_util.fully_connected(net, nb_classes, activation_fn=None, scope='train_without_local')
            # B x 4

    return prediction, end_points #net is the final prediction of 4 classes


def get_loss(pred, label, end_points, reg_weight=0.0001, rotation_weight=1000, pose_weight=10):
    """

    :param pred: B*NUM_CLASSES,
    :param label: B,
    :param end_points: some loss
    :param reg_weight: regularization term, not a big deal
    :param rotation_weight: weight for quaternion
    :param pose_weight: balance between pose and classification
    :return:
    """

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)   # pred:(Bx4) label:(B,),loss:(B,),originally logits = label, label=pred
    classify_loss = tf.reduce_mean(loss)   # you need mean
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform']  # BxKxK the feature transformation, did not used it anymore
    K = transform.get_shape()[1].value  # K=64

    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0, 2, 1]))  # BxKxK
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)  # the difference with identity matrix

    trans_diff = rotation_weight * end_points['trans_dis'][0] + end_points['trans_dis'][1]  # rotation and translation

    tf.summary.scalar('mat loss', mat_diff_loss)

    if pose_weight > 100000:
        final_loss = trans_diff
    else:

        final_loss = tf.add(classify_loss, mat_diff_loss * reg_weight)
        final_loss = tf.add(final_loss, trans_diff*pose_weight, name="final_loss")   # if trans_diff * 0.0 means end-to-end trainning

    return final_loss



def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_learning_rate(batch, truncated=0.0000001):
    learning_rate = tf.train.exponential_decay(
                        init_learning_rate,  # Base learning rate. default value:0.00001
                        batch * batchsize,  # Current index into the dataset.
                        decay_step,          # learning rate Decay step.
                        decay_rate,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, truncated)  # CLIP THE LEARNING RATE! do not small than this
    return learning_rate


def train_one_epoch(sess, ops, train_writer, epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    current_data = pc_tile  # (20000) x 1024 x 3 numpy array
    current_local_eig = pc_local_eigs   # (20000) x 102 x 9 numpy array
    current_label = pc_label

    current_data, current_label, current_local, _ = \
        shuffle_data(current_data, np.squeeze(current_label), current_local_eig)  # Shuffle train files
    current_label = np.squeeze(current_label)

    # here we implement asy's chunk sort method for test******

    file_size = current_data.shape[0]  # 4 x tile_size
    num_batches = file_size // batchsize  # devide in integer

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(num_batches):
        if (batch_idx+1) % 100 == 0:
            log_string('**** batch %03d ****' % (batch_idx+1))
        start_idx = batch_idx * batchsize #0
        end_idx = (batch_idx + 1) * batchsize #4

        # Augment batched point clouds by rotation and jittering
        # rotated_data = rotate_point_cloud(current_data[start_idx:end_idx, :, :])

        rotated_data = current_data[start_idx:end_idx, :, :] #batch x 1024 x 3
        jittered_data = jitter_point_cloud(rotated_data)
        jittered_data = jittered_data
        # jittered_data = apply_np_homo(jittered_data)  #jittered data here, random homogeneous by default

        feed_dict = {ops['pointclouds_pl']: jittered_data,
                     ops['pt_local_eigs_pl']: current_local[start_idx:end_idx],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training, }

        summary, step, _, loss_val, pred_val, trans_dis, compare, layer1, layer2 = sess.run([ops['train_summary_merged'], ops['step'],
                                                                                   ops['train_op'], ops['loss'],
                                                                                   ops['pred'], ops['trans_dis'],
                                                                                   ops['compare'], ops['original_pc'],
                                                                                   ops['recovered_pc']], feed_dict= feed_dict)

        # if epoch % 20 == 0 and batch_idx == 0:  # show the first batch
        #     # print('not show now')
        #
        #     show_pc.show_trans(layer1, layer2,      # origin transformed original dark, transformed light
        #                        colorset, scale=300)

        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)  # asy anotationed
        #print('pred_label:', pred_val)  # asy anotationed
        #print('truth_label:', current_label[start_idx:end_idx])  # asy anotationed
        correct = np.sum(pred_val == current_label[start_idx:end_idx])   # asy anotationed
        total_correct += correct  # asy anotationed
        total_seen += batchsize  # asy anotationed
        loss_sum += loss_val
    print('the distance between predict and ground truth is(ang and pos, training):', trans_dis)
    # the comparison between prediction and ground truth is listed here
    print('prediction is:', compare[0][0], '\n', 'ground truth' + '\n', compare[1][0])
    #print('test layer value1(training):', test_layer1[0, :])
    #print('test layer value2(training):', test_layer2[0, :])
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))  # asy anotationed


def eval_one_epoch(sess, ops, test_writer, epoch):
    """ ops: dict mapping from string to tf ops """
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(4)]  # number of class
    total_correct_class = [0 for _ in range(4)]  # number of class

    # current_label = np.squeeze(current_label)

    file_size = pc_tile.shape[0]
    num_batches = file_size // batchsize

    for batch_idx in range(num_batches):

        start_idx = batch_idx * batchsize
        end_idx = (batch_idx + 1) * batchsize

        feed_dict = {ops['pointclouds_pl']: pc_tile[start_idx:end_idx, :, :],
                     ops['labels_pl']: pc_label[start_idx:end_idx],
                     ops['pt_local_eigs_pl']: pc_local_eigs[start_idx:end_idx],
                     ops['is_training_pl']: False}

        #print('evel data:', current_data[start_idx:end_idx, :, :])

        summary, step, loss_val, pred_val, layer1, layer2, eval_posdis = sess.run([ops['train_summary_merged'], ops['step'],
                                                                                   ops['loss'], ops['pred'],
                                                                                   ops['original_pc'],
                                                                                   ops['recovered_pc'],
                                                                                   ops['trans_dis']],
                                                                                   feed_dict=feed_dict)

        test_writer.add_summary(summary, step)

    print('last step in batch of evaluation posdistance is:', eval_posdis, 'ang and trans')
        # if epoch % 5 == 0 and batch_idx == 0:  # show the first batch of every epoch
        #     layer1 *= 1000
        #     layer1 += -500+1000*np.random.random(size=(np.shape(layer1)[0], 1, 3))
        #     layer2 *= 1000
        #     layer2 += -500 + 1000 * np.random.random(size=(np.shape(layer2)[0], 1, 3))
        #     show_pc.show_trans(layer1, layer2, shade1, light1, shade2, light2, shade3, light3, shade4, light4)

            # fake_layer2 = layer1+10*np.random.random([1, 3])
            # # batch = np.shape(layer1)[0]
            #
            # fake_layer2[i, :, :] += 10*np.random.random([1, 3])
            # show_pc.show_trans(layer1, fake_layer2, shade1, dark1, shade2, dark2, shade3, dark3, shade4, dark4)
    """
    pred_val = np.argmax(pred_val, 1)
    correct = np.sum(pred_val == current_label[start_idx:end_idx])
    total_correct += correct
    total_seen += batchsize
    loss_sum += (loss_val * batchsize)
    for i in range(start_idx, end_idx):
        l = current_label[i]
        total_seen_class[l] += 1
        total_correct_class[l] += (pred_val[i - start_idx] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
    np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    """
        #print('loss equals(eval):', loss_val)


def zero_center_and_norm(batch_input):
    """

    :param batch_input: Bxnumx3
    :return:
    """
    #batch = batch_input.shape[0].value
    #num = batch_input.shape[1].value
    center, var = tf.nn.moments(batch_input, 1, keep_dims=True)  # Bx1x3 Bx1x3
    var = tf.sqrt(tf.reduce_sum(tf.square(var), axis=2, keep_dims=True))
    batch_output = batch_input-center
    batch_output /= tf.sqrt(var)
    return batch_output


def tf_euler_pos_2_homo(batch_input):
    """
    :param batch_input: batchx6  3 euler angle psi:x, theta:y, phi:z 3 position:x,y,z
    :return: transform:batch homogeneous matrix batch x 4 x 4
    """
    batch = batch_input.shape[0].value  #or tensor.get_shape().as_list()
    psi = tf.slice(batch_input, [0, 0], [batch, 1])       #all shape of: (batch,1)
    theta = tf.slice(batch_input, [0, 1], [batch, 1])
    phi = tf.slice(batch_input, [0, 2], [batch, 1])
    pos_x = tf.expand_dims(tf.slice(batch_input, [0, 3], [batch, 1]), axis=2) #all shape of: (batch,1, 1)
    pos_y = tf.expand_dims(tf.slice(batch_input, [0, 4], [batch, 1]), axis=2)
    pos_z = tf.expand_dims(tf.slice(batch_input, [0, 5], [batch, 1]), axis=2)

    rotation_x = tf.reshape(tf.concat([tf.constant(1.0, shape=[batch, 1]), tf.constant(0.0, shape=[batch, 1]),
                            tf.constant(0.0, shape=[batch, 1]), tf.constant(0.0, shape=[batch, 1]),
                            tf.cos(psi), -tf.sin(psi), tf.constant(0.0, shape=[batch, 1]), tf.sin(psi), tf.cos(psi)],
                            axis=1), shape=[batch, 3, 3])  # Bx3x3

    rotation_y = tf.reshape(tf.concat([tf.cos(theta), tf.constant(0.0, shape=[batch, 1]), tf.sin(theta),
                            tf.constant(0.0, shape=[batch, 1]), tf.constant(1.0, shape=[batch, 1]),
                            tf.constant(0.0, shape=[batch, 1]), -tf.sin(theta), tf.constant(0.0, shape=[batch, 1]),
                            tf.cos(theta)], axis=1), shape=[batch, 3, 3])

    rotation_z = tf.reshape(tf.concat([tf.cos(phi), -tf.sin(phi), tf.constant(0.0, shape=[batch, 1]),
                            tf.sin(phi), tf.cos(phi), tf.constant(0.0, shape=[batch, 1]),
                            tf.constant(0.0, shape=[batch, 1]), tf.constant(0.0, shape=[batch, 1]),
                            tf.constant(1.0, shape=[batch, 1])], axis=1), shape=[batch, 3, 3])

    rotation = tf.matmul(tf.matmul(rotation_x, rotation_y), rotation_z)  # Bx3x3
    transition = tf.concat([pos_x, pos_y,pos_z], axis=1)    # Bx3x1
    batch_out = tf.concat([rotation, transition], axis=2)  # Bx3x4
    pad = tf.concat([tf.constant(0.0, shape=[batch, 1, 3]), tf.ones([batch, 1, 1], dtype=tf.float32)], axis=2)  # Bx1x4
    batch_out = tf.concat([batch_out, pad], axis=1)  # Bx4x4
    return batch_out


def tf_quat_pos_2_homo(batch_input):
    """

    :param batch_input: batchx7 4 quaternion 3 position xyz
    :return: transformation: batch homogeneous matrix batch x 4 x 4
    """
    batch = batch_input.shape[0].value  #or tensor.get_shape().as_list()

    w = tf.slice(batch_input, [0, 0], [batch, 1])       #all shape of: (batch,1)
    x = tf.slice(batch_input, [0, 1], [batch, 1])
    y = tf.slice(batch_input, [0, 2], [batch, 1])
    z = tf.slice(batch_input, [0, 3], [batch, 1])

    pos_x = tf.expand_dims(tf.slice(batch_input, [0, 4], [batch, 1]), axis=2)  # all shape of: (batch,1, 1)
    pos_y = tf.expand_dims(tf.slice(batch_input, [0, 5], [batch, 1]), axis=2)
    pos_z = tf.expand_dims(tf.slice(batch_input, [0, 6], [batch, 1]), axis=2)

    rotation = tf.reshape(tf.concat([
        1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w,
        2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w,
        2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y], axis=1), shape=[batch, 3, 3])

    transition = tf.concat([pos_x, pos_y, pos_z], axis=1)  # Bx3x1
    batch_out = tf.concat([rotation, transition], axis=2)  # Bx3x4
    pad = tf.concat([tf.constant(0.0, shape=[batch, 1, 3]), tf.ones([batch, 1, 1], dtype=tf.float32)], axis=2) #Bx1x4
    batch_out = tf.concat([batch_out, pad], axis=1)  #Bx4x4
    return batch_out


def tf_homo_2_quat_pos(batch_input):
    """

    :param batch_input: Bx4x4
    :return: inverse homogeneous pos B x 7
    """
    batch = batch_input.shape[0].value
    q11 = tf.slice(batch_input, [0, 0, 0], [batch, 1, 1])
    q22 = tf.slice(batch_input, [0, 1, 1], [batch, 1, 1])
    q33 = tf.slice(batch_input, [0, 2, 2], [batch, 1, 1])
    q12 = tf.slice(batch_input, [0, 0, 1], [batch, 1, 1])
    q13 = tf.slice(batch_input, [0, 0, 2], [batch, 1, 1])
    q21 = tf.slice(batch_input, [0, 1, 0], [batch, 1, 1])
    q23 = tf.slice(batch_input, [0, 1, 2], [batch, 1, 1])
    q31 = tf.slice(batch_input, [0, 2, 0], [batch, 1, 1])
    q32 = tf.slice(batch_input, [0, 2, 1], [batch, 1, 1])

    R = tf.slice(batch_input, [0, 0, 0], [batch, 3, 3])
    trans = tf.slice(batch_input, [0, 0, 3], [batch, 3, 1])
    t = Qxx+Qyy+Qzz  # the trace of batch_input Bx1x1
    w = 1/2 * tf.sqrt((t+1))    # Bx1x1
    x = 1/2 * tf.sqrt((1 + Qxx - Qyy - Qzz)) * tf.sign((tf.slice(batch_input, [0, 2, 1], [batch, 1, 1])
                                                        -tf.slice(batch_input, [0, 1, 2], [batch, 1, 1])))
    y = 1/2 * tf.sqrt((1 - Qxx + Qyy - Qzz)) * tf.sign((tf.slice(batch_input, [0, 0, 2], [batch, 1, 1])
                                                        -tf.slice(batch_input, [0, 2, 0], [batch, 1, 1])))
    z = 1/2 * tf.sqrt((1 - Qxx - Qyy + Qzz)) * tf.sign((tf.slice(batch_input, [0, 1, 0], [batch, 1, 1])
                                                        -tf.slice(batch_input, [0, 0, 1], [batch, 1, 1])))

    pos = -tf.matmul(tf.transpose(R, perm=[0, 2, 1]), trans)  # B x 3 x 3 * B x 3 x 1 = B x 3 x 1 , pos = -inv(R)*trans
    batch_out = tf.concat([w, x, y, z, pos], axis=1)  # B x 7 x 1  conjugate of quaternion is the inverse of quaternion
    batch_out = tf.squeeze(batch_out)  # B x 7

    return batch_out


def apply_homo_to_pc(pc_batch_input, homo):
    """
    :param pc_batch_input: batchxnx3 tensor
    :param homo: batchx4x4
    :return:    batchxnx3 tensor
    """
    batch = pc_batch_input.shape[0].value
    num = pc_batch_input.shape[1].value
    # batch_out = tf.Variable(tf.zeros(pc_batch_input.shape), trainable=False, dtype=tf.float32, name='batch_out')   # this will cause session restore error!!!
    # batch_out = batch_out.assign(pc_batch_input)
    #
    # batch_out = tf.concat([batch_out, tf.ones((batch, num, 1))], axis=2)   # Bxnx4, add additional ones
    batch_out = tf.concat([pc_batch_input, tf.ones((batch, num, 1))], axis=2)
    batch_out = tf.transpose(batch_out, perm=[0, 2, 1])                    # Bx4xn

    batch_out = tf.matmul(homo, batch_out)  # Bx4x4 batch multiply Bx4xn, points coordinates in column vector,left-handed rotation matrix

    batch_out = tf.div(batch_out, tf.slice(batch_out, [0, 3, 0], [batch, 1, num]))  # every element divided the
    # last element to get true coordinates
    batch_out = tf.slice(batch_out, [0, 0, 0], [batch, 3, num])  # Bx3xn
    batch_out = tf.transpose(batch_out, perm=[0, 2, 1])    # Bxnx3

    return batch_out


def compute_pos_distance(batch_pos1, batch_pos2):
    """
    :param batch_pos1: B x 7
    :param batch_pos2: compare the inverse with batch_pos1 B x 7
    :return:
    """
    batch = batch_pos1.shape[0].value
    assert batch_pos1.get_shape().as_list() == batch_pos2.get_shape().as_list()
    assert batch_pos1.get_shape().as_list()[1] == 7

    homo_pos1 = tf_quat_pos_2_homo(batch_pos1)  # B x 4 x 4
    inv_homo_pos1 = tf.matrix_inverse(homo_pos1)  # B x 4 x 4
    # inverse of the quaternion of q (w x y z) is q*(w -x -y -z)
    inv_quat_pos1 = tf.concat(
        [tf.slice(batch_pos1, [0, 0], [batch, 1]), -1.0 * tf.slice(batch_pos1, [0, 1], [batch, 3]),
         tf.squeeze(tf.slice(inv_homo_pos1, [0, 0, 3], [batch, 3, 1]), axis=2)], axis=1)  # B x 7

    angle_dis = tf.minimum(tf.sqrt(tf.reduce_sum(tf.square(inv_quat_pos1[:, 0:4] - batch_pos2[:, 0:4]), axis=1)),
                           tf.sqrt(tf.reduce_sum(tf.square(inv_quat_pos1[:, 0:4] + batch_pos2[:, 0:4]),
                                                 axis=1)))  # (B, 0) todo, minimum should be taken for every one in the batch

    pos_dis = tf.sqrt(tf.reduce_sum(tf.square(inv_quat_pos1[:, 4:7]-batch_pos2[:, 4:7]), axis=1))

    mean_angle = tf.reduce_mean(angle_dis, axis=0)
    mean_pos = tf.reduce_mean(pos_dis, axis=0)
    #
    # angle_dis = tf.reduce_sum(angle_dis, axis=0)
    # pos_dis = tf.reduce_sum(pos_dis, axis=0)
    return [mean_angle, mean_pos]


def shuffle_data(data, labels, local_eig):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], local_eig[idx, ...], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def quat2mat(quat):
    """
    quat: B x 4
    return B x 3 x 3
    """
    w, x, y, z = quat
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def apply_np_homo(batch_point_cloud, homo='random'):
    """

    :param batch_point_cloud: B x n x 3 , np.array
    :param homo: if random, apply random, or B*4*4 homo matrix
    :return: B x n x 3 np.array
    """

    batch = batch_point_cloud.shape[0]
    num = batch_point_cloud.shape[1]

    #batch_out = tf.Variable(tf.zeros(pc_batch_input.shape), trainable=False, dtype=tf.float32)
    #batch_out = batch_out.assign(pc_batch_input)
    batchout = np.concatenate([batch_point_cloud, np.ones((batch, num, 1))], axis=2)
    batchout = np.transpose(batchout, (0, 2, 1))

    if homo =='random':
        quat = 20 * np.random.random((batch, 4)) - 10
        quat_pos = np.concatenate([nor4vec(quat), 20 * np.random.random((batch, 4)) - 10], axis=1)     #B x 7
        homo = np_quat_pos_2_homo(quat_pos)

    batchout = np.matmul(homo, batchout) #Bx4x4 * B x 4 x n
    batchout = np.divide(batchout, batchout[:, np.newaxis, 3, :])
    batchout = batchout[:, :3, :]
    batchout = np.transpose(batchout, (0, 2, 1))

    return batchout


def nor4vec(vector):
    """
    :param vector: B x 4
    :return: B x 4
    """
    return vector/np.linalg.norm(vector, axis=1)[:, np.newaxis]


def np_quat_pos_2_homo(batch_input):
    """

    :param batch_input: batchx7 4 quaternion 3 position xyz
    :return: transformation: batch homogeneous matrix batch x 4 x 4
    """
    batch = batch_input.shape[0]  #or tensor.get_shape().as_list()

    w = np.expand_dims(batch_input[:, 0], axis=1)   # all shape of: (batch,1)
    x = np.expand_dims(batch_input[:, 1], axis=1)
    y = np.expand_dims(batch_input[:, 2], axis=1)
    z = np.expand_dims(batch_input[:, 3], axis=1)

    pos_x = batch_input[:, 4, np.newaxis]  # all shape of: (batch,1, 1)
    pos_y = batch_input[:, 5, np.newaxis]
    pos_z = batch_input[:, 6, np.newaxis]

    rotation = np.reshape(np.concatenate([
        1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w,
        2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w,
        2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y], axis=1), [batch, 3, 3])

    transition = np.concatenate([pos_x, pos_y, pos_z], axis=1)[:, :, np.newaxis]  # Bx3x1
    batch_out = np.concatenate([rotation, transition], axis=2)  # Bx3x4
    pad = np.concatenate([np.zeros([batch, 1, 3]), np.ones([batch, 1, 1])], axis=2)  # Bx1x4
    batch_out = np.concatenate([batch_out, pad], axis=1)  # B x 4 x 4
    return batch_out


if __name__ == "__main__":

    train(model_name="object8_6my.ckpt", use_local=True)

    # inference(os.path.join('tmp', "object8_5.ckpt"), use_local=True, show_result=True, times=1, test_batchsize=1)  # test time
    # inference(os.path.join('tmp', "object8_2.ckpt"), use_local=True, show_result=False, times=1, vis_feature=True)
    # inference(os.path.join('tmp', "object8.ckpt"), use_local=True, show_result=False, times=10, vis_tsne=True, test_batchsize=50)
    # LOG_FOUT.close()