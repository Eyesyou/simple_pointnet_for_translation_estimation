import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import matplotlib.pyplot as plt
import show_pc
from mpl_toolkits.mplot3d import Axes3D


def plotit(pc_tile):


    ran_pos = tf.concat([tf.random_uniform([16, 4], minval=-10, maxval=10),
                             tf.random_uniform([16, 3], minval=-10, maxval=10)],
                            axis=1)  # random_ROTATION_and POSITION, batch x 7

    ran_pos = tf.concat([tf.divide(tf.slice(ran_pos, [0, 0], [16, 4]),
                             tf.norm(tf.slice(ran_pos, [0, 0], [16, 4]), axis=1, keep_dims=True)),
                             tf.slice(ran_pos, [0, 4], [16, 3])], axis=1)  # normalize random pose

    ran_homo = tf_quat_pos_2_homo(ran_pos)  # Bx4x4

    point_cloud_jitterd = apply_homo_to_pc(pc_tile, ran_homo)  # add this to input batch for data_augmentation

    result = sess.run(point_cloud_jitterd)  #16 x 1024 x 3
    result = result[0:128, :, :]
    show_pc.show_custom(result)


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

    pos_x = tf.expand_dims(tf.slice(batch_input, [0, 4], [batch, 1]), axis=2) #all shape of: (batch,1, 1)
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


def apply_homo_to_pc(pc_batch_input, homo):
    """
    :param pc_batch_input: batchxnx3 tensor
    :param homo: batchx4x4
    :return:    batchxnx3 tensor
    """
    batch = pc_batch_input.shape[0].value
    num = pc_batch_input.shape[1].value
    batch_out = tf.Variable(tf.zeros(pc_batch_input.shape), trainable=False, dtype=tf.float32)
    batch_out = batch_out.assign(pc_batch_input)

    batch_out = tf.concat([batch_out, tf.ones((batch, num, 1))], axis=2)   # Bxnx4, add additional ones
    batch_out = tf.transpose(batch_out, perm=[0, 2, 1])                    # Bx4xn

    batch_out = tf.matmul(homo, batch_out)  # Bx4x4 batch multiply Bx4xn, points coordinates in column vector

    batch_out = tf.div(batch_out, tf.slice(batch_out, [0, 3, 0], [batch, 1, num]))  # every element divided the
    # last element to get true coordinates
    batch_out = tf.slice(batch_out, [0, 0, 0], [batch, 3, num])  # Bx3xn
    batch_out = tf.transpose(batch_out, perm=[0, 2, 1])    # Bxnx3

    return batch_out



def plot(pc_tile):
    ran_pos = tf.concat([tf.random_uniform([16*4, 4], minval=-10, maxval=10),
                             tf.random_uniform([16*4, 3], minval=-10, maxval=10)],
                            axis=1)  # random_ROTATION_and POSITION, batch x 7

    ran_pos = tf.concat([tf.divide(tf.slice(ran_pos, [0, 0], [16*4, 4]),
                             tf.norm(tf.slice(ran_pos, [0, 0], [16*4, 4]), axis=1, keep_dims=True)),
                             tf.slice(ran_pos, [0, 4], [16*4, 3])], axis=1)  # normalize random pose

    pos1=[9.99997735e-01, 1.90134009e-03, 6.26082008e-04, -6.98232034e-04, 8.13794422e+00, -2.02135777e+00, -8.69961166e+00]
    pos2=[1., 0., 0., 0., -8.1843853, 2.03576279, 8.75728226]

    ran_homo = tf_quat_pos_2_homo(ran_pos)  # Bx4x4

    point_cloud_jitterd = apply_homo_to_pc(pc_tile, ran_homo)  # add this to input batch for data_augmentation

    result = sess.run(point_cloud_jitterd)  #128*4 x 1024 x 3
    result = result[0:128, :, :]
    show_pc.show_custom(result)



if __name__ =="__main__":
    pc = np.loadtxt('cowbunnyprojectorshaft.txt')
    pc = np.reshape(pc, (4, 1024, 3))

    tile_size = 1  # 4
    pc_tile = np.tile(pc, (tile_size, 1, 1))  # (4*4) x 1024 x 3

    sess = tf.Session()
    pc_tile = tf.convert_to_tensor(pc_tile, dtype=tf.float32)

    pc_tile = tf.slice(pc_tile, (3, 0, 0), [1, 1024, 3])
    print(tf.shape(pc_tile))
    print(pc_tile.get_shape().as_list())
    with sess as se:
        #plotit(pc_tile)
        pc_tile = sess.run(pc_tile)
        print(pc_tile.shape)
        show_pc.show_all(pc_tile,color='y')