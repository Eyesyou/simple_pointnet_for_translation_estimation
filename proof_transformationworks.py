import os
import re
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import quaternion
from pyquaternion import Quaternion
# dir = '/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/noise_out lier/lab4'
# dir_list = os.listdir(dir)
# os.chdir(dir)
# # print(dir_list)
# for i in range(len(dir_list)):
#     if re.search('txt', dir_list[i]):
#         id = re.search(r'project(\d*)', dir_list[i]).group(1)
#         new_name = 'lab_project'+str(id)+'.txt'
#         os.rename(dir_list[i], new_name)


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


if __name__ == "__main__":
    A = [1, 0, 0, 0, 0, 0, 0]
    B = [1, 0, 0, 0, 0, 0, 0]

    c = Quaternion.random().elements
    C = np.concatenate([c, np.asarray([55, 45, 35])])
    D = np.concatenate([c, np.asarray([-0.5, -0.4, -0.3])])

    tsa = tf.reshape(tf.convert_to_tensor(A, dtype=tf.float32), [1, 7])
    tsb = tf.reshape(tf.convert_to_tensor(B, dtype=tf.float32), [1, 7])
    tsc = tf.reshape(tf.convert_to_tensor(C, dtype=tf.float32), [1, 7])
    tsd = tf.reshape(tf.convert_to_tensor(D, dtype=tf.float32), [1, 7])

    tse = tf_quat_pos_2_homo(tsc)  # B x 4 x 4
    tse = tf.matrix_inverse(tse)  # B x 4 x 4
    print('tse shape:', tse.shape)
    tse = tf.concat(
        [tf.slice(tsc, [0, 0], [1, 1], name='S1'), -1.0 * tf.slice(tsc, [0, 1], [1, 3], name='S2'),
         tf.squeeze(tf.slice(tse, [0, 0, 3], [1, 3, 1], name='S3'), axis=2)], axis=1)  # B x 7

    c1 = compute_pos_distance(tsa, tsb)
    c2 = compute_pos_distance(tsc, tsd)
    c3 = compute_pos_distance(tsc, tse)
    with tf.Session() as sess:
        c1, c2, c3, tsc, tse = sess.run([c1, c2, c3, tsc, tse])
        print(c1, c2, c3)
        print('tsc:', tsc)
        print('tse:', tse)
