import tensorflow as tf
import numpy as np
import sys
import os
import tf_util


def compute_pos_distance(batch_pos1,batch_pos2):
    """

    :param batch_pos1:
    :param batch_pos2:
    :return:
    """
    assert batch_pos1.get_shape().as_list()==batch_pos2.get_shape().as_list()
    assert batch_pos1.get_shape().as_list()[1]==6

    angle_dis = tf.sqrt(tf.reduce_sum(tf.square(batch_pos1[:,0:3]-batch_pos2[:,0:3]),axis=1))
    pos_dis = tf.sqrt(tf.reduce_sum(tf.square(batch_pos1[:,3:6]-batch_pos2[:,3:6]),axis=1))
    mean_angle = tf.reduce_min(angle_dis, axis=0)
    mean_pos = tf.reduce_min(pos_dis, axis=0)

    return [mean_angle, mean_pos]


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
    #create T_11 to T_34:
    #for i in range(1,4):
    #    for j in range(1,5):
    #        t = 'T_'+str(i) + str(j)+'=tf.Variable(tf.zeros([batch,4,4]),trainable=False)'
    #        exec(t)
    #T_44 = tf.Variable(tf.zeros([batch,4,4]))

    rotation_x = tf.Variable(initial_value=tf.reshape(tf.concat([tf.constant(1.0, shape=[batch,1]), tf.constant(0.0, shape=[batch, 1]),
                             tf.constant(0.0, shape=[batch, 1]), tf.constant(0.0, shape=[batch, 1]), tf.cos(psi), -tf.sin(psi),
                             tf.constant(0.0, shape=[batch, 1]), tf.sin(psi), tf.cos(psi)], axis=1), shape=[batch, 3, 3]), trainable=False) #Bx3x3

    rotation_y = tf.Variable(initial_value=tf.reshape(tf.concat([tf.cos(theta), tf.constant(0.0, shape=[batch, 1]), -tf.sin(theta),
                             tf.constant(1.0, shape=[batch, 1]), tf.constant(0.0, shape=[batch, 1]), tf.constant(0.0, shape=[batch, 1]),
                             tf.sin(theta), tf.constant(0.0, shape=[batch, 1]), tf.cos(theta)], axis=1), shape=[batch, 3, 3]), trainable=False)

    rotation_z = tf.Variable(initial_value=tf.reshape(tf.concat([tf.cos(phi), -tf.sin(phi), tf.constant(0.0, shape=[batch, 1]),
                           tf.sin(phi), tf.cos(phi), tf.constant(0.0, shape=[batch, 1]),
                           tf.constant(1.0, shape=[batch, 1]), tf.constant(0.0, shape=[batch, 1]),
                           tf.constant(0.0, shape=[batch, 1])], axis=1), shape=(batch, 3, 3)), trainable=False)

    rotation = tf.matmul(tf.matmul(rotation_x, rotation_y), rotation_z)  # B x 3 x 3
    transition = tf.concat([pos_x, pos_y,pos_z], axis=1)    # Bx3x1
    batch_out = tf.concat([rotation, transition], axis=2)  # Bx3x4
    pad = tf.concat([tf.constant(0.0, shape=[batch, 1, 3]), tf.ones([batch, 1, 1], dtype=tf.float32)], axis=2)  # Bx1x4
    batch_out = tf.concat([batch_out, pad], axis=1)  # Bx4x4
    return batch_out


def homo_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3
        Return:
            Transformation matrix of size BX7 ,K=3 in ordinary"""

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)    # BxNx3x1
    net = input_image * 1   # shrink it only if you want to minimize the input

    test_layer_output1 = net # BX1024X3x1

    #net = tf_util.conv2d(net, 32, [1, 3],
    #                     padding='SAME', stride=[1, 1],
    #                    bn=True, is_training=is_training,
    #                     scope='hconv111', bn_decay=bn_decay, weight_decay=0.0)     #BxNx1x64 4 variables of weight bises beta gamma

    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[1, 3])   # Bx1024x1x64
    test_layer_output2 = net

    #net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
    #                     scope='hconv12', bn_decay=bn_decay, weight_decay=0.0)
    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[1, 1], activation=tf.nn.relu)
    net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=[1, 1], activation=tf.nn.relu)

    #net = tf_util.conv2d(net, 1024, [1, 1], padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
    #                     scope='hconv13', bn_decay=bn_decay, weight_decay=0.0)
    # net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=[1, 1], activation=tf.nn.relu)
    # net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=[1, 1], activation=tf.nn.relu)
    # net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=[1, 1], activation=tf.nn.relu)
    # net = tf.layers.conv2d(inputs=net, filters=1024, kernel_size=[1, 1], activation=tf.nn.relu)
    point_wise = tf.layers.conv2d(inputs=net, filters=1024, kernel_size=[1, 1], activation=tf.nn.relu)   # Bx1024x1x1024
    net = tf_util.max_pool2d(point_wise, [num_point, 1], padding='VALID', scope='hmaxpool')  # Bx1024x1x1024
    
    net = tf.reshape(net, [batch_size, -1])  # B x X

    #net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='tfc20',
    #                              bn_decay=bn_decay, weight_decay=0.0)


    #net = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, scope='tfc21',
    #                              bn_decay=bn_decay, weight_decay=0.0)

    #transform_7 = tf_util.fully_connected(net, 7, bn=True, is_training=is_training, scope='tfc22',
    #                                      bn_decay=bn_decay, weight_decay = 0.0)

    net = tf.layers.dense(net, 512)
    net = tf.layers.dense(net, 256)
    net = tf.layers.dense(net, 64)  # Bx64
    net = tf.expand_dims(net, axis=1)
    net = tf.expand_dims(net, axis=1) # Bx1x1x64
    net = tf.tile(net, [1, 1024, 1, 1])  # BX1024X1X64
    net = tf.concat([net, point_wise], axis=-1)  #BX1024X1X1088
    net = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='hmaxpool2')  # Bx1024x1x1024
    net = tf.reshape(net, [batch_size, -1])  # B x X
    net = tf.layers.dense(net, 512)
    net = tf.layers.dense(net, 256)
    net = tf.layers.dense(net, 64)  # Bx64
    net = tf.layers.dense(net, 7)   # Bx7
    transform_7 = tf.layers.dense(net, 7)

    #reshape = tf.reshape(net, (batch_size, 1024*3)) # B x (1024*3)
    #transform_7 = tf.matmul(reshape, 1*tf.ones((1024*3, 7))) #1024*3 x 7

    transform_7 = transform_7 + 0.00001  # avoid zero dividing
    transform_7 = tf.concat([tf.divide(tf.slice(transform_7, [0, 0], [batch_size, 4]),
                            tf.norm(tf.slice(transform_7, [0, 0], [batch_size, 4]), axis=1, keep_dims=True)),
                            tf.slice(transform_7, [0, 4], [batch_size, 3])], axis=1)     # normalize it


    #transform_7 = tf.concat([tf.ones([batch_size, 1]), tf.zeros([batch_size, 3]), tf.slice(transform_7, [0, 4], [batch_size, 3])], axis=1)

    return transform_7, test_layer_output1, test_layer_output2  # Bx7


def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1, 3],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='itconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='itconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='itconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='itmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='itfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='itfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        assert(K==3)
        weights = tf.get_variable('iweights', [256, 3*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('ibiases', [3*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform


def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK       addressed as batch_size,num_point,
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    net = tf_util.conv2d(inputs, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='ftconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='ftconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='ftconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K],
                                  initializer=tf.truncated_normal_initializer(0.0,0.01),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])  # BxKxK

    return transform
