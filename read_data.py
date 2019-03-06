import numpy as np
import h5py
from show_pc import *
import random
from mayavi import mlab
from matplotlib import pyplot as plt
from show_pc import PointCloud
from plyfile import PlyData, PlyElement
from scipy import spatial
import tensorflow as tf


def save_data(save_path='', n=5000, base_path='', use_key_feature=True):

    # tf.enable_eager_execution()
    pc_tile = np.empty(shape=(4*n, 1024, 3))
    if use_key_feature:
        pc_key_feature = np.empty(shape=(4*n, int(1024*0.1), 9))  # key feature space, 102=1024*0.1,
        # 9 for multi-scale eigen-value
        #pc_pl = tf.placeholder(tf.float32, shape=(1, 1024, 3))

    for i in range(n):

            if i % 10 == 0:
                print('reading number', i + 1, 'th lab1 point clouds')

            if use_key_feature:

                pc_tile[i, :, :] = np.expand_dims(
                    np.loadtxt(base_path+'/lab1/lab1_project'+str(i)+'.txt'), axis=0)
                pc = np.expand_dims(pc_tile[i, :, :], axis=0);  # pc = tf.convert_to_tensor(pc, dtype=tf.float32)

                # print('*****************************************')
                # print('reading point cloud cost time:{}'.format(t1 - t0))
                pc_key_eig = get_local_eig_np(pc)   # 1 x nb_keypoints x 9

                # print('*****************************************')
                # print('get local cost time:{}'.format(t2 - t1))
                #pc_key_feature[i, :, :] = np.squeeze(sess.run(pc_key_eig, feed_dict={pc_pl: pc}))
                pc_key_feature[i, :, :] = np.squeeze(pc_key_eig)
            else:
                pc_tile[i, :, :] = np.expand_dims(
                    np.loadtxt(base_path+'/lab1/lab1_project'+str(i)+'.txt'), axis=0)

            # print('-----------------------------------------')
            # print('one pc cost total:{}second'.format(te-ts))
            # print('----------------------------------------')

    for i, j in enumerate(range(n, 2*n)):
        if i % 10 == 0:
            print('reading number', i + 1, 'th lab2 point clouds')

        if use_key_feature:
            pc_tile[j, :, :] = np.expand_dims(np.loadtxt(base_path + '/lab2/lab_project' + str(i) + '.txt'),
                                              axis=0)
            pc = np.expand_dims(pc_tile[j, :, :], axis=0)

            pc_key_eig = get_local_eig_np(pc)  # 1 x nb_keypoints x 9
            #pc_key_feature[j, :, :] = np.squeeze(sess.run(pc_key_eig, feed_dict={pc_pl: pc}))
            pc_key_feature[j, :, :] = np.squeeze(pc_key_eig)
        else:
            pc_tile[j, :, :] = np.expand_dims(np.loadtxt(base_path+'/lab2/lab_project'+str(i)+'.txt'),
                                              axis=0)

    for i, j in enumerate(range(2*n, 3*n)):
        if i % 10 == 0:
            print('reading number', i + 1, 'th lab3 point clouds')

        if use_key_feature:
            pc_tile[j, :, :] = np.expand_dims(
                np.loadtxt(base_path+'/lab3/lab_project'+str(i)+'.txt'), axis=0)
            pc = np.expand_dims(pc_tile[j, :, :], axis=0)

            pc_key_eig = get_local_eig_np(pc)  # 1 x nb_keypoints x 9
            #pc_key_feature[j, :, :] = np.squeeze(sess.run(pc_key_eig, feed_dict={pc_pl: pc}))
            pc_key_feature[j, :, :] = np.squeeze(pc_key_eig)
        else:
            pc_tile[j, :, :] = np.expand_dims(
                np.loadtxt(base_path+'/lab3/lab_project'+str(i)+'.txt'), axis=0)

    for i, j in enumerate(range(3*n, 4*n)):
        if i % 10 == 0:
            print('reading number', i + 1, 'th lab4 point clouds')

        if use_key_feature:
            pc_tile[j, :, :] = np.expand_dims(np.loadtxt(
                base_path+'/lab4/lab4_project'+str(i)+'.txt'), axis=0)
            pc = np.expand_dims(pc_tile[j, :, :], axis=0)
            pc_key_eig = get_local_eig_np(pc)  # 1 x nb_keypoints x 9
            #pc_key_feature[j, :, :] = np.squeeze(sess.run(pc_key_eig, feed_dict={pc_pl: pc}))
            pc_key_feature[j, :, :] = np.squeeze(pc_key_eig)
        else:
            pc_tile[j, :, :] = np.expand_dims(np.loadtxt(
                base_path+'/lab4/lab4_project'+str(i)+'.txt'), axis=0)

    pc_label = np.concatenate(
        [np.zeros(shape=(n,)), np.ones(shape=(n,)), 2 * np.ones(shape=(n,)), 3 * np.ones(shape=(n,))], axis=0)

    train_set_shape = (4*n, 1024, 3)
    train_set_local_shape = (4*n, 102, 9)
    train_label_shape = (4*n, )

    hdf5_file = h5py.File(save_path, mode='a')
    hdf5_file.create_dataset('train_set', train_set_shape, np.float32)  # be careful about the dtype
    hdf5_file.create_dataset('train_labels', train_label_shape, np.uint8)
    hdf5_file.create_dataset('train_set_local', train_set_local_shape, np.float32)
    hdf5_file["train_set"][...] = pc_tile
    hdf5_file["train_labels"][...] = pc_label
    hdf5_file["train_set_local"][...] = pc_key_feature
    hdf5_file.close()


def read_data(h5_path=''):
    readh5 = h5py.File(h5_path, "r")  # file path
    print(readh5['train_set'][:].shape)
    print(readh5['train_labels'][:].shape)
    return readh5


class OctNode:
    def __init__(self, coordinates, size, data=None):
        assert isinstance(coordinates, np.ndarray)
        assert np.squeeze(coordinates).shape == (3,)
        self.position = coordinates
        self.size = size
        self.data = data  # the points in this node, nx3 np array
        self.ubl = None  # up back left node
        self.ubr = None  # up back right node
        self.ufl = None  # up front left node
        self.ufr = None
        self.dbl = None
        self.dbr = None
        self.dfl = None
        self.dfr = None
        self.children = [self.ubl, self.ubr, self.ufl, self.ufr, self.dbl, self.dbr, self.dfl, self.dfr]


def sample_txt_pointcloud(pc1, pc2, pc3, pc4, n=1024, save_path=''):
    pc1 = np.loadtxt(pc1)
    pc2 = np.loadtxt(pc2)
    pc3 = np.loadtxt(pc3)
    pc4 = np.loadtxt(pc4)
    np.random.shuffle(pc1)
    pc1 = pc1[0:n]
    np.random.shuffle(pc2)
    pc2 = pc2[0:n]
    pc2 = pc2*0.3
    np.random.shuffle(pc3)
    pc3 = pc3[0:n]
    np.random.shuffle(pc4)
    pc4 = pc4[0:n]
    pc4 = pc4*0.2
    stack = np.concatenate([pc1, pc2, pc3, pc4], axis=0)
    np.savetxt(save_path, stack, delimiter=' ')


def test_data(h5_path='', rand_trans=False, showinone=False):
    """
    test and show if the h5 point cloud data is generate correctly
    :param h5_path:
    :param rand_trans:
    :param showinone: show the whole batch of point clouds in one picture
    :return:
    """
    h5file = read_data(h5_path)
    trainset = h5file['train_set'][...]
    train_local = h5file['train_set_local'][...]
    print('train_local:',train_local,'train_local shape:', train_local.shape)
    if rand_trans:
        trainset += -300 + 600 * np.random.random(size=(20000, 1, 3))  # 20000 * 1024 * 3

    if showinone:
        ind = np.random.choice(20000, 20)
        points = trainset[ind, :, :]
        points = np.reshape(points, [-1, 3])
        points = PointCloud(points)
        points.show()

    for i in range(1):
        fig = plt.figure()
        for k in range(4):
            a = np.squeeze(trainset[1+k*5000, :, :])
            a = PointCloud(a)
            origin = a.show(not_show=True)
            mlab.show(origin)

            mlab.gcf().scene.parallel_projection = True  # parallel projection
            f = mlab.gcf()  # this two line for mlab.screenshot to work
            f.scene._lift()
            img = mlab.screenshot(antialiased=False)

            mlab.close()
            ax = fig.add_subplot(4, 4, 1+k*4)
            ax.imshow(img)
            ax.set_axis_off()

            a.add_noise(factor=5/100)
            noise = a.show(not_show=True)
            mlab.show(noise)
            mlab.gcf().scene.parallel_projection = True  # parallel projection
            f = mlab.gcf()  # this two line for mlab.screenshot to work
            f.scene._lift()
            img = mlab.screenshot()
            mlab.close()
            ax = fig.add_subplot(4, 4, 2+k*4)
            ax.imshow(img)
            ax.set_axis_off()

            a.add_outlier(factor=5/100)
            outlier = a.show(not_show=True)
            mlab.show(outlier)
            mlab.gcf().scene.parallel_projection = True  # parallel projection
            f = mlab.gcf()  # this two line for mlab.screenshot to work
            f.scene._lift()
            img = mlab.screenshot()
            mlab.close()
            ax = fig.add_subplot(4, 4, 4+k*4)
            ax.imshow(img)
            ax.set_axis_off()

            a = np.squeeze(trainset[1+k*5000, :, :])
            a = PointCloud(a)
            a.add_outlier(factor=5/100)
            outlier = a.show(not_show=True)
            mlab.show(outlier)
            mlab.gcf().scene.parallel_projection = True  # parallel projection
            f = mlab.gcf()  # this two line for mlab.screenshot to work
            f.scene._lift()
            img = mlab.screenshot()
            mlab.close()
            ax = fig.add_subplot(4, 4, 3+k*4)
            ax.imshow(img)
            ax.set_axis_off()

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()


def augment_data(base_path='', pc_path='', add_noise=0.05, add_outlier=0.05):
    #
    # pc = np.loadtxt('carburator.txt')   # n x 3
    # np.random.shuffle(pc)  # only shuffle the first axis
    # pc = pc[0:10000, :]
    # pc = PointCloud(pc)
    # pc.add_noise(factor=5/100)
    # pc.add_outlier(factor=5/100)
    # for i in range(5000):
    #     if i % 10 == 0:
    #         print('saving number', i+1, 'th carburator point clouds')
    #     pc.half_by_plane()
    #     np.savetxt(base_path+'/carburator/carburator_project'+str(i)+'.txt', pc.visible, delimiter=' ')
    #
    # pc = np.loadtxt('arm_monster.txt')   # n x 3
    # np.random.shuffle(pc)  # only shuffle the first axis
    # pc = pc[0:10000, :]
    # pc = PointCloud(pc)
    # pc.add_noise(factor=5/100)
    # pc.add_outlier(factor=5 / 100)
    # for i in range(5000):
    #     if i % 10 == 0:
    #         print('saving number', i+1, 'th arm_moster point clouds')
    #     pc.half_by_plane()
    #     np.savetxt(base_path+'/arm_monster/arm_monster_project'+str(i)+'.txt', pc.visible, delimiter=' ')

    # pc = np.loadtxt('Dragon_Fusion_bronze_statue.txt')   # n x 3

    plydata = PlyData.read(pc_path)
    vertex = np.asarray([list(subtuple) for subtuple in plydata['vertex'][:]])
    pc = vertex[:, 0:3]
    np.random.shuffle(pc)  # will only shuffle the first axis

    if pc.shape[0] > 10000:
        pc = pc[0:10000, :]
    pc = PointCloud(pc)

    if add_noise is not None:
        pc.add_noise(factor=add_noise)
    if add_outlier is not None:
        pc.add_outlier(factor=add_outlier)

    for i in range(0, 500):
        if i % 10 == 0:
            print('saving number', i+1, 'th lab4_project point clouds')
        try:
            pc.half_by_plane(grid_resolution=(300, 300))
            np.savetxt(base_path+'/lab_project'+str(i)+'.txt', pc.visible, delimiter=' ')
        except ValueError:
            try:
                pc.half_by_plane(grid_resolution=(300, 300))
                np.savetxt(base_path+'/lab_project'+str(i)+'.txt', pc.visible, delimiter=' ')
            except ValueError:
                pc.half_by_plane(grid_resolution=(300, 300))
                np.savetxt(base_path+'/lab_project'+str(i)+'.txt', pc.visible, delimiter=' ')


    # pc = np.loadtxt('blade.txt')   # n x 3
    # np.random.shuffle(pc)  # only shuffle the first axis
    # pc = pc[0:10000, :]
    # pc = PointCloud(pc)
    # pc.add_noise(factor=0.05)
    # pc.add_outlier(factor=0.05)
    # for i in range(5000):
    #     if i % 10 == 0:
    #         print('saving number', i+1, 'th blade point clouds')
    #     pc.half_by_plane()
    #     np.savetxt(base_path+'/blade/blade_project'+str(i)+'.txt', pc.visible, delimiter=' ')
    #
    # pc = np.loadtxt('fullbodyanya1.txt')   # n x 3
    # np.random.shuffle(pc)  # only shuffle the first axis
    # pc = pc[0:10000, :]
    # pc = PointCloud(pc)
    # pc.add_noise(factor=0.05)
    # pc.add_outlier(factor=0.05)
    # for i in range(1, 5000):
    #     if i % 10 == 0:
    #         print('saving number', i+1, 'th fullbodyanya1 point clouds')
    #     try:
    #         pc.half_by_plane()
    #         np.savetxt(base_path + '/fullbodyanya1/fullbodyanya1_project' + str(i) + '.txt', pc.visible, delimiter=' ')
    #     except ValueError:
    #         try:
    #             pc.half_by_plane()
    #             np.savetxt(base_path + '/fullbodyanya1/fullbodyanya1_project' + str(i) + '.txt', pc.visible, delimiter=' ')
    #         except ValueError:
    #             pc.half_by_plane()
    #             np.savetxt(base_path + '/fullbodyanya1/fullbodyanya1_project' + str(i) + '.txt', pc.visible,
    #                        delimiter=' ')


def get_local(point_cloud, key_pts_percentage=0.1, radius_scale=(0.1, 0.2, 0.3)):
    """
    get local feature from pointcloud using multi scale features
    :param point_cloud: Bxnx3 tensor
    :return: output B x k  feature tensor, k is the feature dimension
    """
    # print('inputshape:', point_cloud.get_shape()[:])
    batchsize = point_cloud.get_shape()[0].value
    nb_points = point_cloud.get_shape()[1].value
    nb_key_pts = tf.to_int32(nb_points*key_pts_percentage)
    min_limit = tf.reduce_min(point_cloud, axis=1)  # Bx3
    max_limit = tf.reduce_max(point_cloud, axis=1)  # Bx3
    pts_range = max_limit-min_limit  # Bx3
    pts_range = tf.sqrt(tf.reduce_sum(tf.square(pts_range), axis=1, keepdims=True))  # Bx1
    multi_radius = pts_range*radius_scale  # Bx3
    # print('multi_radius :', multi_radius)
    l2_norm = tf.reduce_sum(point_cloud*point_cloud, axis=2)  # Bxn
    l2_norm = tf.reshape(l2_norm, [batchsize, -1, 1])   # B x 1 x 1

    pts_distance_mat = tf.sqrt(l2_norm-2*(point_cloud @ tf.transpose(point_cloud, perm=(0, 2, 1))) +
                               tf.transpose(l2_norm, perm=(0, 2, 1)))  # Bxnxn
    pts_distance_mat = tf.matrix_set_diag(pts_distance_mat, tf.zeros(pts_distance_mat.shape[0:-1]), name=None)
    idx1 = tf.where((pts_distance_mat < multi_radius[:, 0, tf.newaxis, tf.newaxis]) & (pts_distance_mat > 10**-7) & ~(tf.is_nan(pts_distance_mat)))  # ? x 3
    idx2 = tf.where((pts_distance_mat < multi_radius[:, 1, tf.newaxis, tf.newaxis]) & (pts_distance_mat > 10**-7) & ~(tf.is_nan(pts_distance_mat)))  # ? x 3
    idx3 = tf.where((pts_distance_mat < multi_radius[:, 2, tf.newaxis, tf.newaxis]) & (pts_distance_mat > 10**-7) & ~(tf.is_nan(pts_distance_mat)))  # ? x 3

    pts_r_neighbor_idx1 = tf.py_func(get_pts_nei, [idx1, batchsize, nb_points], tf.int32)  # b x nb_key_pts x max_nb_pts_neighbors
    pts_r_neighbor_idx2 = tf.py_func(get_pts_nei, [idx2, batchsize, nb_points], tf.int32)  # b x nb_key_pts x max_nb_pts_neighbors
    pts_r_neighbor_idx3 = tf.py_func(get_pts_nei, [idx3, batchsize, nb_points], tf.int32)

    pts_r_cov = tf.py_func(get_pts_cov, [point_cloud, pts_r_neighbor_idx2], tf.float32)  # b x n x 3 x 3
    eigen_val, _ = tf.linalg.eigh(pts_r_cov)  # b x n x 3
    _, key_pts_idx = tf.nn.top_k(eigen_val[:, :, 0], k=nb_key_pts, sorted=False)  # b x nb_key_pts

    key_pts_idx = tf.expand_dims(key_pts_idx, axis=-1)  # bx nb_key_pts x 1
    the_range = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(batchsize), axis=-1), (1, nb_key_pts)), axis=-1)  # bx nb x 1

    key_pts_idxs = tf.concat([the_range, key_pts_idx], axis=-1)   # b x nb_key_pts x 2  (batch, row)
    # print('key_pts_idx shape:', key_pts_idx.shape)
    key_eig_val = tf.gather_nd(eigen_val, key_pts_idxs)  # b x nb_key_pts x 3

    # key_pts_cov2 = tf.gather_nd(pts_r_cov, key_pts_idxs)  # b x nb_key_pts x 3 x 3
    key_pts_cov1 = tf.py_func(get_pts_cov, [point_cloud, tf.gather_nd(pts_r_neighbor_idx1, key_pts_idxs)], tf.float32)
    key_pts_cov3 = tf.py_func(get_pts_cov, [point_cloud, tf.gather_nd(pts_r_neighbor_idx3, key_pts_idxs)], tf.float32)
    key_eig_val2 = key_eig_val
    key_eig_val1, _ = tf.linalg.eigh(key_pts_cov1)  # b x nb_key_pts x 3
    key_eig_val3, _ = tf.linalg.eigh(key_pts_cov3)  # b x nb_key_pts x 3
    concat = tf.concat([key_eig_val1, key_eig_val2, key_eig_val3], axis=-1)  # b x nb_key_pts x 9
    concat = tf.expand_dims(concat, axis=-1)                        # b x nb_key_pts x 9 x 1
    concat = tf.reshape(concat, [batchsize, nb_key_pts, 9, 1])      # b x nb_key_pts x 9 x 1
    net = tf.layers.conv2d(inputs=concat, filters=128, kernel_size=[1, 9])  # b x nb_key_pts x 1 x 128
    net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=[1, 1])  # b x nb_key_pts x 1 x 256
    net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=[1, 1])  # b x nb_key_pts x 1 x 256

    net = tf.layers.max_pooling2d(net, pool_size=(int(nb_points*key_pts_percentage), 1), strides=1)  # b x 1 x 1 x 256
    net = tf.reshape(net, [batchsize, -1])  # b x 256
    return net


def get_local_eig(point_cloud, key_pts_percentage=0.1, radius_scale=(0.1, 0.2, 0.3)):
    """

    :param point_cloud: Bxnx3 tensor
    :param key_pts_percentage:
    :param radius_scale:  multi-scale radius ratio
    :return: B x nb_points x 9  tensor
    """

    # print('inputshape:', point_cloud.get_shape()[:])
    batchsize = point_cloud.get_shape()[0].value
    nb_points = point_cloud.get_shape()[1].value
    nb_key_pts = tf.to_int32(nb_points*key_pts_percentage)
    min_limit = tf.reduce_min(point_cloud, axis=1)  # Bx3
    max_limit = tf.reduce_max(point_cloud, axis=1)  # Bx3
    pts_range = max_limit-min_limit  # Bx3
    pts_range = tf.sqrt(tf.reduce_sum(tf.square(pts_range), axis=1, keepdims=True))  # Bx1
    multi_radius = pts_range*radius_scale  # Bx3
    # print('multi_radius :', multi_radius)
    l2_norm = tf.reduce_sum(point_cloud*point_cloud, axis=2)  # Bxn
    l2_norm = tf.reshape(l2_norm, [batchsize, -1, 1])   # B x 1 x 1

    pts_distance_mat = tf.sqrt(l2_norm-2*(point_cloud @ tf.transpose(point_cloud, perm=(0, 2, 1))) +
                               tf.transpose(l2_norm, perm=(0, 2, 1)))  # Bxnxn
    pts_distance_mat = tf.matrix_set_diag(pts_distance_mat, tf.zeros(pts_distance_mat.shape[0:-1]), name=None)

    # print('distance compute cost:{} second time'.format(t1-t0))
    idx1 = tf.where((pts_distance_mat < multi_radius[:, 0, tf.newaxis, tf.newaxis]) & (pts_distance_mat > 10**-7) & ~(tf.is_nan(pts_distance_mat)))  # ? x 3
    idx2 = tf.where((pts_distance_mat < multi_radius[:, 1, tf.newaxis, tf.newaxis]) & (pts_distance_mat > 10**-7) & ~(tf.is_nan(pts_distance_mat)))  # ? x 3
    idx3 = tf.where((pts_distance_mat < multi_radius[:, 2, tf.newaxis, tf.newaxis]) & (pts_distance_mat > 10**-7) & ~(tf.is_nan(pts_distance_mat)))  # ? x 3

    # print('index compute cost:{} second time'.format(t2 - t1))
    # """

    # """
    batchsize = tf.convert_to_tensor(batchsize); nb_points = tf.convert_to_tensor(nb_points)  # ....................
    pts_r_neighbor_idx1 = tf.py_func(get_pts_nei, [idx1, batchsize, nb_points], tf.int32)  # b x nb_key_pts x max_nb_pts_neighbors
    pts_r_neighbor_idx2 = tf.py_func(get_pts_nei, [idx2, batchsize, nb_points], tf.int32)  # b x nb_key_pts x max_nb_pts_neighbors
    pts_r_neighbor_idx3 = tf.py_func(get_pts_nei, [idx3, batchsize, nb_points], tf.int32)

    # print('neighbor compute cost:{} second time'.format(t3 - t2))
    pts_r_cov = tf.py_func(get_pts_cov, [point_cloud, pts_r_neighbor_idx2], tf.float32)  # b x n x 3 x 3

    # print('mid-sclae covariance mat compute cost:{} second time'.format(t4 - t3))

    eigen_val, _ = tf.linalg.eigh(pts_r_cov)  # b x n x 3
    _, key_pts_idx = tf.nn.top_k(eigen_val[:, :, 0], k=nb_key_pts, sorted=False)  # b x nb_key_pts

    key_pts_idx = tf.expand_dims(key_pts_idx, axis=-1)  # bx nb_key_pts x 1
    the_range = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(batchsize), axis=-1), (1, nb_key_pts)), axis=-1)  # bx nb x 1

    key_pts_idxs = tf.concat([the_range, key_pts_idx], axis=-1)   # b x nb_key_pts x 2  (batch, row)
    # print('key_pts_idx shape:', key_pts_idx.shape)
    key_eig_val = tf.gather_nd(eigen_val, key_pts_idxs)  # b x nb_key_pts x 3

    # key_pts_cov2 = tf.gather_nd(pts_r_cov, key_pts_idxs)  # b x nb_key_pts x 3 x 3
    key_pts_cov1 = tf.py_func(get_pts_cov, [point_cloud, tf.gather_nd(pts_r_neighbor_idx1, key_pts_idxs)], tf.float32)
    key_pts_cov3 = tf.py_func(get_pts_cov, [point_cloud, tf.gather_nd(pts_r_neighbor_idx3, key_pts_idxs)], tf.float32)

    # print('multi-sclae covariance mat compute cost:{} second time'.format(t5 - t4))
    key_eig_val2 = key_eig_val
    key_eig_val1, _ = tf.linalg.eigh(key_pts_cov1)  # b x nb_key_pts x 3
    key_eig_val3, _ = tf.linalg.eigh(key_pts_cov3)  # b x nb_key_pts x 3
    concat = tf.concat([key_eig_val1, key_eig_val2, key_eig_val3], axis=-1)  # b x nb_key_pts x 9

    # print('multi-sclae eigens compute cost:{} second time'.format(t6 - t5))
    return concat


def get_local_eig_np(point_cloud, key_pts_percentage=0.1, radius_scale=(0.1, 0.2, 0.3)):
    """

    :param point_cloud:   Bxnx3  np array
    :param key_pts_percentage:
    :param radius_scale:
    :return: B x nb_key_pts x 9 eigen_values
    """
    # print('inputshape:', point_cloud.get_shape()[:])

    batchsize = point_cloud.shape[0]
    nb_points = point_cloud.shape[1]
    nb_key_pts = int(nb_points * key_pts_percentage)
    min_limit = np.min(point_cloud, axis=1)  # Bx3
    max_limit = np.max(point_cloud, axis=1)  # Bx3
    pts_range = max_limit - min_limit  # Bx3
    pts_range = np.sqrt(np.sum(np.square(pts_range), axis=1, keepdims=True))  # Bx1
    multi_radius = pts_range * radius_scale  # Bx3
    # print('multi_radius :', multi_radius)

    max_nb_nei_pts = [0, 0, 0]

    # get max length
    for i in range(batchsize):
        pc = np.squeeze(point_cloud[i])
        pc = PointCloud(pc)
        pc.generate_r_neighbor(rate=0.05)
        idx1 = pc.point_rneighbors  # n x ?
        pc.generate_r_neighbor(rate=0.1)
        idx2 = pc.point_rneighbors  # n x ?
        pc.generate_r_neighbor(rate=0.2)
        idx3 = pc.point_rneighbors  # n x ?
        current = (idx1.shape[1], idx2.shape[1], idx3.shape[1])

        max_nb_nei_pts = np.max(np.asarray([max_nb_nei_pts, current]), axis=0)

        """
        pc = np.squeeze(point_cloud[i])
        kdtree = spatial.KDTree(pc)
        idx1 = kdtree.query_ball_point(pc, multi_radius[i, 0])
        idx2 = kdtree.query_ball_point(pc, multi_radius[i, 1])
        idx3 = kdtree.query_ball_point(pc, multi_radius[i, 2]) 
        print('c length:', idx1.__len__())
        length1 = len(max(idx1, key=len))
        length2 = len(max(idx2, key=len))
        length3 = len(max(idx3, key=len))
        current = (length1, length2, length3)
        max_nb_nei_pts = np.max(np.asarray([max_nb_nei_pts, current]), axis=0)
        print('max_nb:', max_nb_nei_pts)
    """
    np_arr1 = np.empty((batchsize, nb_points, max_nb_nei_pts[0]))  # b x n x l1
    np_arr2 = np.empty((batchsize, nb_points, max_nb_nei_pts[1]))  # b x n x l2
    np_arr3 = np.empty((batchsize, nb_points, max_nb_nei_pts[2]))  # b x n x l3

    np_arr1[:] = np.nan
    np_arr2[:] = np.nan
    np_arr3[:] = np.nan

    for i in range(batchsize):
        pc = np.squeeze(point_cloud[i])
        pc = PointCloud(pc)
        pc.generate_r_neighbor(rate=0.05)
        idx1 = pc.point_rneighbors  # n x ?
        pc.generate_r_neighbor(rate=0.1)
        idx2 = pc.point_rneighbors  # n x ?
        pc.generate_r_neighbor(rate=0.2)
        idx3 = pc.point_rneighbors  # n x ?
        """
        kdtree = spatial.KDTree(pc)
        idx1 = kdtree.query_ball_point(pc, multi_radius[i, 0])
        idx2 = kdtree.query_ball_point(pc, multi_radius[i, 1])
        idx3 = kdtree.query_ball_point(pc, multi_radius[i, 2])
        print('c length:', idx1.__len__())
        length1 = len(max(idx1, key=len))
        length2 = len(max(idx2, key=len))
        length3 = len(max(idx3, key=len))

        print('length1 length2 length3:', length1, length2, length3)
        """

        for j, k in enumerate(idx1):
            np_arr1[i][j][0:len(k)] = k
        for j, k in enumerate(idx2):
            np_arr2[i][j][0:len(k)] = k
        for j, k in enumerate(idx3):
            np_arr3[i][j][0:len(k)] = k

    np_arr2.astype(int)

    pts_r_cov = get_pts_cov(point_cloud, np_arr2)  # np_arr2 is b x n  b x n x 3 x 3

    eigen_val, _ = np.linalg.eigh(pts_r_cov)  # b x n x 3 orderd

    idx = np.argpartition(eigen_val[:, :, 0], nb_key_pts, axis=1)

    # print(eigen_val[idx])
    key_idx = idx[:, 0:nb_key_pts]

    # print('key points coordinates:', point_cloud[idx, :], 'shape:', point_cloud[idx, :].shape)

    # b_dix = np.indices((batchsize, nb_key_pts))[1]  # b x nb_key
    # print('b_dix: ', b_dix, 'shape:', b_dix.shape)
    # batch_idx = np.concatenate([np.expand_dims(b_dix, axis=-1), np.expand_dims(idx, axis=-1)], axis=-1)  # b x nb_key x 2

    key_eig_val = np.empty((batchsize, nb_key_pts, 3))  # b x nb_keypoints x 3
    for i in range(batchsize):
        key_eig_val[i, :, :] = eigen_val[i, key_idx[i, :], :]

    np_key_arr1 = np.empty((batchsize, nb_key_pts, np_arr1.shape[2]))                                     # np_arr1: b x n x nei1  to  b x  nb_key x  nei1
    np_key_arr3 = np.empty((batchsize, nb_key_pts, np_arr3.shape[2]))
    np_key_arr1[:] = np.nan
    np_key_arr3[:] = np.nan
    for i in range(batchsize):
        np_key_arr1[i, :, :] = np_arr1[i, key_idx[i, :], :]
        np_key_arr3[i, :, :] = np_arr3[i, key_idx[i, :], :]

    key_pts_cov1 = get_pts_cov(point_cloud, np_key_arr1)  #    np_arr1: b x nb_key x nei1     b x nb_key x 3 x 3
    key_pts_cov3 = get_pts_cov(point_cloud, np_key_arr3)  #    np_arr3: b x nb_key x nei3     b x nb_key x 3 x 3

    key_eig_val2 = key_eig_val  # ordered
    key_eig_val1, _ = np.linalg.eigh(key_pts_cov1)  # b x nb_key_pts x 3 ordered
    key_eig_val3, _ = np.linalg.eigh(key_pts_cov3)  # b x nb_key_pts x 3 ordered

    concat = np.concatenate((key_eig_val1, key_eig_val2, key_eig_val3), axis=-1)  # b x nb_key_pts x 9

    return concat


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

            this_pt_nei_idx = pts_r_neirhbor_idx[i, j, :][pts_r_neirhbor_idx[i, j, :] >= 0].astype(np.int32)
            #print('this_pt_nei_idx:', this_pt_nei_idx)
            neighbor_pts = pc[i, this_pt_nei_idx, :]  # 1 x nb x 3
            #print('neighbor_pts shape:', neighbor_pts.shape)
            if neighbor_pts.size == 0:
                cov[i, j, :, :] = np.eye(3)
            else:
                cov[i, j, :, :] = np.cov(neighbor_pts, rowvar=False)

    return np.float32(cov)


if __name__ == "__main__":
    # save_data(save_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/project_data.h5',
    #           base_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece')

    # read_data(h5_path='/home/sjtu/Documents/ASY/point_cloud_deep_learning/simple_pointnet for translation estimation/project_data.h5')
    # sample_txt_pointcloud('/home/sjtu/Documents/ASY/point_cloud_deep_learning/simple_pointnet for translation estimation/arm_monster.txt',
    #                       '/home/sjtu/Documents/ASY/point_cloud_deep_learning/simple_pointnet for translation estimation/blade.txt',
    #                       '/home/sjtu/Documents/ASY/point_cloud_deep_learning/simple_pointnet for translation estimation/carburator.txt',
    #                       '/home/sjtu/Documents/ASY/point_cloud_deep_learning/simple_pointnet for translation estimation/fullbodyanya1.txt',
    #                       save_path='/home/sjtu/Documents/ASY/point_cloud_deep_learning/simple_pointnet for translation estimation/monsterbladecaranya.txt')
    # stack_4 = np.loadtxt('/home/sjtu/Documents/ASY/point_cloud_deep_learning/simple_pointnet for translation estimation/cowbunnyprojectorshaft.txt')    # monsterbladecaranya
    # pc1 = PointCloud(stack_4[0:1024, :])
    # pc1 = PointCloud(stack_4[1024:2048, :])
    # pc1 = PointCloud(stack_4[2048:3072, :])
    # pc1 = PointCloud(stack_4[3072:4096, :])
    # stack_4 = np.loadtxt(
    #           '/home/sjtu/Documents/ASY/point_cloud_deep_learning/simple_pointnet for translation estimation/monsterbladecaranya.txt')
    # pc1 = PointCloud(stack_4[0:1024, :])
    # pc1 = PointCloud(stack_4[1024:2048, :])
    # pc1 = PointCloud(stack_4[2048:3072, :])
    # pc1 = PointCloud(stack_4[3072:4096, :])

    augment_data(base_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/clean sample/lab4',
                 pc_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/clean sample/lab4/final.ply',
                 add_noise=None, add_outlier=None)
    # test_data(h5_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/project_data.h5', rand_trans=False, showinone=False)
    # pc = np.loadtxt('/media/sjtu/software/ASY/pointcloud/lab_workpice.txt')
    # pc = PointCloud(pc)
    # pc.show()

