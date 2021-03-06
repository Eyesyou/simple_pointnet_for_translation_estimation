import numpy as np
import h5py
from show_pc import *
import random
import threading
from mayavi import mlab
from matplotlib import pyplot as plt
from show_pc import PointCloud
from plyfile import PlyData, PlyElement
from scipy import spatial
from pointcloud_sample import resolution_kpts
from copy import deepcopy
import tensorflow as tf


def save_data(save_path='', base_path='', n=5000, use_key_feature=True, train_data=True,
              nb_types=4, show_result=False, normalize=True, shuffle=True):
    """
    transform the txt point clouds into h5py dataset for simplicity. data augmentation of projection is implemented here
    :param save_path:
    :param n:
    :param train_data whether it is training data or it is test data. if its testdata, label is random.
    :param base_path:  path contains txt or ply point cloud data
    :param use_key_feature: if you want to use the local key features
    :param nb_types: number of classes of used object
    :return:
    """
    compute_time = []
    if train_data:
        pc_tile = np.empty(shape=(nb_types * n, 1024, 3))
        if use_key_feature:
            pc_key_feature = np.empty(shape=(nb_types*n, int(1024*0.1), 9))  # key feature space, 102=1024*0.1,
            # 9 for multi-scale eigen-value
            #pc_pl = tf.placeholder(tf.float32, shape=(1, 1024, 3))

        for k in range(nb_types):  # number of types of  objects model, test data can ignore type label
            for i, j in enumerate(range(k*n, (k+1)*n)):

                    if i % 10 == 0:
                        print('reading number', i + 1, 'th lab'+str(k+1)+' point clouds')

                    if use_key_feature:
                        pc = np.loadtxt(base_path+'/lab'+str(k+1)+'/lab_project'+str(i)+'.txt')  # pc = tf.convert_to_tensor(pc, dtype=tf.float32)
                        pc = PointCloud(pc)
                        if normalize:
                            pc.normalize()   # partial point cloud should not normalize

                        expand = np.expand_dims(pc.position, axis=0)
                        pc_tile[j, :, :] = expand
                        # print('*****************************************')
                        # print('reading point cloud cost time:{}'.format(t1 - t0))

                        pc_key_eig = get_local_eig_np(expand, useiss=False)   # 1 x nb_keypoints x 9

                        # print('*****************************************')
                        # print('get local cost time:{}'.format(t2 - t1))
                        #pc_key_feature[i, :, :] = np.squeeze(sess.run(pc_key_eig, feed_dict={pc_pl: pc}))
                        pc_key_feature[j, :, :] = np.squeeze(pc_key_eig)
                    else:

                        pc_tile[j, :, :] = np.expand_dims(
                            np.loadtxt(base_path+'/lab'+str(k+1)+'/lab_project'+str(i)+'.txt'), axis=0)

                    # print('-----------------------------------------')
                    # print('one pc cost total:{}second'.format(te-ts))
                    # print('----------------------------------------')

        pc_label = np.tile(np.arange(nb_types), (n, 1)).reshape((-1,), order='F')
        train_set_shape = (nb_types * n, 1024, 3)
        train_set_local_shape = (nb_types * n, 102, 9)
        train_label_shape = (nb_types * n,)
    else:
        ply_list = [base_path + '/' + i for i in os.listdir(base_path) if os.path.splitext(i)[1] == '.ply' or
                    os.path.splitext(i)[1] == '.txt']
        n = len(ply_list)
        less_than_th = []
        for i,j in enumerate(ply_list):
            pc = PointCloud(j)
            if pc.nb_points < 1024:
                less_than_th.append(i)

        n = n-len(less_than_th)
        print("there are: ,",n,' point clouds with # of points available')
        pc_label = np.arange(n)
        train_set_shape = (n, 1024, 3)
        train_set_local_shape = (n, 102, 9)
        train_label_shape = (n,)

        pc_tile = np.empty(shape=(n, 1024, 3))
        pc_key_feature = np.empty(shape=(n, int(1024 * 0.1), 9))  # key feature space, 102=1024*0.1,

        for i, j in enumerate(ply_list):
            if i not in less_than_th:
                print(j)
                start_time = time.clock()
                mypc = PointCloud(j)
                if mypc.nb_points > 1024:
                    mypc.down_sample(number_of_downsample=1024)
                    if normalize:
                        mypc.normalize()
                    expand = np.expand_dims(mypc.position, axis=0)
                    pc_tile[i, :, :] = expand
                    pc_key_eig = get_local_eig_np(expand, useiss=False)
                    end_time = time.clock()
                    compute_time.append([end_time-start_time])
                    if use_key_feature:
                        pc_key_feature[i, :, :] = np.squeeze(pc_key_eig)

    hdf5_file = h5py.File(save_path, mode='a')
    hdf5_file.create_dataset('train_set', train_set_shape, np.float32)  # be careful about the dtype
    hdf5_file.create_dataset('train_labels', train_label_shape, np.uint8)
    hdf5_file.create_dataset('train_set_local', train_set_local_shape, np.float32)
    if shuffle:

        idx = np.arange(np.shape(pc_tile)[0])
        np.random.shuffle(idx)
        pc_tile = pc_tile[idx, ...]
        pc_label = pc_label[idx, ...]
        pc_key_feature =pc_key_feature[idx, ...]

    hdf5_file["train_set"][...] = pc_tile
    hdf5_file["train_labels"][...] = pc_label
    hdf5_file["train_set_local"][...] = pc_key_feature
    hdf5_file.close()
    return compute_time


def read_data(h5_path=''):
    readh5 = h5py.File(h5_path, "r")  # file path
    print(readh5['train_set'][:].shape)
    print(readh5['train_labels'][:].shape)
    return readh5



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
    print('train_local:', train_local, 'train_local shape:', train_local.shape)
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


def augment_data(base_path='', pc_path='', add_noise=0.04, add_outlier=0.04, n=5000, not_project=False,
                 show_result=False):

    pc = PointCloud(pc_path)
    pc.down_sample()

    if add_noise is not None:
        pc.add_noise(factor=add_noise)
    if add_outlier is not None:
        pc.add_outlier(factor=add_outlier)

    if not_project:
        for i in range(n):
            if i % 10 == 0:
                print('saving number', i+1, 'th lab random sample point clouds')
            temp = deepcopy(pc)
            temp.down_sample(number_of_downsample=1024)
            np.savetxt(base_path + '/random_sample' + str(i) + '.txt', temp.position, delimiter=' ')
    else:
        for i in range(n):
            if i % 10 == 0:
                print('saving number', i+1, 'th lab_project point clouds')

            #pc.cut_by_plane()  # todo manually
            #pc2 = PointCloud(pc.visible)
            pc2 = pc
            try:
                pc2.half_by_plane(n=1024, grid_resolution=(200, 200))
            except:
                try:
                    pc2.half_by_plane(n=1024, grid_resolution=(250, 250))
                except:
                    try:
                        pc2.half_by_plane(n=1024, grid_resolution=(300, 300))
                    except:
                        pc2.half_by_plane(n=1024, grid_resolution=(650, 650))

            np.savetxt(base_path+'/lab_project'+str(i)+'.txt', pc2.visible, delimiter=' ')  # pc.visible will variant

    if show_result:
        dir_list = [base_path + '/' + i for i in os.listdir(base_path) if os.path.isdir(i)]
        fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
        for i in dir_list:
            color =np.random.random((1, 3))
            pc = np.loadtxt(i+'/lab_project1')
            mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 2]*10**-9, color=color, figure=fig)

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


def get_local_eig_np(point_cloud, key_pts_percentage=0.1, radius_scale=(0.05, 0.1, 0.2), useiss=True):
    """
    three scale of neighbor by default is choose.
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

    max_nb_nei_pts = [0, 0, 0]

    # get max number of neighbor points.
    for i in range(batchsize):
        pc = np.squeeze(point_cloud[i])
        pc = PointCloud(pc)
        pc.generate_r_neighbor(range_rate=radius_scale[0])
        idx1 = pc.point_rneighbors  # n x ?
        pc.generate_r_neighbor(range_rate=radius_scale[1])
        idx2 = pc.point_rneighbors  # n x ?
        pc.generate_r_neighbor(range_rate=radius_scale[2])
        idx3 = pc.point_rneighbors  # n x ?
        current = (idx1.shape[1], idx2.shape[1], idx3.shape[1])

        max_nb_nei_pts = np.max(np.asarray([max_nb_nei_pts, current]), axis=0)

    np_arr1 = np.empty((batchsize, nb_points, max_nb_nei_pts[0]))  # b x n x l1 store the index of neighbor points.

    np_arr2 = np.empty((batchsize, nb_points, max_nb_nei_pts[1]))  # b x n x l2

    np_arr3 = np.empty((batchsize, nb_points, max_nb_nei_pts[2]))  # b x n x l3

    np_arr1[:] = np.nan
    np_arr2[:] = np.nan
    np_arr3[:] = np.nan

    for i in range(batchsize):
        pc = np.squeeze(point_cloud[i])
        pc = PointCloud(pc)
        pc.generate_r_neighbor(range_rate=0.05)
        idx1 = pc.point_rneighbors  # n x ?
        pc.generate_r_neighbor(range_rate=0.1)
        idx2 = pc.point_rneighbors  # n x ?
        pc.generate_r_neighbor(range_rate=0.2)
        idx3 = pc.point_rneighbors

        for j, k in enumerate(idx1):
            np_arr1[i][j][0:len(k)] = k  # k is the neighbor idx array
        for j, k in enumerate(idx2):
            np_arr2[i][j][0:len(k)] = k
        for j, k in enumerate(idx3):
            np_arr3[i][j][0:len(k)] = k

    np_arr2.astype(int)

    pts_r_cov = get_pts_cov(point_cloud, np_arr2)  # np_arr2 is b x n  b x n x 3 x 3

    eigen_val, _ = np.linalg.eigh(pts_r_cov)  # b x n x 3 orderd, to choose interested points.


    idx = np.argpartition(eigen_val[:, :, 0], nb_key_pts, axis=1)

    # using resolution control: every pixel could only contains one key point
    idx = np.empty((batchsize, nb_key_pts))
    for i in range(batchsize):
        pc = PointCloud(point_cloud[i, :])
        # specify the voxel size of resolution con_dix,trol
        _, idx[i, :] = resolution_kpts(pc.position, eigen_val[i, :, 0], pc.range/40, nb_key_pts)

    # print(eigen_val[idx])

    key_idx = idx[:, 0:nb_key_pts].astype(int)


    # print('key points coordinates:', point_cloud[idx, :], 'shape:', point_cloud[idx, :].shape)
    # b_dix = np.indices((batchsize, nb_key_pts))[1]  # b x nb_key
    # print('b_dix: ', b_dix, 'shape:', b_dix.shape)
    # batch_idx = np.concatenate([np.expand_dims(b_dix, axis=-1), np.expand_dims(idx, axis=-1)], axis=-1)  # b x nb_key x 2

    key_eig_val = np.empty((batchsize, nb_key_pts, 3))  # b x nb_keypoints x 3
    if useiss:
        for i in range(batchsize):
            key_eig_val[i, :, :] = eigen_val[i, key_idx[i, :], :]
    else:  # use my key pts detection method
        for i in range(batchsize):
            pc = PointCloud(point_cloud[i, :])
            keyptspos = pc.region_growing()  # nb_keypts x 3

            # generate r neighbor for key points
            r = pc.range * radius_scale[1]
            p_distance = distance.cdist(keyptspos, pc.position)   # nb_keypts x n
            idx = np.where((p_distance < r) & (p_distance > 0))  # idx is a list of two array

            _, uni_idx, nb_points_with_neighbors = np.unique(idx[0], return_index=True, return_counts=True)
            assert len(nb_points_with_neighbors) == nb_key_pts  # every key point has to have neighbors

            maxnb_points_of_neighbors = np.max(nb_points_with_neighbors)

            keypoint_rneighbors = np.empty((nb_key_pts, maxnb_points_of_neighbors))  # n x ?
            keypoint_rneighbors[:] = np.nan
            k = 0
            for m in range(nb_key_pts):
                for j in range(nb_points_with_neighbors[m]):  # every key point has different nb of neighbor
                    keypoint_rneighbors[idx[0][uni_idx[m]], j] = idx[1][k].astype(np.int32)
                    k += 1

            # compute covariance for key points
            whole_weight = 1 / (~np.isnan(pc.point_rneighbors)).sum(1)  # do as ISS paper said, np array (102,)
            whole_weight[whole_weight == np.inf] = 1  # avoid divided by zero
            # todo: this is an inefficient way
            #  to delete nan effect, so to implement weighted covariance_mat as ISS feature.
            cov = np.empty((nb_key_pts, 3, 3))
            cov[:] = np.nan
            for ii in range(nb_key_pts):  # for every key points
                idx_this_pts_neighbor = keypoint_rneighbors[ii, :][~np.isnan(keypoint_rneighbors[ii, :])].astype(np.int)
                assert idx_this_pts_neighbor.shape[0] > 0  # every key point has to have neighbors
                if idx_this_pts_neighbor.shape[0] > 0:

                    weight = np.append(whole_weight[ii], whole_weight[idx_this_pts_neighbor])  # add this point

                    neighbor_pts = np.append(pc.position[np.newaxis, ii, :],
                                             pc.position[idx_this_pts_neighbor], axis=0)  # (?+1) x 3 coordinates

                    try:
                        cov[ii, :, :] = np.cov(neighbor_pts, rowvar=False, ddof=0, aweights=weight)  # 3 x 3
                    except:
                        print('this point:', pc.position[ii], 'neighbor_pts:', neighbor_pts, 'aweights:', weight)

                else:
                    cov[ii, :, :] = np.eye(3)

                key_eig_val[i, ii, :], _ = np.linalg.eigh(cov[ii, :, :])  # b x nb_keypoints x 3

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


def scene_seg_dataset(pc_path, save_path, samples=1000, max_nb_pc=5, show_result = False):
    """
    default number of points of each point cloud is 1024
    :param pc_path:
    :param save_path:
    :param max_nb_pc:
    :return:
    """

    f_list = [PointCloud(pc_path+'/'+i) for i in os.listdir(pc_path) if os.path.splitext(i)[1] == '.ply']
    for i in f_list:
        i.down_sample()

    nb_classes = len(f_list)
    scene_dataset = np.zeros((samples, max_nb_pc*1024, 3))
    scene_label = np.zeros((samples, max_nb_pc*1024), dtype=np.int32)
    rate_180, rate_240, rate_300, rate_360 = [0, 0, 0, 0]

    for i in range(samples):
        print('generating the {}th scene sample'.format(i))
        nb_pc = np.random.choice(max_nb_pc) + 1
        nb_pc = max_nb_pc

        for j in range(nb_pc):
            k = np.random.choice(nb_classes)
            pc = f_list[k]
            pc.transform()
            pc.cut_by_plane()
            pc2 = PointCloud(pc.visible)
            try:
                pc2.half_by_plane(n=1024, grid_resolution=(190, 190))
                rate_180 +=1
            except:
                try:
                    pc2.half_by_plane(n=1024, grid_resolution=(260, 260))
                    rate_240 +=1
                except:
                    try:
                        pc2.half_by_plane(n=1024, grid_resolution=(330, 330))
                        rate_300 += 1
                    except:
                        pc2.half_by_plane(n=1024, grid_resolution=(400, 400))
                        rate_360 += 1
            scene_dataset[i, j*1024:j*1024+1024, :] = pc2.visible
            scene_label[i, j*1024:j*1024+1024] = k
    print('180 240 300 360:', rate_180, rate_240, rate_300, rate_360)
    if show_result:
        for i in range(1):
            scene_pc = scene_dataset[i, :, :]
            scene_pc = PointCloud(scene_pc)  #
            scene_lb = scene_label[i, :]

            figure = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
            colors = (np.random.random((nb_classes, 4))*255).astype(np.int8)
            colors[:, -1] = 255
            colors = colors[scene_lb, :]

            scalars = np.arange(np.shape(colors)[0])

            pts = mlab.quiver3d(scene_pc.position[:, 0], scene_pc.position[:, 1], scene_pc.position[:, 2],
                                scene_pc.position[:, 0]*10**-9 + 1, scene_pc.position[:, 0]*10**-9 +1,
                                scene_pc.position[:, 0] * 10 ** -9 + 1, scalars=scalars,
                                scale_factor=1, mode='sphere', figure=figure)
            pts.glyph.color_mode = 'color_by_scalar'
            pts.module_manager.scalar_lut_manager.lut.table = colors
            mlab.show()

    hdf5_file = h5py.File(save_path, mode='a')
    hdf5_file.create_dataset('train_set', (samples, max_nb_pc*1024, 3), np.float32)  # be careful about the dtype
    hdf5_file.create_dataset('train_labels', (samples, max_nb_pc*1024), np.uint8)
    hdf5_file["train_set"][...] = scene_dataset
    hdf5_file["train_labels"][...] = scene_label
    hdf5_file.close()


def txt2normalply(txt_path, write_path='/ply/'):
    """

    :param txt_path:
    :param write_path:
    :return: nothing, write file into write_path
    """
    for i, j, k in os.walk(txt_path):
        if i == txt_path:
            for m, l in enumerate(k):
                a = np.loadtxt(i + '/' + l)
                PC = PointCloud(a)
                PC.down_sample(number_of_downsample=1024)

                pc = o3d.PointCloud()
                pc.points = o3d.Vector3dVector(PC.position)
                o3d.estimate_normals(pc, o3d.KDTreeSearchParamHybrid(
                    radius=10, max_nn=10))

                o3d.write_point_cloud(i + write_path + str(m) + '.ply', pc)



if __name__ == "__main__":
    print(save_data(save_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/8object0.04noise/small_resolution_projection/data_s.h5',
              base_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/8object0.04noise/small_resolution_projection',
              normalize=False, train_data=True, shuffle=True, n=5000, nb_types=8))

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
    # for i in range(1, 9):
    #    augment_data(base_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/8object0.04noise/small_resolution_projection/lab'+str(i),
    #                  pc_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/8object0.04noise/small_resolution_projection/lab'+str(i)+'/final.ply',
    #                  add_noise=0.04, add_outlier=0.04, n=5000, not_project=False)
    # test_data(h5_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/project_data.h5', rand_trans=False, showinone=False)
    # pc = np.loadtxt('/media/sjtu/software/ASY/pointcloud/lab_workpice.txt')
    # pc = PointCloud(pc)
    # pc.show()
    #scene_seg_dataset(pc_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece',
    #                  save_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/scene_segmentation3.h5',
    #                  samples=1000, max_nb_pc=5,
    #                 show_result=True)