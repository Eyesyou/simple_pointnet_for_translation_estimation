import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import matplotlib
import open3d as o3d
import time
from matplotlib import pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab
from scipy.spatial import distance
from plyfile import PlyData, PlyElement
from scipy import spatial  # for tree structure
import cv2

SHOW_ = """
    plydata = PlyData.read(pc_path1)
    vertex = np.asarray([list(subtuple) for subtuple in plydata['vertex'][:]])
    vertex = vertex[:, 0:3]
    pc = PointCloud(vertex)
    pc.down_sample(number_of_downsample=10000)
    pc.show()
    """


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


def show_pc(point_cloud):
    x, y, z = point_cloud[0, :, 0], point_cloud[0, :, 1], point_cloud[0, :, 2]
    x_2, y_2, z_2 =point_cloud[1, :, 0], point_cloud[1, :, 1], point_cloud[1, :, 2]
    x_3, y_3, z_3 = point_cloud[2, :, 0], point_cloud[2, :, 1], point_cloud[2, :, 2]
    x_4, y_4, z_4 = point_cloud[3, :, 0], point_cloud[3, :, 1], point_cloud[3, :, 2]
    x, y, z = np.squeeze(x), np.squeeze(y), np.squeeze(z)
    x_2, y_2, z_2 = np.squeeze(x_2), np.squeeze(y_2), np.squeeze(z_2)
    x_3, y_3, z_3 = np.squeeze(x_3), np.squeeze(y_3), np.squeeze(z_3)
    x_4, y_4, z_4 = np.squeeze(x_4), np.squeeze(y_4), np.squeeze(z_4)

    ax = plt.subplot(221, projection='3d')
    bx = plt.subplot(222, projection='3d')
    cx = plt.subplot(223, projection='3d')
    dx = plt.subplot(224, projection='3d')
    ax.scatter(x, y, z, c='y', s=20)
    bx.scatter(x_2, y_2, z_2, c='r', s=20)
    cx.scatter(x_3, y_3, z_3, c='g', s=20)
    dx.scatter(x_4, y_4, z_4, c='blue', s=20)
    plt.axis('equal')
    plt.show()


def show_all(point_cloud, color=None, plot_plane=False, plot_arrow=True):
    """

    :param point_cloud:
    :param color:
    :param plot_plane:
    :param plot_arrow:
    :return:
    """

    plt3d = plt.figure().gca(projection='3d')
    # plot the plane
    if plot_plane:
        if point_cloud.plane is not None:
            xx, yy = np.meshgrid(range(100), range(100))
            z = (-point_cloud.plane[0] * xx - point_cloud.plane[1] * yy - point_cloud.plane[3]) * 1. / point_cloud.plane[2]

            plt3d.plot_surface(xx, yy, z)

    plt3d.quiver(point_cloud.plane_origin[0], point_cloud.plane_origin[1], point_cloud.plane_origin[2],
                 point_cloud.plane_origin[0] - point_cloud.center[0],
                 point_cloud.plane_origin[1] - point_cloud.center[1],
                 point_cloud.plane_origin[2] - point_cloud.center[2], length=1000, normalize=True)

    if plot_arrow:
        x = point_cloud.visible[:, 0]
        y = point_cloud.visible[:, 1]
        z = point_cloud.visible[:, 2]
        u = point_cloud.plane_project_points[:, 0] - point_cloud.visible[:, 0]
        v = point_cloud.plane_project_points[:, 1] - point_cloud.visible[:, 1]
        w = point_cloud.plane_project_points[:, 2] - point_cloud.visible[:, 2]
        plt3d.quiver(x, y, z, u, v, w, length=100, )

        mlab.quiver3d(x, y, z, u, v, w, scale_factor=0.01)   # scale_factor=1
        mlab.show()

    point_cloud = point_cloud.visible

    if len(point_cloud.shape) == 2:
        point_cloud = np.expand_dims(point_cloud, axis=0)

    B = point_cloud.shape[0]
    a, b, c = point_cloud[:, :, 0], point_cloud[:, :, 1], point_cloud[:, :, 2]   # Bxnx1

    a, b, c = np.reshape(a, (1, -1)), np.reshape(b, (1, -1)), np.reshape(c, (1, -1))

    for i in range(B):
        if color==None:
            plt3d.scatter(a[i, :], b[i, :], c[i, :], color=[random.random(), random.random(), random.random()], s=5)
        else:
            plt3d.scatter(a[i, :], b[i, :], c[i, :], color=color, s=5)

    Axes3D.grid(plt3d, b=False)
    set_axes_equal(plt3d)
    plt3d.set_axis_off()
    plt3d.set_aspect('equal')
    # plt.axis('equal')
    # plt.show()


def show_trans(point_cloud1, point_cloud2, colorset=[], use_mayavi=True, scale=4):
    """
        plot a batch of point clouds
    :param point_cloud1: Bx1024x3 point_cloud2: Bx1024x3 np array
    :param point_cloud2: Bx1024x3 point_cloud2: Bx1024x3 np array
    :param colorset:   for point color
    :param use_mayavi:
    :param scale:
    :return: nothing
    """

    a1, b1, c1 = point_cloud1[:, :, 0], point_cloud1[:, :, 1], point_cloud1[:, :, 2]  # Bxnx1
    a1, b1, c1 = np.squeeze(a1), np.squeeze(b1), np.squeeze(c1)  # Bxn
    a2, b2, c2 = point_cloud2[:, :, 0], point_cloud2[:, :, 1], point_cloud2[:, :, 2]  # Bxnx1
    a2, b2, c2 = np.squeeze(a2), np.squeeze(b2), np.squeeze(c2)  # Bxn

    ax = plt.subplot(111, projection='3d', facecolor='w')
    mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    ax.set_axis_off()
    B = point_cloud1.shape[0] #batch

    dark_multiple = 2.5   # greater than 1
    for idx, i in enumerate(colorset):
        if idx % 2 == 0 and i is None:
            colorset[idx] = tuple((1/dark_multiple)*np.random.random([3, ]))
            colorset[idx+1] = [i * dark_multiple for i in colorset[idx]]

    colorset = [tuple(i) for i in colorset]

    for i in range(B):
        i_th_color = int(i % len(colorset))
        if not use_mayavi:
            ax.scatter(a1[i, :], b1[i, :], c1[i, :], color=colorset[0], s=scale)
            ax.scatter(a2[i, :], b2[i, :], c2[i, :], color=colorset[1], s=scale)
        else:
            points = mlab.points3d(a1[i, :], b1[i, :], c1[i, :], c1[i, :] * 10**-9 + scale, color=colorset[i][0], scale_factor=1)
            points2 = mlab.points3d(a2[i, :], b2[i, :], c2[i, :], c2[i, :] * 10**-9 + scale, color=colorset[i][1], scale_factor=1)
            points.glyph.glyph_source.glyph_source.phi_resolution = 64
            points.glyph.glyph_source.glyph_source.theta_resolution = 64
            points2.glyph.glyph_source.glyph_source.phi_resolution = 64
            points2.glyph.glyph_source.glyph_source.theta_resolution = 64

    Axes3D.grid(ax, b=False)
    if use_mayavi:
        mlab.show()
    else:
        plt.show()


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def show_custom(point_cloud, color=None):
    """
    :param point_cloud: Bxnx3,numpy array
    :return:
    """

    result1= point_cloud[0::4, :, :]  #every cow
    result2 = point_cloud[1::4, :, :] #every bunny
    result3 = point_cloud[2::4, :, :] #every shaft
    result4 = point_cloud[3::4, :, :] #every projector

    B = point_cloud.shape[0]
    a1, b1, c1 = result1[:, :, 0], result1[:, :, 1], result1[:, :, 2]   # Bxnx1
    a1, b1, c1 = np.squeeze(a1), np.squeeze(b1), np.squeeze(c1)   # Bxn
    a2, b2, c2 = result2[:, :, 0], result2[:, :, 1], result2[:, :, 2]   # Bxnx1
    a2, b2, c2 = np.squeeze(a2), np.squeeze(b2), np.squeeze(c2)   # Bxn
    a3, b3, c3 = result3[:, :, 0], result3[:, :, 1], result3[:, :, 2]   # Bxnx1
    a3, b3, c3 = np.squeeze(a3), np.squeeze(b3), np.squeeze(c3)   # Bxn
    a4, b4, c4 = result4[:, :, 0], result4[:, :, 1], result4[:, :, 2]   # Bxnx1
    a4, b4, c4 = np.squeeze(a4), np.squeeze(b4), np.squeeze(c4)   # Bxn

    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')
    #ax = fig.gca(projection='3d')

    if color == None:
        ax.scatter(a1, b1, c1, s=5, c='r', depthshade=False)
        ax.scatter(a2, b2, c2, s=5, c='g', depthshade=False)
        ax.scatter(a3, b3, c3, s=5, c='b', depthshade=False)
        ax.scatter(a4, b4, c4, s=5, c='y', depthshade=False)
    else:
        ax.scatter(a1, b1, c1, s=5, c=color, depthshade=False)
        ax.scatter(a2, b2, c2, s=5, c=color, depthshade=False)
        ax.scatter(a3, b3, c3, s=5, c=color, depthshade=False)
        ax.scatter(a4, b4, c4, s=5, c=color, depthshade=False)

    Axes3D.grid(ax, b=False)
    plt.show()


def half_pc_by_projection(batch_pc):
    """

    :param batch_pc: B*n*3 point cloud input nparray
    :return: B*m*3 point cloud input m < n
    """
    batch = np.shape(batch_pc)[0]
    nb_points = np.shape(batch_pc)[1]
    pc_range = batch_pc   # Bx1
    front_flag = np.zeros([batch, nb_points])


    return


def tf_quat_pos_2_homo(batch_input):
    """

    :param batch_input: batchx7 4 quaternion 3 position xyz
    :return: transformation: batch homogeneous matrix batch x 4 x 4
    """
    batch = batch_input.shape[0].value  #or tensor.get_shape().as_list()

    w = tf.slice(batch_input, [0, 0], [batch, 1])       # all shape of: (batch,1)
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
    batch_out = tf.concat([batch_out, pad], axis=1)  # Bx4x4
    return batch_out


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

    batch_out = tf.matmul(homo, batch_out)  # Bx4x4 batch multiply Bx4xn, points coordinates in column vector,left-handed rotation matrix

    batch_out = tf.div(batch_out, tf.slice(batch_out, [0, 3, 0], [batch, 1, num]))  # every element divided the
    # last element to get true coordinates
    batch_out = tf.slice(batch_out, [0, 0, 0], [batch, 3, num]) # Bx3xn
    batch_out = tf.transpose(batch_out, perm=[0, 2, 1])    # Bxnx3

    return batch_out


def region_growing_cluster_keypts(arr1, nb_pts=10, pts_range=None, intra_dist= 1/100, inter_dist=1/100):
    """
    compute key points of arr1, keep in mind the arr1 are salient points from point clouds
    :param arr1:
    :param nb_pts:
    :param pts_range:
    :return:
    """
    if isinstance(arr1, PointCloud):
        pts_range = arr1.range
        arr1 = arr1.position

    n = np.shape(arr1)[0]
    intra_cluster_dist = pts_range * intra_dist  # parameters to be tuned!
    inter_cluster_dist = pts_range * inter_dist  # parameters to be tuned!

    tree = spatial.cKDTree(arr1)
    clusters = []

    for i in np.random.permutation(n):
        pts = arr1[i, :]
        cluster = tree.query_ball_point(pts, intra_cluster_dist)
        cluster = list(filter(lambda a: a != i, cluster))
        if len(cluster) >= 2:  # minimum number of points in a cluster
            clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    clusters = [sorted(x) for x in clusters]
    b_set = set(tuple(x) for x in clusters)
    clusters = [list(x) for x in b_set]

    assert clusters

    centers = np.mean(arr1[clusters[0], :], axis=0, keepdims=True)

    tree = spatial.cKDTree(centers)
    # print('before reject points by inter cluster distance, the number of points is:', len(clusters))
    # rejection based on inter_distance
    for i in range(len(clusters) - 1):
        center = np.mean(arr1[clusters[i + 1], :], axis=0, keepdims=False)

        if tree.query_ball_point(center, inter_cluster_dist):
            # query = tree.query_ball_point(center, inter_cluster_dist)
            # print('inter_cluster_dist:', inter_cluster_dist,
            #       'this point:', center, 'queryed point:', centers[query,:])
            pass
        else:
            centers = np.concatenate([centers, center[np.newaxis, :]], axis=0)
            tree = spatial.cKDTree(centers)  # update the tree
    # print('after reject points by inter cluster distance, the number of points is:', np.shape(centers))
    assert np.shape(centers)[0] > nb_pts
    centers = centers[:nb_pts, :]
    return centers


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


class PointCloud:
    def __init__(self, one_pointcloud):
        if isinstance(one_pointcloud, str):  # ply file path
            if os.path.splitext(one_pointcloud)[-1] == '.ply':
                plydata = PlyData.read(one_pointcloud)
                points = np.asarray([list(subtuple) for subtuple in plydata['vertex'][:]])
            elif os.path.splitext(one_pointcloud)[-1] == '.txt':
                points = np.loadtxt(one_pointcloud)
            else:
                raise RuntimeError('format do not support')
            one_pointcloud = points[:, 0:3]

        assert isinstance(one_pointcloud, np.ndarray)
        one_pointcloud = np.squeeze(one_pointcloud)
        assert one_pointcloud.shape[1] == 3
        self.min_limit = np.amin(one_pointcloud, axis=0)  # 1x3
        self.max_limit = np.amax(one_pointcloud, axis=0)  # 1x3
        self.range = self.max_limit - self.min_limit
        self.range = np.sqrt(self.range[0] ** 2 + self.range[1] ** 2 + self.range[2] ** 2)  # diagonal distance
        self.position = one_pointcloud  # nx3 numpy array
        self.center = np.mean(self.position, axis=0)  # 1x3
        self.nb_points = np.shape(self.position)[0]
        self.visible = self.position  # for projection use
        self.plane = None
        self.plane_origin = None
        self.plane_project_points = None
        self.root = None
        self.depth = 0
        self.point_kneighbors = None  # n x k  k is the index of the neighbor points
        # n x (0-inf)  the index of the neighbor points,number may vary according to different points
        self.point_rneighbors = None
        self.keypoints = None  # k , index of k key points in the point cloud
        self.weighted_covariance_matix = None
        self.deficiency = None
        self.saliency =None
        print(self.nb_points, ' points', 'range:', self.range)

    def __add__(self, other):
        return PointCloud(np.concatenate([self.position, other.position], axis=0))

    def half_by_plane(self, plane=None, n=1024, grid_resolution=(256, 256), show_result=False):
        """
        implement the grid plane projection method
        :param plane:  the plane you want to project the point cloud into, and generate the image-like grid,
        define the normal of the plane is to the direction of point cloud center
        :param n:
        :param grid_resolution:
        :return: nothing, update self.visible points
        """
        if plane is None:
            # generate a random plane whose distance to the center bigger than self.range
            # d = abs(Ax+By+Cz+D)/sqrt(A**2+B**2+C**2)
            plane_normal = -0.5 + np.random.random(size=[3, ])  # random A B C for plane Ax+By+Cz+D=0
            A = plane_normal[0]
            B = plane_normal[1]
            C = plane_normal[2]
            D = -(A * self.center[0] + B * self.center[1] + C * self.center[2]) + (np.random.binomial(1, 0.5) * 2 - 1) * \
                self.range * np.sqrt(A ** 2 + B ** 2 + C ** 2)

        else:
            A = plane[0]
            B = plane[1]
            C = plane[2]
            D = plane[3]

        # compute the project point of center in the grid plane:
        t = (A * self.center[0] + B * self.center[1] + C * self.center[2] + D) / (A ** 2 + B ** 2 + C ** 2)
        x0 = self.center[0] - A * t  # point cloud center project point in the plane
        y0 = self.center[1] - B * t
        z0 = self.center[2] - C * t
        self.plane_origin = [x0, y0, z0]
        if (self.center[0] - x0) / A < 0:  # inverse a b c d denotes the same plane
            A = -A
            B = -B
            C = -C
            D = -D
        self.plane = [A, B, C, D]
        try:
            assert math.isclose((self.center[0] - x0) / A, (self.center[1] - y0) / B) and \
                   math.isclose((self.center[1] - y0) / B, (self.center[2] - z0) / C) and (self.center[0] - x0) / A > 0
        except AssertionError:
            print('AssertionError', (self.center[0] - x0) / A, (self.center[1] - y0) / B, (self.center[2] - z0) / C, A,
                  B, C, D)
        x1 = x0  # Parallelogram points of the grid,define x1,y1,z1 by plane function and
        a = 1 + B ** 2 / C ** 2  # range distance limitation
        b = 2 * B / C * (z0 + (D + A * x1) / C) - 2 * y0
        c = y0 ** 2 - self.range ** 2 / 4 + (x1 - x0) ** 2 + (z0 + (D + A * x1) / C) ** 2
        y1 = np.roots([a, b, c])  # Unary two degree equation return two root
        if np.isreal(y1[0]):
            y1 = y1[0]
        else:
            print('not real number')
        z1 = -(D + A * x1 + B * y1) / C
        # the y direction of the plane, this is a vector
        y_nomal = np.cross([self.center[0] - x0, self.center[1] - y0, self.center[2] - z0], [x1 - x0, y1 - y0, z1 - z0])

        # the minimal distance for every grid, the second index store the point label

        min_dist = 10 * self.range * np.ones(shape=[grid_resolution[0], grid_resolution[1], 2])
        point_label = np.zeros(shape=(self.nb_points,))
        for i in range(self.nb_points):

            t_ = (A * self.position[i, 0] + B * self.position[i, 1] + C * self.position[i, 2] + D) \
                 / (A ** 2 + B ** 2 + C ** 2)
            project_point = np.asarray([self.position[i, 0] - A * t_, self.position[i, 1] - B * t_,
                                        self.position[i, 2] - C * t_])

            project_y = point2line_dist(project_point, np.asarray([x0, y0, z0]),
                                        np.asarray([x1 - x0, y1 - y0, z1 - z0]))
            project_x = np.sqrt(np.sum(np.square(project_point - np.asarray([x0, y0, z0]))) - project_y ** 2)

            # print('project x', project_x, 'project y', project_y)
            if (project_point[0] - x0) * (x1 - x0) + (project_point[1] - y0) * (y1 - y0) + (project_point[2] - z0) * (
                    z1 - z0) >= 0:
                # decide if it is first or fourth quadrant
                if np.dot(y_nomal, project_point - np.asarray([x0, y0, z0])) < 0:
                    # fourth quadrant
                    project_y = -project_y

            else:
                project_x = - project_x
                if np.dot(y_nomal, project_point - np.asarray([x0, y0, z0])) < 0:
                    # third quadrant
                    project_y = -project_y

            pixel_width = self.range * 2 / grid_resolution[0]
            pixel_height = self.range * 2 / grid_resolution[1]
            distance = point2plane_dist(self.position[i, :], [A, B, C, D])
            index_x = int(grid_resolution[0] / 2 + np.floor(project_x / pixel_width))
            index_y = int(grid_resolution[1] / 2 + np.floor(project_y / pixel_height))
            try:
                if distance < min_dist[index_x, index_y, 0]:
                    min_dist[index_x, index_y, 0] = distance
                    # if other points is already projected, set it to 0
                    if np.equal(np.mod(min_dist[index_x, index_y, 1], 1), 0):
                        old_point_index = min_dist[index_x, index_y, 1]
                        old_point_index = int(old_point_index)
                        point_label[old_point_index] = 0
                    min_dist[index_x, index_y, 1] = i  # new point index
                    point_label[i] = 1  # visible points
            except AssertionError:
                print('AssertionError:', np.floor(project_x / pixel_width), pixel_width)

        if n is not None:
            # sample the visible points to given number of points
            medium = self.position[point_label == 1]
            try:
                assert medium.shape[0] >= n  # sampled points have to be bigger than n
            except AssertionError:
                print('sampled points number is:', medium.shape[0])

                raise ValueError('value error, increase grid number')
            np.random.shuffle(medium)  # only shuffle the first axis
            self.visible = medium[0:n, :]
        else:
            self.visible = self.position[point_label == 1]

        t_1 = (A * self.visible[:, 0] + B * self.visible[:, 1] + C * self.visible[:, 2] + D) \
              / (A ** 2 + B ** 2 + C ** 2)
        t_1 = np.expand_dims(t_1, axis=1)

        self.plane_project_points = np.concatenate([np.expand_dims(self.visible[:, 0], axis=1) - A * t_1,
                                                    np.expand_dims(self.visible[:, 1], axis=1) - B * t_1,
                                                    np.expand_dims(self.visible[:, 2], axis=1) - C * t_1], axis=1)

        if show_result:
            mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
            fig = mlab.points3d(self.visible[:, 0], self.visible[:, 1], self.visible[:, 2],
                                self.visible[:, 2] * 10 ** -2 + 1, color=(0, 1, 0),  # +self.range * scale
                                scale_factor=0.4)  # colormap='Spectral', color=(0, 1, 0)
            fig.glyph.glyph_source.glyph_source.phi_resolution = 64
            fig.glyph.glyph_source.glyph_source.theta_resolution = 64
            mlab.show()

    def cut_by_plane(self, plane=None, show_result=False):
        if plane is None:
            # generate a random plane whose distance to the center bigger than self.range
            # d = abs(Ax+By+Cz+D)/sqrt(A**2+B**2+C**2)
            plane_normal = -0.5 + np.random.random(size=[3, ])  # random A B C for plane Ax+By+Cz+D=0
            A = plane_normal[0]
            B = plane_normal[1]
            C = plane_normal[2]
            D = -(A * self.center[0] + B * self.center[1] + C * self.center[2])

        else:
            A = plane[0]
            B = plane[1]
            C = plane[2]
            D = plane[3]

        idx_direction = A*self.position[:, 0]+B*self.position[:, 1]+C*self.position[:, 2]+D  # nx1
        self.visible = self.position[idx_direction > 0, :]

        if show_result:
            mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
            fig = mlab.points3d(self.visible[:, 0], self.visible[:, 1], self.visible[:, 2],
                                self.visible[:, 2] * 10 ** -2 + 1, color=(0, 1, 0),  # +self.range * scale
                                scale_factor=0.4)  # colormap='Spectral', color=(0, 1, 0)
            mlab.show()

    def show(self, not_show=False, scale=0.4):
        mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
        fig = mlab.points3d(self.position[:, 0], self.position[:, 1], self.position[:, 2],
                            self.position[:, 2] * 10**-2 + 1, color=(0, 1, 0),  # +self.range * scale
                            scale_factor=scale)   # colormap='Spectral', color=(0, 1, 0)

        if not not_show:
            mlab.show()
        else:
            return fig

    def add_noise(self, factor=1 / 100):
        """
        jitter noise for every points in the point cloud
        :param factor:
        :return:
        """
        noise = np.random.random([self.nb_points, 3]) * factor * self.range
        self.position += noise

    def add_outlier(self, factor=1 / 100):
        """
        randomly delete points and make it to be the outlier
        :param factor:
        :return:
        """

        inds = np.random.choice(np.arange(self.nb_points), size=int(factor * self.nb_points))
        self.position[inds] = self.center + -self.range / 2 + self.range / 1 * np.random.random(size=(len(inds), 3))

    def normalize(self):
        self.position -= self.center
        self.position /= self.range
        self.center = np.mean(self.position, axis=0)
        self.min_limit = np.amin(self.position, axis=0)
        self.max_limit = np.amax(self.position, axis=0)
        self.range = self.max_limit - self.min_limit
        self.range = np.sqrt(self.range[0] ** 2 + self.range[1] ** 2 + self.range[2] ** 2)
        print('normalized_center: ', self.center, 'normalized_range:', self.range)

    def transform(self, homo_transformation=None):
        """
        return nothing, update point cloud
        :param homo_transformation:
        :return:
        """
        batch = 1
        num = self.nb_points

        # batch_out = tf.Variable(tf.zeros(pc_batch_input.shape), trainable=False, dtype=tf.float32)
        # batch_out = batch_out.assign(pc_batch_input)
        batchout = np.concatenate([self.position[np.newaxis, :], np.ones((batch, num, 1))], axis=2)
        batchout = np.transpose(batchout, (0, 2, 1))

        if homo_transformation == None:
            ran_pos = np.concatenate([np.random.uniform(low=0.99, high=1, size=(1, 1)),
                                      np.random.uniform(low=0.6, high=1, size=(1, 3)),
                                      np.random.uniform(low=-self.range*0.8, high=self.range*0.8, size=(1, 3))
                                      ], axis=1)
            ran_pos = np.concatenate([ran_pos[:, 0:4]/np.linalg.norm(ran_pos[:, 0:4], axis=1), ran_pos[:, 4:7]], axis=1)
            homo_transformation = np_quat_pos_2_homo(ran_pos)

        batchout = np.matmul(homo_transformation, batchout)  # Bx4x4 * B x 4 x n
        batchout = np.divide(batchout, batchout[:, np.newaxis, 3, :])
        batchout = batchout[:, :3, :]
        batchout = np.transpose(batchout, (0, 2, 1))
        self.position = batchout[0, :]
        self.center = np.mean(self.position, axis=0)
        self.min_limit = np.amin(self.position, axis=0)
        self.max_limit = np.amax(self.position, axis=0)

    def octree(self, show_layers=0, colors=None):
        def width_first_traversal(position, size, data):
            root = OctNode(position, size, data)
            if np.shape(np.array(list(set([tuple(t) for t in root.data]))))[0] == 1:
                return root

            min = root.position + [-root.size / 2, -root.size / 2, -root.size / 2]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            lbd = root.data[np.all(media, axis=1), :]
            if lbd.shape[0] > 1:
                root.dbl = width_first_traversal(
                    root.position + [-1 / 4 * root.size, -1 / 4 * root.size, -1 / 4 * root.size],
                    root.size * 1 / 2, data=lbd)
            elif lbd.shape[0] == 1:
                root.dbl = OctNode(root.position + [-1 / 4 * root.size, -1 / 4 * root.size, -1 / 4 * root.size],
                                   root.size * 1 / 2, data=lbd)

            min = root.position + [0, -root.size / 2, -root.size / 2]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            rbd = root.data[np.all(media, axis=1), :]
            if rbd.shape[0] > 1:
                root.dbr = width_first_traversal(
                    root.position + [1 / 4 * root.size, -1 / 4 * root.size, -1 / 4 * root.size],
                    root.size * 1 / 2, data=rbd)
            elif rbd.shape[0] == 1:
                root.dbr = OctNode(root.position + [1 / 4 * root.size, -1 / 4 * root.size, -1 / 4 * root.size],
                                   root.size * 1 / 2, data=rbd)

            min = root.position + [-root.size / 2, 0, -root.size / 2]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            lfd = root.data[np.all(media, axis=1), :]
            if lfd.shape[0] > 1:
                root.dfl = width_first_traversal(
                    root.position + [-1 / 4 * root.size, 1 / 4 * root.size, -1 / 4 * root.size],
                    root.size * 1 / 2, data=lfd)
            elif lfd.shape[0] == 1:
                root.dfl = OctNode(
                    root.position + [-1 / 4 * root.size, 1 / 4 * root.size, -1 / 4 * root.size],
                    root.size * 1 / 2, data=lfd)

            min = root.position + [0, 0, -root.size / 2]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            rfd = root.data[np.all(media, axis=1), :]
            if rfd.shape[0] > 1:
                root.dfr = width_first_traversal(
                    root.position + [1 / 4 * root.size, 1 / 4 * root.size, -1 / 4 * root.size],
                    root.size * 1 / 2, data=rfd)
            elif rfd.shape[0] == 1:
                root.dfr = OctNode(
                    root.position + [1 / 4 * root.size, 1 / 4 * root.size, -1 / 4 * root.size],
                    root.size * 1 / 2, data=rfd)

            min = root.position + [-root.size / 2, -root.size / 2, 0]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            lbu = root.data[np.all(media, axis=1), :]
            if lbu.shape[0] > 1:
                root.ubl = width_first_traversal(
                    root.position + [-1 / 4 * root.size, -1 / 4 * root.size, 1 / 4 * root.size],
                    root.size * 1 / 2, data=lbu)
            elif lbu.shape[0] == 1:
                root.ubl = OctNode(
                    root.position + [-1 / 4 * root.size, -1 / 4 * root.size, 1 / 4 * root.size],
                    root.size * 1 / 2, data=lbu)

            min = root.position + [0, -root.size / 2, 0]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            rbu = root.data[np.all(media, axis=1), :]
            if rbu.shape[0] > 1:
                root.ubr = width_first_traversal(
                    root.position + [1 / 4 * root.size, -1 / 4 * root.size, 1 / 4 * root.size],
                    root.size * 1 / 2, data=rbu)
            elif rbu.shape[0] == 1:
                root.ubr = OctNode(
                    root.position + [1 / 4 * root.size, -1 / 4 * root.size, 1 / 4 * root.size],
                    root.size * 1 / 2, data=rbu)

            min = root.position + [-root.size / 2, 0, 0]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            lfu = root.data[np.all(media, axis=1), :]
            if lfu.shape[0] > 1:
                root.ufl = width_first_traversal(
                    root.position + [-1 / 4 * root.size, 1 / 4 * root.size, 1 / 4 * root.size],
                    root.size * 1 / 2, data=lfu)
            elif lfu.shape[0] == 1:
                root.ufl = OctNode(
                    root.position + [-1 / 4 * root.size, 1 / 4 * root.size, 1 / 4 * root.size],
                    root.size * 1 / 2, data=lfu)

            min = root.position + [0, 0, 0]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            rfu = root.data[np.all(media, axis=1), :]
            if rfu.shape[0] > 1:
                root.ufr = width_first_traversal(
                    root.position + [1 / 4 * root.size, 1 / 4 * root.size, 1 / 4 * root.size],
                    root.size * 1 / 2, data=rfu)
            elif rfu.shape[0] == 1:
                root.ufr = OctNode(
                    root.position + [1 / 4 * root.size, 1 / 4 * root.size, 1 / 4 * root.size],
                    root.size * 1 / 2, data=rfu)

            root.children = [root.ubl, root.ubr, root.ufl, root.ufr, root.dbl, root.dbr, root.dfl, root.dfr]

            try:
                number_points = np.shape(root.data)[0]
                sum = 0
                for child in root.children:
                    if child is not None:
                        sum = sum + np.shape(child.data)[0]

                assert number_points == sum

            except AssertionError:
                print('assertion error, total number of points in this level is:', number_points)
                pass
                for child in root.children:
                    if child is not None:
                        print(np.shape(child.data)[0], 'and', end=' ')
                        pass
                print('\n')
            return root

        self.root = width_first_traversal(self.center,
                                          size=2*max(max(self.max_limit - self.center), max(self.center-self.min_limit)),
                                          data=self.position)

        def maxdepth(node):
            if not any(node.children):
                return 0
            else:
                return max([maxdepth(babe) + 1 for babe in node.children if babe is not None])

        self.depth = maxdepth(self.root)

        if show_layers:
            if colors is None:
                colors = np.random.random((8, 8, 8, 3))
            mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
            for i, child in enumerate(self.root.children):
                if child is not None:
                    for j, grand_child in enumerate(child.children):
                        if grand_child is not None:
                            # for k, grandgrand_child in enumerate(grand_child.children):
                            #     if grandgrand_child is not None:
                            k = 0
                            mlab.points3d(grand_child.data[:, 0], grand_child.data[:, 1], grand_child.data[:, 2],
                                          grand_child.data[:, 2]*10**-9+1, color=tuple(colors[i, j, k, :].tolist()),
                                          scale_mode='scalar', scale_factor=1)

            mlab.show()

    def generate_k_neighbor(self, k=10, show_result=False):
        assert k < self.nb_points
        self.point_kneighbors = None
        p_distance = distance.cdist(self.position, self.position)
        idx = np.argpartition(p_distance, (1, k + 1), axis=0)[1:k + 1]
        self.point_kneighbors = np.transpose(idx)  # n x 3

        if show_result:
            for i in range(10):  # number of pictures you want to show
                j = np.random.choice(self.nb_points, 1)  # point
                fig2 = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))

                neighbor_idx = self.point_kneighbors[j, :]
                neighbor_idx = neighbor_idx[~np.isnan(neighbor_idx)].astype(np.int32)
                # show the neighbor point cloud
                mlab.points3d(self.position[neighbor_idx, 0], self.position[neighbor_idx, 1],
                              self.position[neighbor_idx, 2],
                              self.position[neighbor_idx, 2] * 10**-6 + self.range * 0.05,
                              color=tuple(np.random.random((3,)).tolist()),
                              scale_factor=0.2)  # tuple(np.random.random((3,)).tolist())

                # show the whole point cloud
                mlab.points3d(self.position[:, 0], self.position[:, 1], self.position[:, 2],
                              self.position[:, 2] * 10**-9 + self.range * 0.05,
                              color=(0, 1, 0), scale_factor=0.1)
                mlab.show()

    def generate_r_neighbor(self, range_rate=0.05, show_result=False, use_dificiency=False):
        """

        :param rate: range ratio of the neighbor scale
        :param show_result:
        :param use_dificiency: compute dificiency for every point
        :return:
        """
        assert 0 < range_rate < 1
        r = self.range * range_rate
        p_distance = distance.cdist(self.position, self.position)
        idx = np.where((p_distance < r) & (p_distance > 0))  # choose axis 0 or axis 1

        _, uni_idx, nb_points_with_neighbors = np.unique(idx[0], return_index=True, return_counts=True)

        maxnb_points_with_neighbors = np.max(nb_points_with_neighbors)

        self.point_rneighbors = np.empty((self.nb_points, maxnb_points_with_neighbors))
        self.point_rneighbors[:] = np.nan
        k = 0
        for i in range(len(nb_points_with_neighbors)):
            for j in range(nb_points_with_neighbors[i]):
                self.point_rneighbors[idx[0][uni_idx[i]], j] = idx[1][k].astype(np.int32)
                k += 1

        if use_dificiency:
            dificiency = np.ones((self.nb_points,))
            neighbor_idx = self.point_rneighbors
            for i in range(self.nb_points):
                # neighbor_idx = neighbor_idx[~np.isnan(neighbor_idx)].astype(np.int32)
                neighbor_points = neighbor_idx[i, :][~np.isnan(neighbor_idx[i, :])].astype(np.int32)
                # print('neighbor_points:', neighbor_points)
                neighbor_points = self.position[neighbor_points, :]
                flag = np.zeros((5, 5, 5), dtype=np.int32)
                for j in range(np.shape(neighbor_points)[0]):
                    voxel_index = (neighbor_points[j, :]-self.position[i, :])/(r/2)
                    voxel_index = np.rint(voxel_index).astype(np.int32)+2
                    flag[voxel_index[0]][voxel_index[1]][voxel_index[2]] = 1    # correct way to use array as index
                dificiency[i] = np.sum(flag)/64
                # print('dificiency:', np.sum(flag), '/', 64)
                # [self.position[neighbor_idx, 0], self.position[neighbor_idx, 1],
                #  self.position[neighbor_idx, 2]]

            self.deficiency = dificiency

        if show_result:
            if self.keypoints is not None:
                print('key points already exist, plot them now')
                fig2 = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))

                neighbor_idx = self.point_rneighbors[self.keypoints, :]
                neighbor_idx = neighbor_idx[~np.isnan(neighbor_idx)].astype(np.int32)
                # show the neighbor point cloud in color
                mlab.points3d(self.position[neighbor_idx, 0], self.position[neighbor_idx, 1],
                              self.position[neighbor_idx, 2],
                              self.position[neighbor_idx, 0] * 10 ** -9 + self.range * 0.005,
                              color=(1, 1, 0),
                              scale_factor=2, figure=fig2)  # color can be tuple(np.random.random((3,)).tolist())

                # show the whole point cloud
                mlab.points3d(self.position[:, 0], self.position[:, 1], self.position[:, 2],
                              self.position[:, 2] * 10 ** -9 + self.range * 0.005,
                              color=(0, 1, 0), scale_factor=1.5)

                # show the sphere on the neighbor in transparent color
                mlab.points3d(self.position[self.keypoints, 0], self.position[self.keypoints, 1], self.position[self.keypoints, 2],
                              self.position[self.keypoints, 0] * 10 ** -9 + r * 2, color=(0, 0, 1), scale_factor=1,
                              transparent=True, opacity=0.025)  # 0.1 for rate 0.05, 0.04 for rate 0.1, 0.025 for rate 0.15 in empirical

                # show the key points
                mlab.points3d(self.position[self.keypoints, 0], self.position[self.keypoints, 1],
                              self.position[self.keypoints, 2],
                              self.position[self.keypoints, 0] * 10 ** -9 + self.range * 0.005,
                              color=(1, 0, 0), scale_factor=2)
                mlab.show()

            else:
                fig2 = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
                self.compute_key_points(percentage=0.01, resolution_control=1/7, rate=range_rate, use_deficiency=True) # get the key points id

                for i in range(15):
                    # j = np.random.choice(self.nb_points, 1)  # choose one random point index to plot
                    j = self.keypoints[i]
                    neighbor_idx = self.point_rneighbors[j, :]
                    neighbor_idx = neighbor_idx[~np.isnan(neighbor_idx)].astype(np.int32)
                    # show the neighbor point cloud
                    mlab.points3d(self.position[neighbor_idx, 0], self.position[neighbor_idx, 1],
                                  self.position[neighbor_idx, 2],
                                  self.position[neighbor_idx, 0] * 10**-9 + self.range * 0.005,
                                  color=tuple(np.random.random((3,)).tolist()),
                                  scale_factor=3, figure=fig2, resolution=16)  # tuple(np.random.random((3,)).tolist())

                    # show the whole point cloud
                    mlab.points3d(self.position[:, 0], self.position[:, 1], self.position[:, 2],
                                  self.position[:, 2] * 10**-9 + self.range * 0.005,
                                  color=(0, 1, 0), scale_factor=1.5, resolution=16, figure=fig2)

                    # show the key point
                    mlab.points3d(self.position[j, 0], self.position[j, 1], self.position[j, 2],
                                  self.position[j, 0] * 10 ** -9 + self.range * 0.005, color=(1, 0, 0), scale_factor=6,
                                  figure=fig2, resolution=256)

                    # show the sphere on the neighbor
                    mlab.points3d(self.position[j, 0], self.position[j, 1], self.position[j, 2],
                                  self.position[j, 0]*10**-9+r*2, color=(0, 0, 1), scale_factor=1,
                                  transparent=True, opacity=0.0, resolution=16, figure=fig2)
                return fig2
                # mlab.show()

    def down_sample(self, number_of_downsample=10000):
        if number_of_downsample>self.nb_points:
            pass
        else:
            choice_idx = np.random.choice(self.nb_points, [number_of_downsample, ])
            self.position = self.position[choice_idx]
            self.min_limit = np.amin(self.position, axis=0)  # 1x3
            self.max_limit = np.amax(self.position, axis=0)  # 1x3
            self.range = self.max_limit - self.min_limit
            self.range = np.sqrt(self.range[0] ** 2 + self.range[1] ** 2 + self.range[2] ** 2)  # diagonal distance
            self.center = np.mean(self.position, axis=0)  # 1x3
            self.nb_points = np.shape(self.position)[0]
            self.visible = self.position
            print('after down_sampled points:', self.nb_points, ' points', 'range:', self.range)

    def compute_covariance_mat(self, neighbor_pts=None, rate=0.05, use_dificiency=False):
        """
        weighted covariance matrix, also scatter matrix
        :param neighbor_pts: b x k x d array, b is the number of points , k is the nb_neighbors,
         maybe nan because of different nb_neighbor, d is the dimension
        :param weight:
        :return: b x d x d  covariance array
        """

        if neighbor_pts is None:  # default neighbors are k neighbors
            self.generate_k_neighbor()
            neighbor_pts = self.position[self.point_kneighbors]  # nx3[nxk] = nxkx3
            n = neighbor_pts.shape[0]  # n
            k = neighbor_pts.shape[1]  # k
            tmp = np.ones(shape=(n, k, k)) @ neighbor_pts
            a = neighbor_pts - 1 / k * tmp
            result = np.transpose(a, axes=[0, 2, 1]) @ a * 1 / k  # b x k x n @ # b x n x k = b x k x k

        if neighbor_pts == 'point_rneighbors':
            print('using ball query to find key points')
            self.generate_r_neighbor(range_rate=rate, use_dificiency=use_dificiency)

            whole_weight = 1 / (~np.isnan(self.point_rneighbors)).sum(1)  # do as ISS paper said
            whole_weight[whole_weight == np.inf] = 1  # avoid divided by zero
            # todo: this is an inefficient way
            #  to delete nan effect, so to implement weighted covariance_mat as ISS feature.
            result = np.empty((self.nb_points, 3, 3))
            result[:] = np.nan
            for i in range(self.nb_points):
                idx_this_pts_neighbor = self.point_rneighbors[i, :][~np.isnan(self.point_rneighbors[i, :])].astype(np.int)
                if idx_this_pts_neighbor.shape[0] > 0:

                    weight = np.append(whole_weight[i], whole_weight[idx_this_pts_neighbor])  # add this point

                    neighbor_pts = np.append(self.position[np.newaxis, i, :],
                                             self.position[idx_this_pts_neighbor], axis=0)  # (?+1) x 3 coordinates

                    try:
                        result[i, :, :] = np.cov(neighbor_pts, rowvar=False, ddof=0, aweights=weight)   # 3 x 3
                    except:
                        print('this point:', self.position[i], 'neighbor_pts:', neighbor_pts, 'aweights:', weight)

                else:
                    result[i, :, :] = np.eye(3)

            assert not np.isnan(result.any())
        return result

    def resolution_kpts(self, Importance_Ranking, Voxel_Size, Sampled_Number):
        """
        :param Pointcloud:  点云nx3
        :param Importanace_Ranking:重要度 每个点的重要度排序索引（浮点数） nx1
        :param Voxel_Size:体素大小
        :param Sampled_Number:采样点的个数
        :return:
        """
        ranking_set = {}  # 字典里面每个键代表一个有点的体素
        sampled_pointcloud = np.zeros((Sampled_Number, 3))  # 初始化输出点云数组

        # 计算点云在空间中的立方体
        distance_min = np.amin(self.position, axis=0)

        # 用点云减去最小坐标再除以体素尺寸，得到的nx3为xyz方向上以体素尺寸为单位长度的坐标(浮点数)
        float_index = (self.position - distance_min) / Voxel_Size
        for i in range(len(float_index)):  # 对每个点
            sequence = str(math.ceil(float_index[i][0]))+str(math.ceil(float_index[i][1]))+\
                       str(math.ceil(float_index[i][2]))  # 计算这个点在体素空间中的位置
            if sequence in ranking_set:
                if Importance_Ranking[i] > ranking_set[sequence][0]:
                    ranking_set[sequence] = [Importance_Ranking[i], i]
                 #   print('updates points')
               # print('rejecting points')
            else:
                ranking_set[sequence] = [Importance_Ranking[i], i]  # 如果字典里面没有这个体素，则需要新建一个该体素的键，然后将【重要度，索引】存进去
        if len(ranking_set) < Sampled_Number:
            print("The value of Voxel_Size is too large and needs to be reduced!!!")
            raise ValueError
        sample_sequence = np.zeros(shape=[len(ranking_set), 2])
        for i, j in enumerate(ranking_set):
            sample_sequence[i, :] = ranking_set[j]  # 字典里面每个键都是一个列表，保存的是一个体素内所有点的重要度，取最大的生成一个列表
        sample_sequence = sample_sequence[(-1*sample_sequence[:, 0]).argsort()]  # decsendence sort，得到重要度从大到小的排序
        ind = np.empty((Sampled_Number,))

        for k in range(Sampled_Number):
            sampled_pointcloud[k, :] = self.position[int(sample_sequence[k, 1]), :]
            ind[k] = sample_sequence[k, 1]
        return sampled_pointcloud, ind.astype(int)

    def compute_key_points(self, percentage=0.1, show_result=False, resolution_control=0.02, rate=0.05,
                           use_deficiency=False, show_saliency=False):
        """
        Intrinsic shape signature key point detection, salient point detection
        :param percentage:  ratio of key points to be detected
        :param show_result:
        :param rate: range ratio to search covariance
        :param use_deficiency: whether use deficiency for saliency computation
        :param usr_resolution_control:   whether use resolution_control for key points computation
        :param show_saliency： show the heat map of saliency of a point cloud
        :return:  return nothing, store the key points in self.keypoints
        """

        nb_key_pts = int(self.nb_points*percentage)
        self.weighted_covariance_matix = self.compute_covariance_mat(neighbor_pts='point_rneighbors', rate=rate, use_dificiency=use_deficiency)  # nx3x3

        # compute the eigen value, the smallest eigen value is the variation of the point
        eig_vals = np.linalg.eigvals(self.weighted_covariance_matix)
        assert np.isreal(eig_vals.all())
        eig_vals = np.sort(eig_vals, axis=1)  # n x 3
        smallest_eigvals = eig_vals[:, 0]  # n x 1
        if use_deficiency:
            smallest_eigvals = smallest_eigvals/self.deficiency
        smallest_eigvals[np.isinf(smallest_eigvals)] = 1
        smallest_eigvals = np.real(smallest_eigvals)
        self.saliency = smallest_eigvals
        if resolution_control:
            _, key_pts_idx = self.resolution_kpts(smallest_eigvals, Voxel_Size=self.range*resolution_control, Sampled_Number=nb_key_pts)
        else:
            key_pts_idx = np.argpartition((-1*smallest_eigvals),  nb_key_pts, axis=0)[0:nb_key_pts]  # sort descending

        self.keypoints = key_pts_idx

        if show_result:
            fig2 = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))

            # show the key point cloud
            mlab.points3d(self.position[key_pts_idx, 0], self.position[key_pts_idx, 1],
                          self.position[key_pts_idx, 2],
                          self.position[key_pts_idx, 0] * 10 ** -9 + self.range * 0.005,
                          color=(1, 0, 0),
                          scale_factor=2, figure=fig2)  # tuple(np.random.random((3,)).tolist())

            # show the whole point cloud
            mlab.points3d(self.position[:, 0], self.position[:, 1], self.position[:, 2],
                          self.position[:, 2] * 10 ** -9 + self.range * 0.005,
                          color=(0, 1, 0), scale_factor=1.5)
            mlab.show()

        if show_saliency:
            fig3 = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
            points = mlab.points3d(self.position[:, 0], self.position[:, 1], self.position[:, 2],
                          smallest_eigvals, scale_mode='none', scale_factor=2,
                          colormap='blue-red', figure=fig3)
            cb = mlab.colorbar(object=points, title='saliency')  # legend for the saliency
            # sb = mlab.scalarbar(object=points, title='saliensy')
            cb.label_text_property.bold = 1
            cb.label_text_property.color = (0, 0, 0)
            #mlab.show()
            return fig3

    def estimate_normals(self, max_nn=20, show_result=False):
        o3dpc = o3d.PointCloud()
        o3dpc.points = o3d.Vector3dVector(self.position)

        o3d.estimate_normals(o3dpc, search_param=o3d.KDTreeSearchParamHybrid(
            radius=self.range/30, max_nn=max_nn))

        if show_result:
            fig = mlab.figure(bgcolor=(1, 1, 1), size=(4000, 4000))

            mlab.quiver3d(self.position[:, 0], self.position[:, 1], self.position[:, 2],
                          np.asarray(o3dpc.normals)[:, 0], np.asarray(o3dpc.normals)[:, 1],
                          np.asarray(o3dpc.normals)[:, 2], figure=fig, line_width=2, scale_factor=2, resolution=64)  # normal vectors

            mlab.points3d(self.position[:, 0], self.position[:, 1], self.position[:, 2],
                          self.position[:, 2] * 10**-2 + 2, color=(0, 1, 0),  # +self.range * scale
                          scale_factor=0.2, figure=fig, line_width=2, resolution=64)
            mlab.show()

    def kd_tree(self, show_result=False, colors=None):
        kd_tree = spatial.KDTree(self.position, leafsize=256)

        def leaf_traverse(tree, set):
            if hasattr(tree, 'greater'):
                leaf_traverse(tree.greater, set)
                leaf_traverse(tree.less, set)
            else:
                set.append(tree.idx.tolist())

        if show_result:
            if colors is None:
                colors = np.random.random((100, 3))
            kd_tree = kd_tree.tree
            leaf_set = []
            leaf_traverse(kd_tree, leaf_set)
            fig = mlab.figure(bgcolor=(1, 1, 1), size=(4000, 4000))
            for i, idx in enumerate(leaf_set):
                x = self.position[idx, 0]
                y = self.position[idx, 1]
                z = self.position[idx, 2]
                mlab.points3d(x, y, z, z * 10**-2 + 2, color=tuple(colors[i,:].tolist()),  # +self.range * scale
                              scale_factor=0.8, figure=fig, line_width=2, resolution=64)
            mlab.show()

    def region_growing(self, show_result=False, range_rate=0.05, percentage=0.1, inter_dist=1/100, intra_dist=1/20):
        """

        :param show_result:
        :return: region_growing centers
        """
        self.compute_key_points(rate=range_rate, percentage=percentage*4)
        centers = region_growing_cluster_keypts(self.position[self.keypoints, :], nb_pts=int(self.nb_points*percentage),
                                                pts_range=self.range, inter_dist=inter_dist, intra_dist=intra_dist)

        # print(' region growing key points is :', centers)
        if show_result:
            fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
            # origin points
            mlab.points3d(self.position[:, 0], self.position[:, 1], self.position[:, 2],
                          self.position[:, 2] * 10 ** -9 + self.range * 0.005,
                          color=(0, 1, 0), scale_factor=1.5)
            # salient points
            mlab.points3d(self.position[self.keypoints, 0], self.position[self.keypoints, 1],
                          self.position[self.keypoints, 2],
                          self.position[self.keypoints, 0] * 10 ** -9 + self.range * 0.005,
                          color=(1, 1, 0),
                          scale_factor=2, figure=fig)  # color can be tuple(np.random.random((3,)).tolist())
            # clusters sphere
            r = self.range*inter_dist

            mlab.points3d(centers[:, 0], centers[:, 1], centers[:, 2],
                          centers[:, 0] * 10 ** -9 + r * 2, color=(0, 0, 1), scale_factor=1,
                          transparent=True, resolution=64,
                          opacity=0.03)  # 0.1 for rate 0.05, 0.04 for rate 0.1, 0.025 for rate 0.15 in empirical

            # key points
            mlab.points3d(centers[:, 0], centers[:, 1], centers[:, 2],
                          centers[:, 0] * 10 ** -9 + self.range * 0.005,
                          color=(1, 0, 0), scale_factor=3)
            mlab.show()
        return centers


def point2plane_dist(point, plane):
    """

    :param point: xyz 1x3
    :param plane:  a b c d 1x4, aX+bY+cZ+d=0
    :return: distance scale
    """
    a = plane[0]
    b = plane[1]
    c = plane[2]
    d = plane[3]
    x = point[0]
    y = point[1]
    z = point[2]
    numerator = abs(a*x+b*y+c*z+d)
    denominator = math.sqrt(a*a+b*b+c*c)

    return numerator/denominator


def point2line_dist(point, line_origin, line_vector):
    """

    :param point: 1x3
    :param line_origin: 1x3
    :param line_vector:  1x3
    :return: scale distance
    """
    S = np.linalg.norm(np.cross((point-line_origin), line_vector))
    return S/np.linalg.norm(line_vector)


def show_projection(pc_path='', nb_sample=10000, show_origin=False, add_noise=True):
    """
    show the vary projection result for one point cloud
    :param pc_path: ply file format, numpy array of nx3
    :param nb_sample:  how many points
    :param show_origin:  show before projection or not
    :return:
    """

    plydata = PlyData.read(pc_path)
    vertex = np.asarray([list(subtuple) for subtuple in plydata['vertex'][:]])
    vertex = vertex[:, 0:3]
    for k in range(4):
        np.random.shuffle(vertex)  # will only shuffle the first axis
        pc = vertex[0:nb_sample, :]

        pc_class = PointCloud(np.squeeze(pc))

        # create a pyplot
        fig = plt.figure(figsize=(19, 10))
        if show_origin:
            # the origin point cloud:
            m_fig = mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
            size = pc_class.position[:, 2]*10**-2+1

            mlab.points3d(pc_class.position[:, 0], pc_class.position[:, 1], pc_class.position[:, 2],
                          size, colormap='Spectral', scale_factor=2)
            # mlab.show()
            mlab.gcf().scene.parallel_projection = True  # parallel projection
            f = mlab.gcf()  # this two line for mlab.screenshot to work
            f.scene._lift()
            img = mlab.screenshot(figure=m_fig)
            mlab.close()
            ax = fig.add_subplot(2, 3, 1)
            ax.imshow(img)
            ax.set_axis_off()

            for i in range(5):
                pc_class.half_by_plane(grid_resolution=(400, 400))

                x = 30  # cone offset todo you have to manually ajust the cone origin here !!!
                y = -30
                z = 50
                u = pc_class.plane_project_points[1, 0] - pc_class.visible[1, 0]
                v = pc_class.plane_project_points[1, 1] - pc_class.visible[1, 1]
                w = pc_class.plane_project_points[1, 2] - pc_class.visible[1, 2]
                vector = np.stack([u, v, w])
                vector = vector / np.linalg.norm(vector) * 280
                u = vector[0]
                v = vector[1]
                w = vector[2]

                m_fig = mlab.figure(bgcolor=(0, 0, 0), size=(1000, 1000))

                mlab.points3d(pc_class.visible[:, 0], pc_class.visible[:, 1], pc_class.visible[:, 2],
                              pc_class.visible[:, 2]*10**-2+1, colormap='Spectral', scale_factor=2, figure=m_fig)

                mlab.quiver3d(x, y, z, u, v, w, colormap='RdYlGn', mode='cone',   # show the cone
                              figure=m_fig, scale_factor=0.1, line_width=2.0)    # color map can be RdYlGn, Spectral

                mlab.gcf().scene.parallel_projection = True  # parallel projection
                f = mlab.gcf()  # this two line for mlab.screenshot to work
                f.scene._lift()
                # mlab.show()  # for testing
                img = mlab.screenshot(figure=m_fig)
                mlab.close()
                ax = fig.add_subplot(2, 3, i+2)
                ax.imshow(img)
                ax.set_axis_off()

            # # the projected point cloud
            # m_fig = mlab.figure(bgcolor=(0, 0, 0), size=(1000, 1000))
            # x = pc_class.visible[:, 0]
            # y = pc_class.visible[:, 1]
            # z = pc_class.visible[:, 2]
            # u = pc_class.plane_project_points[:, 0] - pc_class.visible[:, 0]
            # v = pc_class.plane_project_points[:, 1] - pc_class.visible[:, 1]
            # w = pc_class.plane_project_points[:, 2] - pc_class.visible[:, 2]
            # mlab.quiver3d(x, y, z, u, v, w, colormap='RdYlGn', mode='2darrow', figure=m_fig)  # scale_factor=1
            # mlab.gcf().scene.parallel_projection = True  # parallel projection
            # f = mlab.gcf()  # this two line for mlab.screenshot to work
            # f.scene._lift()
            # img = mlab.screenshot(figure=m_fig)
            # mlab.close()
            # ax = fig.add_subplot(2, 2, 3)
            # ax.imshow(img)
            # ax.set_axis_off()

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()

        else:

            # only shows the projection front point cloud
            for i in range(2):
                # add the screen capture0
                if add_noise:
                    pc_class.add_noise()
                    pc_class.add_outlier()

                pc_class.half_by_plane(grid_resolution=(200, 200))

                x = pc_class.visible[:, 0]
                y = pc_class.visible[:, 1]
                z = pc_class.visible[:, 2]
                u = pc_class.plane_project_points[:, 0] - pc_class.visible[:, 0]

                v = pc_class.plane_project_points[:, 1] - pc_class.visible[:, 1]

                w = pc_class.plane_project_points[:, 2] - pc_class.visible[:, 2]

                # x = pc_class.position[:, 0]
                # y = pc_class.position[:, 1]
                # z = pc_class.position[:, 2]

                m_fig = mlab.figure(bgcolor=(1, 1, 1), size=(960, 540))

                #mlab.quiver3d(x, y, z, u, v, w, colormap='RdYlGn', mode='2darrow', figure=m_fig)   # scale_factor=0.05

                mlab.points3d(x, y, z, z*10**-3+1, colormap='Spectral', scale_factor=2, figure=m_fig)

                mlab.gcf().scene.parallel_projection = True  # parallel projection
                mlab.show() # for testing, annotate this for automation
                f = mlab.gcf()  # this two line for mlab.screenshot to work
                f.scene._lift()
                img = mlab.screenshot()
                mlab.close()

                ax = fig.add_subplot(1, 2, i+1)
                ax.imshow(img)
                ax.set_axis_off()

            plt.subplots_adjust(wspace=0, hspace=0)
            #plt.show()
        # plt.savefig('/home/sjtu/Pictures/asy/point clouds/dataset/lab1_'+str(k+1)+'.png')


def chamfer_dist(arr1, arr2, chose_rate=1.0):
    """
    return the chamfer distance of two point cloud, point cloud don't have to be the same length
    :param arr1: nx3 np array
    :param arr2: mx3 np array
    :param chose_rate: chose minimum distance in the correspondense such that partial point cloud can be measured
    :return: chamfer distance of two point cloud
    """
    if isinstance(arr1, PointCloud):
        arr1 = arr1.position
    if isinstance(arr2, PointCloud):
        arr2 = arr2.position
    assert np.shape(arr1)[1] == np.shape(arr2)[1]

    def compute_dist(A, B, chose_rate=chose_rate):
        m = np.shape(A)[0]
        n = np.shape(B)[0]
        k_th = np.rint(m * chose_rate) - 1
        k_th = k_th.astype(np.int)
        dim = np.shape(A)[1]
        dist = np.zeros((m, ))
        for k in range(m):
            C = np.ones([n, 1]) @ A[[k], :]
            D = np.multiply((C-B), (C-B))
            D = np.sqrt(D@np.ones((dim, 1)))
            dist[k] = np.min(D)
        result = np.partition(dist, k_th)
        result = result[:k_th]
        return np.mean(result)
    return max([compute_dist(arr1, arr2), compute_dist(arr2, arr1)])


def hausdorff_dist(arr1, arr2, chose_rate=1):
    """
    return the hausdorff distance of two point cloud, point cloud don't have to be the same length
    :param arr1: nx3 np array
    :param arr2: mx3 np array
    :param chose_rate: chose minimum distance in the correspondense such that partial point cloud can be measured, [0,1]
    :return: hausdorff distance of two point clouds
    """
    if isinstance(arr1, PointCloud):
        arr1 = arr1.position
    if isinstance(arr2, PointCloud):
        arr2 = arr2.position

    assert np.shape(arr1)[1] == np.shape(arr2)[1]

    def compute_dist(A, B, chose_rate=chose_rate):
        m = np.shape(A)[0]
        n = np.shape(B)[0]
        k_th = np.rint(m*chose_rate)-1
        k_th = k_th.astype(np.int)
        dim = np.shape(A)[1]
        dist = np.zeros((m, ))
        for k in range(m):
            C = np.ones([n, 1]) @ A[[k], :]
            D = np.multiply((C-B), (C-B))
            D = np.sqrt(D@np.ones((dim, 1)))
            dist[k] = np.min(D)
        result = np.partition(dist, k_th)
        return result[k_th]
    return max([compute_dist(arr1, arr2), compute_dist(arr2, arr1)])


def robust_test_kpts(pc_path, samples=15, chamfer=True, percentage=0.1, range_rate=0.05, region_growing=False, chose_rate=0.5):

    f_list = [pc_path + '/' + i for j, i in enumerate(os.listdir(pc_path)) if os.path.splitext(i)[1] == '.txt' and j<samples]
    distance_array = []
    for i in f_list:
        pc1 = PointCloud(i)
        pc1.compute_key_points(percentage=percentage, rate=range_rate)

        for j in f_list:
            if j != i:
                pc2 = PointCloud(j)
                if region_growing:
                    if chamfer:
                        distance_array.append(
                            chamfer_dist(pc1.region_growing(range_rate=range_rate, percentage=percentage),
                                         pc2.region_growing(range_rate=range_rate, percentage=percentage),
                                         chose_rate=chose_rate))
                    else:
                        distance_array.append(
                            hausdorff_dist(pc1.region_growing(range_rate=range_rate, percentage=percentage),
                                           pc2.region_growing(range_rate=range_rate, percentage=percentage),
                                           chose_rate=chose_rate))
                else:
                    pc2.compute_key_points(percentage=percentage, rate=range_rate)
                    if chamfer:
                        distance_array.append(
                            chamfer_dist(pc1.position[pc1.keypoints, :], pc2.position[pc2.keypoints, :], chose_rate=chose_rate))
                    else:
                        distance_array.append(
                            hausdorff_dist(pc1.position[pc1.keypoints, :], pc2.position[pc2.keypoints, :], chose_rate=chose_rate))
    mean = np.mean(distance_array)
    print('mean is ', mean, 'distance array is ', distance_array)





if __name__ == "__main__":
    # org = cv2.imread('/home/sjtu/Pictures/asy/point clouds/1th_image_30th_epoch.png')
    # now = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('aaa', now)
    # cv2.imwrite('/home/sjtu/Pictures/asy/point clouds/1th_image_30th_epoch_gray.png', now)
    # cv2.waitKey(0)  # Waits forever for user to press any key
    # cv2.destroyAllWindows()  # Closes displayed windows
    # a = time.time()
    # pc = PointCloud(np.random.random(size=(1024, 3)))
    # readh5 = h5py.File('/media/sjtu/software/ASY/pointcloud/train_set4noiseout/project_data.h5')  # file path
    # pc_tile = readh5['train_set'][:]  # 20000 * 1024 * 3
    # pc = np.squeeze(pc_tile[5001, :, :])

    # pc.octree()
    # print(pc.root)
    # mlab.figure(size=(1000, 1000))
    # for child in pc.root.children:
    #     for grandchild in child.children:
    #         if grandchild is not None:
    #             # for grandgrandchild in grandchild.children:
    #             #     if grandgrandchild is not None:
    #             mlab.points3d(grandchild.data[:, 0], grandchild.data[:, 1], grandchild.data[:, 2],
    #                           color=tuple(np.random.random((3,)).tolist()), scale_factor=0.05)   # tuple(np.random.random((3,)).tolist())
    # print(time.time() - a, 's')
    # mlab.show()
    #base_path = '/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/8object/lab1'
    #robust_test_kpts(pc_path=base_path, percentage=0.1, range_rate=0.05, region_growing=True, chamfer=False, chose_rate=0.7)


    #    pc.compute_key_points(use_deficiency=True, show_saliency=True)
    #    pc.compute_key_points(show_saliency=True)

    # pc1.down_sample(number_of_downsample=4096)
    # colors = np.random.random((100, 3))
    # pc1.kd_tree(show_result=True, colors=colors)

    # pc.estimate_normals(max_nn=10, show_result=True)
    # pc.down_sample(number_of_downsample=1024)
    # pc.estimate_normals(max_nn=10, show_result=True)
    # pc2 = PointCloud(pc_path2)

    # pc4.octree(show_layers=1, colors=colors)

    # base_path = '/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/noise_out lier'
    # pc_path1 = base_path + '/lab4/final.ply'
    # pc = PointCloud(base_path + '/lab4/final.ply')
    # pc.down_sample(2048)
    # pc.compute_key_points(rate=0.1, show_result=True, resolution_control=0)
    # pc.compute_key_points(rate=0.1, show_result=True, resolution_control=0.02)
    #

    # show_projection(pc_path=pc_path1, nb_sample=10000, show_origin=False, add_noise=False)
    # pc = PointCloud(pc_path1)
    # pc.down_sample(number_of_downsample=10000)
    # pc.compute_key_points(show_saliency=True, rate=0.05, use_deficiency=True)
    # pc.compute_key_points(show_saliency=True, rate=0.05)
    # pc.compute_key_points(show_saliency=True, rate=0.05)
    # pc.compute_key_points(show_saliency=True, rate=0.1)
    # key_pts = region_growing_cluster_keypts(pc)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # mlab.points3d(key_pts[:, 0], key_pts[:, 1], key_pts[:, 2], key_pts[:, 2] * 10 ** -9 + 1, color=(1,0,0), scale_factor=3)
    # mlab.points3d(pc.position[:, 0], pc.position[:, 1], pc.position[:, 2], pc.position[:, 2] * 10 ** -9 + 1, color=(0, 1, 0), scale_factor=1)
    # mlab.show()
    base_path = '/media/sjtu/software/ASY/pointcloud/lab scanned workpiece'
    f_list = [base_path + '/' + i for i in os.listdir(base_path) if os.path.splitext(i)[1] == '.ply']
    for i in f_list:
        pc = PointCloud(i)
        pc.cut_by_plane()
        pc2 = PointCloud(pc.visible)
        pc2.half_by_plane(show_result=True)

    pass