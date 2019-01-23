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
import time
from matplotlib import pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab
from scipy.spatial import distance
from plyfile import PlyData, PlyElement
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


def show_all(point_cloud, color=None , plot_plane=False, plot_arrow=True):
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


def show_trans(point_cloud1, point_cloud2, color1=None, color2=None, color3=None, color4=None,
               color5=None, color6=None, color7=None, color8=None, use_mayavi=True, scale=4):
    """
    plot a batch of point clouds
    :param point_cloud1: Bx1024x3 point_cloud2: Bx1024x3 np array
    :return:
    """
    a1, b1, c1 = point_cloud1[:, :, 0], point_cloud1[:, :, 1], point_cloud1[:, :, 2]  # Bxnx1
    a1, b1, c1 = np.squeeze(a1), np.squeeze(b1), np.squeeze(c1)  # Bxn
    a2, b2, c2 = point_cloud2[:, :, 0], point_cloud2[:, :, 1], point_cloud2[:, :, 2]  # Bxnx1
    a2, b2, c2 = np.squeeze(a2), np.squeeze(b2), np.squeeze(c2)  # Bxn

    ax = plt.subplot(111, projection='3d', facecolor='w')
    mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    ax.set_axis_off()
    B = point_cloud1.shape[0] #batch
    colorset = [color1, color2, color3, color4, color5, color6, color7, color8]
    dark_multiple = 2.5   # greater than 1
    for idx, i in enumerate(colorset):
        if idx % 2 == 0 and i is None:
            colorset[idx] = tuple((1/dark_multiple)*np.random.random([3, ]))
            colorset[idx+1] = [i * dark_multiple for i in colorset[idx]]

    colorset = [tuple(i) for i in colorset]

    for i in range(B):

        if i % 4 == 0:
            if not use_mayavi:
                ax.scatter(a1[i, :], b1[i, :], c1[i, :], color=colorset[0], s=5)
                ax.scatter(a2[i, :], b2[i, :], c2[i, :], color=colorset[1], s=5)
            else:
                mlab.points3d(a1[i, :], b1[i, :], c1[i, :], c1[i, :] * 10**-9 + scale, color=colorset[0], scale_factor=1)
                mlab.points3d(a2[i, :], b2[i, :], c2[i, :], c2[i, :] * 10**-9 + scale, color=colorset[1], scale_factor=1)
        elif i % 4 == 1:
            if not use_mayavi:
                ax.scatter(a1[i, :], b1[i, :], c1[i, :], color=colorset[2], s=5)
                ax.scatter(a2[i, :], b2[i, :], c2[i, :], color=colorset[3], s=5)
            else:
                mlab.points3d(a1[i, :], b1[i, :], c1[i, :], c1[i, :] * 10**-9 + scale, color=colorset[2], scale_factor=1)
                mlab.points3d(a2[i, :], b2[i, :], c2[i, :], c2[i, :] * 10**-9 + scale, color=colorset[3], scale_factor=1)
        elif i % 4 == 2:
            if not use_mayavi:
                ax.scatter(a1[i, :], b1[i, :], c1[i, :], color=colorset[4], s=5)
                ax.scatter(a2[i, :], b2[i, :], c2[i, :], color=colorset[5], s=5)
            else:
                mlab.points3d(a1[i, :], b1[i, :], c1[i, :], c1[i, :] * 10**-9 + scale, color=colorset[4], scale_factor=1)
                mlab.points3d(a2[i, :], b2[i, :], c2[i, :], c2[i, :] * 10**-9 + scale, color=colorset[5], scale_factor=1)
        elif i % 4 == 3:
            if not use_mayavi:
                ax.scatter(a1[i, :], b1[i, :], c1[i, :], color=colorset[6], s=5)
                ax.scatter(a2[i, :], b2[i, :], c2[i, :], color=colorset[7], s=5)
            else:
                mlab.points3d(a1[i, :], b1[i, :], c1[i, :], c1[i, :] * 10**-9 + scale, color=colorset[6], scale_factor=1)
                mlab.points3d(a2[i, :], b2[i, :], c2[i, :], c2[i, :] * 10**-9 + scale, color=colorset[7], scale_factor=1)

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

    batch_out = tf.matmul(homo, batch_out)  # Bx4x4 batch multiply Bx4xn, points coordinates in column vector,left-handed rotation matrix

    batch_out = tf.div(batch_out, tf.slice(batch_out, [0, 3, 0], [batch, 1, num]))  # every element divided the
    # last element to get true coordinates
    batch_out = tf.slice(batch_out, [0, 0, 0], [batch, 3, num]) # Bx3xn
    batch_out = tf.transpose(batch_out, perm=[0, 2, 1])    # Bxnx3

    return batch_out




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
        print(self.nb_points, ' points', 'range:', self.range)

    def half_by_plane(self, plane=None, n=1024, grid_resolution=(256, 256)):
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
                raise ValueError('value error')
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

    def show(self, not_show=False, scale=0.4):
        mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
        fig = mlab.points3d(self.position[:, 0], self.position[:, 1], self.position[:, 2],
                            self.position[:, 2] * 10**-2 + 2, color=(0, 1, 0),  # +self.range * scale
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
        self.position[inds] = self.center + -self.range / 6 + self.range / 3 * np.random.random(size=(len(inds), 3))

    def normalize(self):
        self.position -= self.center
        self.position /= self.range
        self.center = np.mean(self.position, axis=0)
        self.min_limit = np.amin(self.position, axis=0)
        self.max_limit = np.amax(self.position, axis=0)
        self.range = self.max_limit - self.min_limit
        self.range = np.sqrt(self.range[0] ** 2 + self.range[1] ** 2 + self.range[2] ** 2)
        print('center: ', self.center, 'range:', self.range)

    def octree(self):
        def width_first_traversal(position, size, data):
            root = OctNode(position, size, data)

            min = root.position + [-root.size / 2, -root.size / 2, -root.size / 2]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            lbd = root.data[np.all(media, axis=1), :]
            if lbd.shape[0] > 1:
                root.dbl = width_first_traversal(
                    root.position + [-1 / 4 * root.size, -1 / 4 * root.size, -1 / 4 * root.size],
                    root.size * 1 / 2, data=lbd)

            min = root.position + [0, -root.size / 2, -root.size / 2]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            rbd = root.data[np.all(media, axis=1), :]
            if rbd.shape[0] > 1:
                root.dbr = width_first_traversal(
                    root.position + [1 / 4 * root.size, -1 / 4 * root.size, -1 / 4 * root.size],
                    root.size * 1 / 2, data=rbd)

            min = root.position + [-root.size / 2, 0, -root.size / 2]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            lfd = root.data[np.all(media, axis=1), :]
            if lfd.shape[0] > 1:
                root.dfl = width_first_traversal(
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

            min = root.position + [-root.size / 2, -root.size / 2, 0]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            lbu = root.data[np.all(media, axis=1), :]
            if lbu.shape[0] > 1:
                root.ubl = width_first_traversal(
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

            min = root.position + [-root.size / 2, 0, 0]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            lfu = root.data[np.all(media, axis=1), :]
            if lfu.shape[0] > 1:
                root.ufl = width_first_traversal(
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
            root.children = [root.ubl, root.ubr, root.ufl, root.ufr, root.dbl, root.dbr, root.dfl, root.dfr]
            return root

        self.root = width_first_traversal(self.center, size=max(self.max_limit - self.min_limit), data=self.position)

        def maxdepth(node):
            if not any(node.children):
                return 0
            else:
                return max([maxdepth(babe) + 1 for babe in node.children if babe is not None])

        self.depth = maxdepth(self.root)

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

    def generate_r_neighbor(self, rate=0.05, show_result=False):

        assert 0 < rate < 1
        r = self.range * rate
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

        if show_result:
            if self.keypoints is not None:

                fig2 = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))

                neighbor_idx = self.point_rneighbors[self.keypoints, :]
                neighbor_idx = neighbor_idx[~np.isnan(neighbor_idx)].astype(np.int32)
                # show the neighbor point cloud
                mlab.points3d(self.position[neighbor_idx, 0], self.position[neighbor_idx, 1],
                              self.position[neighbor_idx, 2],
                              self.position[neighbor_idx, 0] * 10 ** -9 + self.range * 0.005,
                              color=(1, 1, 0),
                              scale_factor=2, figure=fig2)  # color can be tuple(np.random.random((3,)).tolist())

                # show the whole point cloud
                mlab.points3d(self.position[:, 0], self.position[:, 1], self.position[:, 2],
                              self.position[:, 2] * 10 ** -9 + self.range * 0.005,
                              color=(0, 1, 0), scale_factor=1.5)

                # show the sphere on the neighbor
                mlab.points3d(self.position[self.keypoints, 0], self.position[self.keypoints, 1], self.position[self.keypoints, 2],
                              self.position[self.keypoints, 0] * 10 ** -9 + r * 2, color=(0, 0, 1), scale_factor=1,
                              transparent=True, opacity=0.025)  # 0.1 for rate 0.05, 0.04 for rate 0.1, 0.025 for rate 0.15

                # show the key points
                mlab.points3d(self.position[self.keypoints, 0], self.position[self.keypoints, 1],
                              self.position[self.keypoints, 2],
                              self.position[self.keypoints, 0] * 10 ** -9 + self.range * 0.005,
                              color=(1, 0, 0), scale_factor=2,)

                mlab.show()

            else:
                for i in range(5):
                    j = np.random.choice(self.nb_points, 1)  # choose one random point index
                    fig2 = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))

                    neighbor_idx = self.point_rneighbors[j, :]
                    neighbor_idx = neighbor_idx[~np.isnan(neighbor_idx)].astype(np.int32)
                    # show the neighbor point cloud
                    mlab.points3d(self.position[neighbor_idx, 0], self.position[neighbor_idx, 1],
                                  self.position[neighbor_idx, 2],
                                  self.position[neighbor_idx, 0] * 10**-9 + self.range * 0.005,
                                  color=tuple(np.random.random((3,)).tolist()),
                                  scale_factor=2, figure=fig2)  # tuple(np.random.random((3,)).tolist())

                    # show the whole point cloud
                    mlab.points3d(self.position[:, 0], self.position[:, 1], self.position[:, 2],
                                  self.position[:, 2] * 10**-9 + self.range * 0.005,
                                  color=(0, 1, 0), scale_factor=1)

                    # show the sphere on the neighbor
                    mlab.points3d(self.position[j, 0], self.position[j, 1], self.position[j, 2],
                                  self.position[j, 0]*10**-9+r*2, color=(0, 0, 1), scale_factor=1,
                                  transparent=True, opacity=0.3)
                    mlab.show()

    def down_sample(self, number_of_downsample=10000):
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

    def compute_covariance_mat(self, neighbor_pts=None, rate=0.05):
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
            self.generate_r_neighbor(rate=rate)
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

    def compute_key_points(self, percentage=0.1, show_result=False, rate=0.05):
        """
        Intrinsic shape signature key point detection
        :param percentage:  10%
        :param show_result:
        :param rate:
        :return:  return nothing, store the key points in self.keypoints
        """

        # todo resolution control, voxelization for the key p
        nb_key_pts = int(self.nb_points*percentage)
        self.weighted_covariance_matix = self.compute_covariance_mat(neighbor_pts='point_rneighbors', rate=rate)  # nx3x3

        # compute the eigen value, the smallest eigen value is the variation of the point
        eig_vals = np.linalg.eigvals(self.weighted_covariance_matix)
        assert np.isreal(eig_vals.all())
        eig_vals = np.sort(eig_vals, axis=1)  # n x 3
        smallest_eigvals = eig_vals[:, 0]  # nx1
        key_pts_idx = np.argpartition(smallest_eigvals,  nb_key_pts, axis=0)[0:nb_key_pts]
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


def show_projection(pc_path='', nb_sample=10000, show_origin=False):
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

                x = 30  # cone offset todo you have to manually ajust the cone origin
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
                pc_class.add_noise()
                pc_class.add_outlier()

                pc_class.half_by_plane(grid_resolution=(256, 256))

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

                mlab.points3d(x, y, z, z*10**-3+1, colormap='Spectral', scale_factor=3, figure=m_fig)

                mlab.gcf().scene.parallel_projection = True  # parallel projection
                # mlab.show() # for testing, annotate this for automation
                f = mlab.gcf()  # this two line for mlab.screenshot to work
                f.scene._lift()
                img = mlab.screenshot()
                mlab.close()

                ax = fig.add_subplot(1, 2, i+1)
                ax.imshow(img)
                ax.set_axis_off()

            plt.subplots_adjust(wspace=0, hspace=0)
            #plt.show()
        plt.savefig('/home/sjtu/Pictures/asy/point clouds/dataset/lab1_'+str(k+1)+'.png')

if __name__ == "__main__":

    # show_projection(pc_path='fullbodyanya1.txt', show_origin=True)
    # org = cv2.imread('/home/sjtu/Pictures/asy/point clouds/1th_image_30th_epoch.png')
    # now = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('aaa', now)
    # cv2.imwrite('/home/sjtu/Pictures/asy/point clouds/1th_image_30th_epoch_gray.png', now)
    # cv2.waitKey(0)  # Waits forever for user to press any key
    # cv2.destroyAllWindows()  # Closes displayed windows
    # a = time.time()
    #
    # pc = PointCloud(np.random.random(size=(1024, 3)))
    # readh5 = h5py.File('/media/sjtu/software/ASY/pointcloud/train_set4noiseout/project_data.h5')  # file path
    # pc_tile = readh5['train_set'][:]  # 20000 * 1024 * 3
    # pc = np.squeeze(pc_tile[5001, :, :])
    # # pc = np.loadtxt('model.txt')
    # pc = PointCloud(pc)
    # print('limit:', max(pc.max_limit-pc.min_limit))
    #
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

    pc_path1 = '/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/noise_out lier/lab1/final.ply'
    # base_path = '/media/sjtu/software/ASY/pointcloud/lab scanned workpiece'
    # pc = np.loadtxt(base_path + '/lab4/lab4_project' + str(1) + '.txt')
    # pc = PointCloud(pc)
    # pc.compute_key_points(rate=0.05, show_result=True)
    # pc.generate_r_neighbor(rate=0.15, show_result=True)

    show_projection(pc_path=pc_path1, nb_sample=10000, show_origin=False)