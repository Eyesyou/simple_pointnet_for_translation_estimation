import argparse
import math
import h5py
import numpy as np
import socket
import importlib
import os
import sys
import matplotlib
import open3d as o3d
import time
from matplotlib import pyplot as plt
import random
from time import time
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab
from scipy.spatial import distance
from plyfile import PlyData, PlyElement
from scipy import spatial  # for tree structure
from sklearn import (manifold, datasets, decomposition, ensemble,  random_projection)
import cv2

from show_pc import PointCloud

def saliancey2range(resolution_control=0.005):
    for j, i in enumerate(f_list):
        print(' point cloud is', i)
        pc = PointCloud(i)
        pc.down_sample(number_of_downsample=2048)
        for k in range(4):
            if k == 0:
                k = -0.5
            fig = pc.compute_key_points(percentage=0.1, show_result=False, resolution_control=resolution_control, rate=0.05 * k + 0.05,
                                        use_deficiency=False, show_saliency=True)
            f = mlab.gcf()  # this two line for mlab.screenshot to work
            f.scene._lift()
            img = mlab.screenshot()
            mlab.savefig(filename=str(j) + str(k) + '_without.png')
            mlab.close()
            fig = pc.compute_key_points(percentage=0.1, show_result=False, resolution_control=resolution_control, rate=0.05 * k + 0.05,
                                        use_deficiency=True, show_saliency=True)
            f = mlab.gcf()  # this two line for mlab.screenshot to work
            f.scene._lift()
            img = mlab.screenshot()
            mlab.savefig(filename=str(j) + str(k) + '_with.png')
            mlab.close()

        del pc


def rneighbor2range():
    for j, i in enumerate(f_list):
        print(' point cloud is', i)
        pc = PointCloud(i)
        pc.down_sample(number_of_downsample=2048)
        for k in range(4):
            fig = pc.generate_r_neighbor(range_rate=0.025*k+0.025, show_result=True)
            pc.keypoints = None
            f = mlab.gcf()  # this two line for mlab.screenshot to work
            f.scene._lift()
            mlab.savefig(filename=str(j) + str(k) + 'r.png')
            mlab.close()
        del pc


def plot_embedding_3d(X, Y, title=None, point_clouds=None):
    """

    :param X:  B x n features
    :param Y:  (B, ) labels
    :param title:
    :return:
    """
    Y = Y.astype(np.int)
    tsne = manifold.TSNE(n_components=3, perplexity=30, early_exaggeration=4.0, learning_rate=1000, init='pca',
                         random_state=0, n_iter=10000, verbose=0, method='barnes_hut', angle=0.5)
    X_tsne = tsne.fit_transform(X)
    n = np.shape(X_tsne)[0]

    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X_tsne, axis=0), np.max(X_tsne, axis=0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)  # n x 3

    #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    mfig = mlab.figure(size=(1920, 1080), bgcolor=(1, 1, 1))

    point_clouds = point_clouds / 1000  # n x 1024 x 3 you have to shrink point cloud for clearity
    nb_points = np.shape(point_clouds)[1]
    point_clouds += np.tile(X_tsne[:, np.newaxis, :], (1, nb_points, 1))  # n x 1024 x 3

    for i in range(n):

        ax.text(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2], str(Y[i]),
                 color=plt.cm.Set1((Y[i]+1) / 10.),  # plt.cm.Set1(y[i] / 10.)
                 fontdict={'weight': 'bold', 'size': 24})
        if point_clouds is not None:
            mlab.points3d(point_clouds[i, :, 0], point_clouds[i, :, 1], point_clouds[i, :, 2],
                          point_clouds[i, :, 2] * 10 ** -9 + 0.5,
                          color=plt.cm.Set1((Y[i] + 1) / 10.)[:3],
                          scale_factor=0.003, figure=mfig)
        else:

            mlab.points3d(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2], X_tsne[i, 2]*10**-9+0.5,
                          color=plt.cm.Set1((Y[i] + 1) / 10.)[:3],
                          scale_factor=0.01, figure=mfig)

    ax.grid(False)
    if title is not None:
        plt.title(title)
    plt.grid(b=None)
    plt.show()
    mlab.show()

def noise_outliers(pointclous):
    fig = plt.figure(figsize=(38, 20), dpi=600, facecolor='w')
    for j, i in enumerate(pointclous):
        pc = PointCloud(i)
        pc.down_sample(number_of_downsample=2048)

        for k in range(4):
            if k == 3:
                k = 4
            pc.add_noise(factor=k*0.02)
            pc.add_outlier(factor=k*0.02)
            m_fig = mlab.figure(bgcolor=(1, 1, 1))

            mlab.points3d(pc.position[:, 0], pc.position[:, 1], pc.position[:, 2],
                          pc.position[:, 2] * 10 ** -2 + 1, colormap='Spectral', scale_factor=2, figure=m_fig)

            # mlab.gcf().scene.parallel_projection = True  # parallel projection
            f = mlab.gcf()  # this two line for mlab.screenshot to work
            f.scene._lift()
            # mlab.show()  # for testing
            img = mlab.screenshot(figure=m_fig)
            mlab.close()
            if k == 4:
                k = 3
            ax = fig.add_subplot(4, 8, (j+1)+k*8)
            ax.imshow(img)
            ax.set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def vis_first_layer(net_input_point_cloud, first_layrer_output, vis_rate=1/10, square_plot_nb = 16):
    """
    input to the network should be cubic grid point cloud
    :param net_input_point_cloud: 1024 x 3
    :param first_layrer_output:  1024 x 64   number of points x features
    :return:
    """
    assert np.shape(first_layrer_output)[1] > square_plot_nb
    nb_points = np.shape(net_input_point_cloud)[0]
    fig = plt.figure(figsize=(19, 10), dpi=300, facecolor='w')

    small_value = np.ones([square_plot_nb, 2])/.0  # create array of infs

    for i in range(np.shape(first_layrer_output)[1]):
        values = np.reshape(first_layrer_output[:, i], [-1])
        idx = np.argpartition(values, int(nb_points*vis_rate))
        idx = idx[:int(nb_points*vis_rate)]
        mean_value = np.mean(idx)

        if mean_value < np.max(small_value[:, 0]):
            small_value[np.argmax(small_value, axis=0), :] = np.array([mean_value, i])

    for j in range(np.shape(small_value)[0]):

        i = int(small_value[j, 1])
        values = np.reshape(first_layrer_output[:, i], [-1])
        idx = np.argpartition(values, int(nb_points * vis_rate))
        idx = idx[:int(nb_points * vis_rate)]

        m_fig = mlab.figure(size=(500, 500), bgcolor=(1, 1, 1))

        points = mlab.points3d(net_input_point_cloud[idx, 0], net_input_point_cloud[idx, 1], net_input_point_cloud[idx, 2],
                      net_input_point_cloud[idx, 2] * 10 ** -2 + 1, colormap='Spectral', scale_factor=0.1, figure=m_fig,
                      resolution=64)
        points.glyph.glyph_source.glyph_source.phi_resolution = 64
        points.glyph.glyph_source.glyph_source.theta_resolution = 64
        # mlab.gcf().scene.parallel_projection = True  # parallel projection
        f = mlab.gcf()  # this two line for mlab.screenshot to work
        f.scene._lift()
        # mlab.show()  # for testing
        img = mlab.screenshot(figure=m_fig)
        ax = fig.add_subplot(int(math.sqrt(square_plot_nb)), int(math.sqrt(square_plot_nb)), (j + 1))
        ax.imshow(img)
        ax.set_axis_off()
        mlab.close()

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def feature_mean_deviation(pc_path, samples=15, chamfer=True, method='ball'):
    """

    :param pc_path:
    :param samples:
    :param chamfer:
    :param method: ball-default 0.05*range knn-default 64 points octree-default 64 points kdtree-default 3 layer
    :return:
    """
    f_list = [pc_path + '/' + i for j, i in enumerate(os.listdir(pc_path)) if
              os.path.splitext(i)[1] == '.txt' and j < samples]
    for i in f_list:
        pc = PointCloud(i)
        if method =='ball':
            features = pc.generate_r_neighbor()

        elif method =='knn':
            pass
        elif method == 'octree':
            pass
        elif method == 'kdtree':
            pass


def icp_registration_error(sourcepc, targetpc, threshold=0.02):

    source_o3dpc = o3d.geometry.PointCloud()
    source_o3dpc.points = o3d.utility.Vector3dVector(sourcepc.position)
    target_o3dpc = o3d.geometry.PointCloud()
    target_o3dpc.points = o3d.utility.Vector3dVector(targetpc.position)

    threshold = 0.02

    draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    evaluation = o3d.evaluate_registration(source, target,
            threshold, trans_init)
    print('evaluation:', evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.registration_icp(source, target, threshold, init=np.eye(4),
                                   estimation_method=o3d.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("reg_p2p Transformation is:")
    print(reg_p2p.transformation)
    print("transformation type is :", type(reg_p2p.transformation))
    target.transform(trans_init)
    draw_registration_result(source, target, reg_p2p.transformation)


def knn_plot(pc_path=''):

    f_list = [base_path + '/' + i for i in os.listdir(base_path) if os.path.splitext(i)[1] == '.ply']
    for j, i in enumerate(f_list):
        if j < 4:
            pc = PointCloud(i)
            pc.down_sample(number_of_downsample=4096)
            pc.add_noise(factor=0.04)
            pc.add_outlier(factor=0.04)
            fig = pc.compute_key_points(percentage=0.02, resolution_control=1/15, rate=0.05, use_deficiency=False,show_result=True) # get the key points id

            f = mlab.gcf()  # this two line for mlab.screenshot to work
            f.scene._lift()
            mlab.savefig(filename=str(j) + '_0.png')
            mlab.close()

            colorset = np.random.random((100, 3))
            fig = pc.generate_k_neighbor(k=32, show_result=True, colorset=colorset)

            f = mlab.gcf()  # this two line for mlab.screenshot to work
            f.scene._lift()
            mlab.savefig(filename=str(j) + '_1.png')
            mlab.close()

            fig = pc.generate_k_neighbor(k=64, show_result=True, colorset=colorset)
            f = mlab.gcf()  # this two line for mlab.screenshot to work
            f.scene._lift()
            mlab.savefig(filename=str(j) + '_2.png')
            mlab.close()

            fig = pc.generate_k_neighbor(k=128, show_result=True, colorset=colorset)
            f = mlab.gcf()  # this two line for mlab.screenshot to work
            f.scene._lift()
            mlab.savefig(filename=str(j) + '_3.png')
            mlab.close()


def key_points_plot(flist):
    for i in flist:
        Pc = PointCloud(i)
        Pc.down_sample(4096)
        fig = Pc.compute_key_points(percentage=0.1, resolution_control=None, show_result=True)

        f = mlab.gcf()  # this two line for mlab.screenshot to work
        f.scene._lift()
        img = mlab.screenshot()
        mlab.savefig(filename=str(i) + 'key_points.png')
        mlab.close()

        fig = Pc.compute_key_points(percentage=0.1, resolution_control=0.01, show_result=True)

        f = mlab.gcf()  # this two line for mlab.screenshot to work
        f.scene._lift()
        img = mlab.screenshot()
        mlab.savefig(filename=str(i) + 'key_points_with_resolution_ctrl.png')
        mlab.close()


def segmentation_pcs_plot(pcs_path='', colorset=None):
    if colorset is None:
        colorset = [(226,50,226),(202,44,66),(111,41,66),(43,173,80),(51,200,200),(255,1,128),(23,48,217),(24,121,73)]
    f_list = [pcs_path + '/' + i for i in os.listdir(pcs_path) if os.path.splitext(i)[1] == '.ply']
    mfig = mlab.figure(bgcolor=(1, 1, 1))
    for j,i in enumerate(f_list):
        if j <=7:
            pc = PointCloud(i)

            mlab.points3d(pc.position[:, 0], pc.position[:, 1], pc.position[:, 2],
                          pc.position[:, 2] * 10 ** -9 + 1,
                          color=tuple((np.asarray(colorset[j],dtype=np.float)/255).tolist()),
                          scale_factor=3, figure=mfig)

    mlab.show()


def projection_plot(pcpath=''):
    f_list = [pcpath + '/' + i for i in os.listdir(pcpath) if os.path.splitext(i)[1] == '.ply']
    fig = plt.figure(figsize=(38, 20), dpi=600, facecolor='w')
    for i,j in enumerate(f_list):
        pc = PointCloud(j)
        pc.down_sample(number_of_downsample=10000)
        size = 2.5
        if i ==7:
            size  = 1
        try:
            mfig = pc.half_by_plane(n=1024, grid_resolution=(200, 200), show_result=size)
        except:
            try:
                mfig = pc.half_by_plane(n=1024, grid_resolution=(250, 250), show_result=size)
            except:
                try:
                    mfig = pc.half_by_plane(n=1024, grid_resolution=(300, 300), show_result=size)
                except:
                    mfig = pc.half_by_plane(n=1024, grid_resolution=(650, 650), show_result=size)


        f = mlab.gcf()  # this two line for mlab.screenshot to work
        f.scene._lift()
        mlab.savefig(str(i)+'.png')
        img = mlab.screenshot(figure=mfig)
        mlab.close()
        ax = fig.add_subplot(2, 4, i+1)
        ax.imshow(img)
        ax.set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()



if __name__ == "__main__":

    # print('the type of X_tsne is {}:, the shape is {}'.format(type(X), X.shape))
    # plot_embedding_3d(X, Y)

    # base_path = '/media/sjtu/software/ASY/pointcloud/lab scanned workpiece'
    # f_list = [base_path + '/' + i for i in os.listdir(base_path) if os.path.splitext(i)[1] == '.ply']
    # noise_outliers(f_list)
    # x = np.linspace(0, 1, 16)
    # y = np.linspace(0, 1, 16)
    # z = np.linspace(0, 1, 16)
    # xi, yi, zi = np.meshgrid(x, y, z)
    # points = np.concatenate([np.reshape(xi, [-1, 1]), np.reshape(yi, [-1, 1]), np.reshape(zi, [-1,1])], axis=1)
    # np.random.shuffle(points)
    # points = points[0:1024, :]
    # first_ly = np.load('first_ly.npy')
    # print(first_ly)
    # vis_first_layer(points, np.squeeze(first_ly, axis=0))
    # x = np.load('classification_output4tsne.npy')
    # y = np.load('tsne_label.npy')
    # z = np.load('point_clouds.npy')
    # plot_embedding_3d(x, y, point_clouds=z)
    #
    # x = np.arange(1000)*10
    # for i in range(8):
    #     n, bins, patches = plt.hist(x, 50, density=True, color=plt.cm.Set1((i+1) / 10.), alpha=0.75)
    #     plt.grid(False)
    #     # plt.show()
    #     plt.savefig(str(i)+'.png')
    #     plt.close()
    # pass
    # print(plt.cm.Set1((2 + 1) / 10.)[:3])
    # pass
    # pc = h5py.File('aishuo.h5', 'r')
    # pc = pc['data'][:][0]
    # layer = np.load('data64_1.npy')
    # print('layer shape:', np.shape(layer))
    # layer = np.reshape(layer, [1024, -1])
    # print(pc.shape, layer.shape)
    # vis_first_layer(pc, layer, vis_rate=1/10)
    # feature_mean_deviation('/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/8object0.02noise/lab1')
    # base_path = '/media/sjtu/software/ASY/pointcloud/lab scanned workpiece'
    # f_list = [base_path + '/' + i for i in os.listdir(base_path) if os.path.splitext(i)[1] == '.ply']
    # key_points_plot(f_list)
    # saliancey2range()
    # for j,i in enumer
    # ate(f_list):
    #     if j <4:
    #         pc = PointCloud(i)
    #         pc.down_sample(number_of_downsample=20000)
    #         fig = pc.show(not_show=True, scale=0.4)
    #         f = mlab.gcf()  # this two line for mlab.screenshot to work
    #         f.scene._lift()
    #         mlab.savefig(filename=str(j) + '_2.png')
    #         mlab.close()
    # segmentation_pcs_plot(pcs_path='/media/sjtu/software/ASY/pointcloud/三维扫描7.8/24')

    projection_plot(pcpath='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/8object')