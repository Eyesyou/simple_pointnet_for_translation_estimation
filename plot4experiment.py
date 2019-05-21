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



def saliancey2range():
    for j, i in enumerate(f_list):
        print(' point cloud is', i)
        pc = PointCloud(i)
        pc.down_sample(number_of_downsample=2048)
        for k in range(4):
            if k == 0:
                k = -0.5
            fig = pc.compute_key_points(percentage=0.1, show_result=False, resolution_control=0.005, rate=0.05 * k + 0.05,
                                        use_deficiency=True, show_saliency=True)
            f = mlab.gcf()  # this two line for mlab.screenshot to work
            f.scene._lift()
            img = mlab.screenshot()
            mlab.savefig(filename=str(j) + str(k) + '.png')
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


def plot_embedding_3d(X, Y, title=None):

    tsne = manifold.TSNE(n_components=3, perplexity=30, early_exaggeration=4.0, learning_rate=1000, init='pca',
                         random_state=0, n_iter=1000, verbose=0, method='barnes_hut', angle=0.5)
    X_tsne = tsne.fit_transform(X)

    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X_tsne, axis=0), np.max(X_tsne, axis=0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)

    #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i in range(X_tsne.shape[0]):
        ax.text(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2], str(Y[i]),
                 color=tuple(np.random.random((3,)).tolist()),  # plt.cm.Set1(y[i] / 10.)
                 fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        plt.title(title)
    plt.show()


def noise_outliers(pointclous):
    fig = plt.figure(figsize=(19, 10))
    for j, i in enumerate(pointclous):
        pc = PointCloud(i)
        pc.down_sample(number_of_downsample=2048)

        for k in range(4):
            if k ==3:
                k = 4
            pc.add_outlier(factor=k*0.02)
            pc.add_outlier(factor=k*0.02)
            m_fig = mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))

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

if __name__ == "__main__":
    # rneighbor2range()

    # digits = datasets.load_digits(n_class=5)
    # X = digits.data
    # X = np.random.random((1024, 3096))
    # Y = np.random.randint(low=0, high=5, size=(1024, ))
    # y = digits.target
    # print('X shape', X.shape)
    # print('y shape', y.shape)
    # n_img_per_row = 20
    # img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
    # for i in range(n_img_per_row):
    #     ix = 10 * i + 1
    #     for j in range(n_img_per_row):
    #         iy = 10 * j + 1
    #         img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))
    # plt.imshow(img, cmap=plt.cm.binary)
    # plt.title('A selection from the 64-dimensional digits dataset')

    # LLE,Isomap,LTSA需要设置n_neighbors这个参数
    # n_neighbors = 30

    # print('the type of X_tsne is {}:, the shape is {}'.format(type(X), X.shape))
    # plot_embedding_3d(X, Y)

    base_path = '/media/sjtu/software/ASY/pointcloud/lab scanned workpiece'
    f_list = [base_path + '/' + i for i in os.listdir(base_path) if os.path.splitext(i)[1] == '.ply']
    noise_outliers(f_list)
    pass