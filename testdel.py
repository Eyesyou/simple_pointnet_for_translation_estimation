import tensorflow as tf
import numpy as np
import cv2
from show_pc import *
from read_data import *
from show_pc import PointCloud
# import pcl
import open3d as od

def compute_covariance_mat(numpy_arr):
    """

    :param numpy_arr:  b x n x k array, b is the key points number, n is the nb_neighbors,
     maybe nan because of different nb_neighbor, k is the dimension
    :return: b x k x k  covariance array
    """
    b = numpy_arr.shape[0]  # b
    n = numpy_arr.shape[1]  # n
    tmp = np.ones(shape=(b, n, n)) @ numpy_arr
    # tmp = np.tensordot(np.ones(shape=(b, n, n)), numpy_arr, axes=(2, [1, 2]))  # 2 x 4 x 4 dot 2 x 4 x 3
    a = numpy_arr - 1/n * tmp
    # todo: delete nan effect

    return np.transpose(a, axes=[0, 2, 1]) @ a * 1/n   #  b x k x n @ # b x n x k = b x k x k


def mycov(g, v):
    number = v.shape[0]
    dimension = v.shape[1]
    covmat = np.zeros((dimension, dimension))
    for i in range(number):
        covmat += g[i, :].T @ v[i, :]
    return covmat


if __name__ == '__main__':
    print(np.random.random([4, 3]))

