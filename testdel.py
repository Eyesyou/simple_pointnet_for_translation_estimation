import tensorflow as tf
import numpy as np
import cv2
from show_pc import *
from read_data import *
from show_pc import PointCloud
# import pcl
import open3d as od
# import jax.numpy as np
from jax import random

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

@timeit
def mat_multi(a):

    return np.matmul(a, a)


if __name__ == '__main__':
    # a = random.uniform(random.PRNGKey(1), [500, 500])
    a = np.random.random((5000, 5000))
    b = mat_multi(a)

    print(b)

