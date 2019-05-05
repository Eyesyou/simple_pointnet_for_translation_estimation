import numpy as np
import h5py
import time
import tensorflow as tf
from mayavi import mlab
from scipy.spatial import distance
import math
from plyfile import PlyData, PlyElement
from scipy import spatial
from show_pc import PointCloud
import matplotlib.pyplot as plt
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



# readh5 = h5py.File('/media/sjtu/software/ASY/pointcloud/train_set4noiseout/project_data.h5', 'r')  # file path

# pc_tile = readh5['train_set'][:]  # 20000 * 1024 * 3
# pc_tile *= 100
# pc = PointCloud(pc_tile[1, :, :])


def test_uniform_rotation(n=50000):
    p = np.asarray([[1, 0, 0], [1, 0, 0]])  # 2, 3

    p = np.expand_dims(p, axis=0)  # 1, 2, 3
    p = np.tile(p, [n, 1, 1])   # n, 2, 3

    rand_pos = -1+2*np.random.random([n, 4])  # n, 4
    ind_sign = np.where(rand_pos[:, 0] > 0)[0]

    rand_pos[ind_sign, :] *= -1

    rand_pos = np.concatenate([rand_pos, np.zeros([n, 3])], axis=1)
    norm = np.linalg.norm(rand_pos, axis=1, keepdims=True)
    rand_pos = rand_pos/norm

    # ten = tf.convert_to_tensor(rand_pos)
    # ten = tf_quat_pos_2_homo(ten)
    #
    # with tf.Session() as sess:
    #     tfresult = sess.run([ten, ])
    #     print('tfresult: ', tfresult)

    rand_pos = np_quat_pos_2_homo(rand_pos)

    inv = np.linalg.inv(rand_pos)

    p = apply_np_homo(p, rand_pos)
    print(p.shape)
    p = p.reshape([-1, 3])
    print(p.shape)
    print(np.linalg.norm(p, axis=1))
    mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    mlab.points3d(p[:, 0], p[:, 1], p[:, 2], p[:, 0]*10**-9+1, colormap='Spectral', scale_factor=0.02)
    # mlab.points3d(0, 0, 0, color=(1, 0, 0), scale_factor=0.2)
    mlab.show()


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
    pad = tf.concat([tf.constant(0.0, shape=[batch, 1, 3], dtype=tf.float64), tf.ones([batch, 1, 1], dtype=tf.float64)], axis=2) #Bx1x4
    batch_out = tf.concat([batch_out, pad], axis=1)  #Bx4x4
    return batch_out


def apply_np_homo(batch_point_cloud, homo='random'):
    """

    :param batch_point_cloud: B x n x 3 , np.array
    :param homo: if random, apply random, or B*4*4 homo matrix
    :return: B x n x 3 np.array
    """

    batch = batch_point_cloud.shape[0]
    num = batch_point_cloud.shape[1]

    #batch_out = tf.Variable(tf.zeros(pc_batch_input.shape), trainable=False, dtype=tf.float32)
    #batch_out = batch_out.assign(pc_batch_input)
    batchout = np.concatenate([batch_point_cloud, np.ones((batch, num, 1))], axis=2)
    batchout = np.transpose(batchout, (0, 2, 1))

    if homo =='random':
        quat = 20 * np.random.random((batch, 4)) - 10
        quat_pos = np.concatenate([nor4vec(quat), 20 * np.random.random((batch, 4)) - 10], axis=1 )     #B x 7
        homo = np_quat_pos_2_homo(quat_pos)

    batchout = np.matmul(homo, batchout) #Bx4x4 * B x 4 x n
    batchout = np.divide(batchout, batchout[:, np.newaxis, 3, :])
    batchout = batchout[:, :3, :]
    batchout = np.transpose(batchout, (0, 2, 1))

    return batchout


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


def tf_cov_mat(tensor):
    """

    :param tensor: n x k tensor
    :return:
    """
    n = tensor.get_shape().as_list()[0]

    a = tensor - 1/n * tf.matmul(tf.ones(shape=(n, n), dtype=tf.float64), tensor)

    print('type', type(tf.constant(1/n, dtype=tf.float64)))
    return tf.matmul(tf.transpose(a), a)*tf.constant(1/n, dtype=tf.float64)


def tf_knn(tensor, k=3):
    """

    :param tensor: B x n x m array
    :param k:  k in k neareast neighbor
    :return: n x k , for each row, return the k small elemen
    :return:
    """

    # l2 distance
    if len(tensor.get_shape().as_list()) == 2:
        tensor = tf.expand_dims(tensor, axis=0)
    r = tf.reduce_sum(tensor*tensor, axis=-1)  # B x n x 1
    D = r - 2*tf.matmul(tensor, tf.transpose(tensor)) + tf.transpose(r)  # B x n x n
    # ind = tf.argsort(D, axis=-1)                                                                     # B x n x n

    distance = tf.reduce_sum(tf.math.sqaure(tensor), axis=-1)

    _, top_k_indices = tf.math.top_k(tf.negative(D), k=k)


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


def nor4vec(vector):
    """
    :param vector: B x 4
    :return: B x 4
    """
    return vector/np.linalg.norm(vector, axis=1)[:,np.newaxis]


def point2line_dist(point, line_origin, line_vector):
    """

    :param point: 1x3
    :param line_origin: 1x3
    :param line_vector:  1x3
    :return: scale distance
    """
    S = np.linalg.norm(np.cross((point-line_origin), line_vector))
    return S/np.linalg.norm(line_vector)

@timeit
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
        print('current:', current)
        max_nb_nei_pts = np.max(np.asarray([max_nb_nei_pts, current]), axis=0)
        print('max_nb:', max_nb_nei_pts)
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

    print('idx2: ', idx2, 'shape:', idx2.shape)
    np_arr2.astype(int)
    print('np_arr2: ', np_arr2, 'shape:', np_arr2.shape)
    pts_r_cov = get_pts_cov(point_cloud, np_arr2)  # np_arr2 is b x n  b x n x 3 x 3

    eigen_val, _ = np.linalg.eigh(pts_r_cov)  # b x n x 3 orderd

    idx = np.argpartition(eigen_val[:, :, 0], nb_key_pts, axis=1)
    print('idx:', idx)   # b x n
    # print(eigen_val[idx])
    key_idx = idx[:, 0:nb_key_pts]

    print('idx:', key_idx)   # b x nb_key

    # print('key points coordinates:', point_cloud[idx, :], 'shape:', point_cloud[idx, :].shape)

    # b_dix = np.indices((batchsize, nb_key_pts))[1]  # b x nb_key
    # print('b_dix: ', b_dix, 'shape:', b_dix.shape)
    # batch_idx = np.concatenate([np.expand_dims(b_dix, axis=-1), np.expand_dims(idx, axis=-1)], axis=-1)  # b x nb_key x 2

    key_eig_val = np.empty((batchsize, nb_key_pts, 3))  # b x nb_keypoints x 3
    for i in range(batchsize):
        key_eig_val[i, :, :] = eigen_val[i, key_idx[i, :], :]

    print('key_eig_val: ', key_eig_val, 'shape:', key_eig_val.shape)

    print('np_arr1: ', np_arr1, 'shape:', np_arr1.shape)
    print('np_arr3: ', np_arr3, 'shape:', np_arr3.shape)
    np_key_arr1 = np.empty((batchsize, nb_key_pts, np_arr1.shape[2]))                                     # np_arr1: b x n x nei1  to  b x  nb_key x  nei1
    np_key_arr3 = np.empty((batchsize, nb_key_pts, np_arr3.shape[2]))
    np_key_arr1[:] = np.nan
    np_key_arr3[:] = np.nan
    for i in range(batchsize):
        np_key_arr1[i, :, :] = np_arr1[i, key_idx[i, :], :]
        np_key_arr3[i, :, :] = np_arr3[i, key_idx[i, :], :]

    print('np_key_arr1: ', np_key_arr1, 'shape:', np_key_arr1.shape)
    print('np_key_arr3: ', np_key_arr3, 'shape:', np_key_arr3.shape)

    key_pts_cov1 = get_pts_cov(point_cloud, np_key_arr1)  #    np_arr1: b x nb_key x nei1     b x nb_key x 3 x 3
    key_pts_cov3 = get_pts_cov(point_cloud, np_key_arr3)  #    np_arr3: b x nb_key x nei3     b x nb_key x 3 x 3

    key_eig_val2 = key_eig_val
    key_eig_val1, _ = np.linalg.eigh(key_pts_cov1)  # b x nb_key_pts x 3 orderd
    key_eig_val3, _ = np.linalg.eigh(key_pts_cov3)  # b x nb_key_pts x 3 orderd

    concat = np.concatenate((key_eig_val1, key_eig_val2, key_eig_val3), axis=-1)  # b x nb_key_pts x 9

    return concat


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

            this_pt_nei_idx = pts_r_neirhbor_idx[i, j, :][pts_r_neirhbor_idx[i, j, :] >= 0]
            this_pt_nei_idx = this_pt_nei_idx.astype(np.int32)
            #print('this_pt_nei_idx:', this_pt_nei_idx)
            neighbor_pts = pc[i, this_pt_nei_idx, :]  # 1 x nb x 3
            #print('neighbor_pts shape:', neighbor_pts.shape)
            if neighbor_pts.size == 0:
                cov[i, j, :, :] = np.eye(3)
            else:
                cov[i, j, :, :] = np.cov(neighbor_pts, rowvar=False)

    return np.float32(cov)




if __name__ == "__main__":

    # A = PointCloud(A)
    # A.generate_k_neighbor(show_result=True)
    # plydata = PlyData.read('./pointcloud/lab work peace by reconstruction.ply')
    #
    # vertex = np.asarray([list(subtuple) for subtuple in plydata['vertex'][:]])
    # vertex = vertex[:, 0:3]
    # pc = PointCloud(vertex)
    # pc.down_sample()
    # pc.generate_r_neighbor(show_result=True)

    # print('vertext[:].tolist():\n', plydata['vertex'][:])
    # A = tf.constant(A, dtype=tf.float64)
    # A = tf_cov_mat(A)
    # with tf.Session() as sess:
    #     b = sess.run(A)
    #     print(b)
    a = np.zeros((4, 4))
    b = np.array([2, 3])
    a[b[0]][b[1]] = 1
    print(a)



    pass

    # test_uniform_rotation()