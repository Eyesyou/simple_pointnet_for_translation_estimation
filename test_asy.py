import numpy as np
import h5py
import tensorflow as tf
from mayavi import mlab
from scipy.spatial import distance
import math
from plyfile import PlyData, PlyElement


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
        self.point_kneighbors = None  # n x k , k is the index of the neighbor points
        self.point_rneighbors = None  # n x (0 to nb_points), the index of the neighbor points,
        # number may differ according to different points
        print(self.nb_points, ' points', 'range:', self.range)

    def half_by_plane(self, plane=None, n=1024, grid_resolution=(256, 256)):
        """
        :param plane:  the plane you want to project the point cloud into, and generate the image-like grid,
        define the normal of the plane is to the direction of point cloud center
        :param n:
        :param grid_resolution:
        :return:
        """
        if plane is None:
            # generate a random plane whose distance to the center bigger than self.range
            # d = abs(Ax+By+Cz+D)/sqrt(A**2+B**2+C**2)
            plane_normal = -0.5 + np.random.random(size=[3, ])  # random A B C for Ax+By+Cz+D=0
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
        x0 = self.center[0] - A * t  # project point in the plane
        y0 = self.center[1] - B * t
        z0 = self.center[2] - C * t
        self.plane_origin = [x0, y0, z0]
        if (self.center[0] - x0) / A < 0:
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

    def show(self, not_show=False, scale=0.005):
        mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
        fig = mlab.points3d(self.position[:, 0], self.position[:, 1], self.position[:, 2],
                            self.position[:, 2] * 10**-9 + self.range * scale,
                            colormap='Spectral', scale_factor=1)
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
                mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))

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

    def generate_r_neighbor(self, r=None, show_result=False):
        if r is None:
            r = self.range / 40
        else:
            assert 0 < r < self.range

        p_distance = distance.cdist(self.position, self.position)
        idx = np.where((p_distance < r) & (p_distance != 0))  # choose axis 0 or axis 1

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
            for i in range(5):
                j = np.random.choice(self.nb_points, 1)  # random point index
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
        print(self.nb_points, ' points', 'range:', self.range)

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


if __name__ == "__main__":
    A = np.random.random([1, 4, 3])
    B = np.random.random([1, 4, 3])  # n x (0 to nb_points), the index of the neighbor points,
    B[0, 3, :] = np.nan
    C = np.concatenate((A, B), axis=0)  # 2 x 4 x 3
    print("C:\n", C)
    print("diff neighbors compute_covariance_mat:\n", compute_covariance_mat(C))
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
    pass

    # test_uniform_rotation()