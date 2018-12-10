import numpy as np
import h5py
from show_pc import *
import random
from mayavi import mlab
from matplotlib import pyplot as plt


def save_data(directory_lists=None, save_path='',n=5000, base_path=''):

    pc_tile = np.empty(shape=(4*n, 1024, 3))

    for i in range(n):
        if i % 10 == 0:
            print('reading number', i + 1, 'th carburator point clouds')
        try:
            pc_tile[i, :, :] = np.expand_dims(
                np.loadtxt(base_path+'/carburator/carburator_project'+str(i)+'.txt'), axis=0)
        except:
            print(i, 'th point cloud shape is:',
                  np.loadtxt(base_path+'/carburator/carburator_project'+str(i)+'.txt').shape)

    for i, j in enumerate(range(n, 2*n)):
        if i % 10 == 0:
            print('reading number', i + 1, 'th blade point clouds')
        try:
            pc_tile[j, :, :] = np.expand_dims(np.loadtxt(base_path+'/blade/blade_project'+str(i)+'.txt'),
                                              axis=0)
        except:
            print(i, 'th point cloud shape is:',
                  np.loadtxt(base_path+'/blade/blade_project'+str(i)+'.txt').shape)

    for i, j in enumerate(range(2*n, 3*n)):
        if i % 10 == 0:
            print('reading number', i + 1, 'th arm_monster point clouds')
        try:
            pc_tile[j, :, :] = np.expand_dims(
                np.loadtxt(base_path+'/arm_monster/arm_monster_project'+str(i)+'.txt'), axis=0)
        except:
            print(i, 'th point cloud shape is:',
                  np.loadtxt(base_path+'/arm_monster/arm_monster_project'+str(i)+'.txt').shape)

    for i, j in enumerate(range(3*n, 4*n)):
        if i % 10 == 0:
            print('reading number', i + 1, 'th Dragon_Fusion_bronze_statue point clouds')
        try:
            pc_tile[j, :, :] = np.expand_dims(np.loadtxt(
                base_path+'/Dragon_Fusion_bronze_statue/Dragon_Fusion_bronze_statue_project'+str(i)+'.txt'), axis=0)
        except:
            print(i, 'th point cloud shape is:', np.loadtxt(
                base_path+'/Dragon_Fusion_bronze_statue/Dragon_Fusion_bronze_statue_project'+str(i)+'.txt').shape)

    pc_label = np.concatenate(
        [np.zeros(shape=(n,)), np.ones(shape=(n,)), 2 * np.ones(shape=(n,)), 3 * np.ones(shape=(n,))], axis=0)

    train_set_shape = (4*n, 1024, 3)
    train_label_shape = (4*n, )

    hdf5_file = h5py.File(save_path, mode='a')
    hdf5_file.create_dataset('train_set', train_set_shape, np.float32)  # be careful about the dtype
    hdf5_file.create_dataset('train_labels', train_label_shape, np.uint8)
    hdf5_file["train_set"][...] = pc_tile
    hdf5_file["train_labels"][...] = pc_label
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


class PointCloud:
    def __init__(self, one_pointcloud):
        assert isinstance(one_pointcloud, np.ndarray)
        one_pointcloud = np.squeeze(one_pointcloud)
        assert one_pointcloud.shape[1] == 3
        self.min_limit = np.amin(one_pointcloud, axis=0)  # 1x3
        self.max_limit = np.amax(one_pointcloud, axis=0)  # 1x3
        self.range = self.max_limit-self.min_limit
        self.range = np.sqrt(self.range[0]**2+self.range[1]**2+self.range[2]**2)  # diagonal distance
        self.position = one_pointcloud     # nx3
        self.center = np.mean(self.position, axis=0)  # 1x3
        self.nb_points = np.shape(self.position)[0]
        self.visible = self.position
        self.plane = None
        self.plane_origin = None
        self.plane_project_points = None
        self.root = None
        self.depth = 0
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
            plane_normal = -0.5+np.random.random(size=[3, ])  # random A B C for Ax+By+Cz+D=0
            A = plane_normal[0]
            B = plane_normal[1]
            C = plane_normal[2]
            D = -(A*self.center[0]+B*self.center[1]+C*self.center[2])+(np.random.binomial(1, 0.5)*2-1) * \
                self.range*np.sqrt(A**2+B**2+C**2)

        else:
            A = plane[0]
            B = plane[1]
            C = plane[2]
            D = plane[3]

        # compute the project point of center in the grid plane:
        t = (A*self.center[0]+B*self.center[1]+C*self.center[2]+D)/(A**2+B**2+C**2)
        x0 = self.center[0] - A * t  # project point in the plane
        y0 = self.center[1] - B * t
        z0 = self.center[2] - C * t
        self.plane_origin = [x0, y0, z0]
        if (self.center[0]-x0)/A < 0:
            A = -A
            B = -B
            C = -C
            D = -D
        self.plane = [A, B, C, D]
        try:
            assert math.isclose((self.center[0]-x0)/A, (self.center[1] - y0)/B) and \
                   math.isclose((self.center[1]-y0)/B, (self.center[2] - z0)/C) and (self.center[0]-x0)/A > 0
        except AssertionError:
            print('AssertionError', (self.center[0]-x0)/A, (self.center[1] - y0)/B, (self.center[2] - z0)/C, A, B, C, D)
        x1 = x0                             # Parallelogram points of the grid,define x1,y1,z1 by plane function and
        a = 1 + B ** 2 / C ** 2             # range distance limitation
        b = 2*B/C*(z0+(D+A*x1)/C)-2*y0
        c = y0**2-self.range**2/4+(x1-x0)**2+(z0+(D+A*x1)/C)**2
        y1 = np.roots([a, b, c])     # Unary two degree equation return two root
        if np.isreal(y1[0]):
            y1 = y1[0]
        else:
            print('not real number')
        z1 = -(D+A*x1+B*y1)/C
        # the y direction of the plane, this is a vector
        y_nomal = np.cross([self.center[0]-x0, self.center[1]-y0, self.center[2]-z0], [x1-x0, y1-y0, z1-z0])

        # the minimal distance for every grid, the second index store the point label

        min_dist = 10*self.range*np.ones(shape=[grid_resolution[0], grid_resolution[1], 2])
        point_label = np.zeros(shape=(self.nb_points, ))
        for i in range(self.nb_points):

            t_ = (A * self.position[i, 0] + B * self.position[i, 1] + C * self.position[i, 2] + D) \
                 / (A ** 2 + B ** 2 + C ** 2)
            project_point = np.asarray([self.position[i, 0] - A * t_, self.position[i, 1] - B * t_,
                                        self.position[i, 2] - C * t_])

            project_y = point2line_dist(project_point, np.asarray([x0, y0, z0]),
                                        np.asarray([x1-x0, y1-y0, z1-z0]))
            project_x = np.sqrt(np.sum(np.square(project_point-np.asarray([x0, y0, z0])))-project_y**2)

            # print('project x', project_x, 'project y', project_y)
            if (project_point[0]-x0)*(x1-x0)+(project_point[1]-y0)*(y1-y0)+(project_point[2]-z0)*(z1-z0) >= 0:
                # decide if it is first or fourth quadrant
                if np.dot(y_nomal, project_point-np.asarray([x0, y0, z0])) < 0:
                    # fourth quadrant
                    project_y = -project_y

            else:
                project_x = - project_x
                if np.dot(y_nomal, project_point-np.asarray([x0, y0, z0])) < 0:
                    # third quadrant
                    project_y = -project_y

            pixel_width = self.range * 2 / grid_resolution[0]
            pixel_height = self.range * 2 / grid_resolution[1]
            distance = point2plane_dist(self.position[i, :], [A, B, C, D])
            index_x = int(grid_resolution[0]/2 + np.floor(project_x/pixel_width))
            index_y = int(grid_resolution[1]/2 + np.floor(project_y/pixel_height))
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
                print('AssertionError:', np.floor(project_x/pixel_width), pixel_width)

        if n is not None:
            # sample the visible points to given number of points
            medium = self.position[point_label == 1]
            try:
                assert medium.shape[0] >= n  # sampled points have to be bigger than n
            except AssertionError:
                print('sampled points number is:', medium.shape[0])
                raise ValueError('value error')
            np.random.shuffle(medium)   # only shuffle the first axis
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
        fig = mlab.points3d(self.position[:, 0], self.position[:, 1], self.position[:, 2], self.position[:, 2]*0.0001+ self.range*scale,
                            colormap='Spectral', scale_factor=1)
        if not not_show:
            mlab.show()
        else:
            return fig

    def add_noise(self, factor=1/100):
        """
        jitter noise for every points in the point cloud
        :param factor:
        :return:
        """
        noise = np.random.random([self.nb_points, 3]) * factor * self.range
        self.position += noise

    def add_outlier(self, factor=1/100):
        """
        randomly delete points and make it to be the outlier
        :param factor:
        :return:
        """

        inds = np.random.choice(np.arange(self.nb_points), size=int(factor*self.nb_points))
        self.position[inds] = self.center + -self.range/6 + self.range/3 * np.random.random(size=(len(inds), 3))

    def normalize(self):
        self.position -= self.center
        self.position /= self.range
        self.center = np.mean(self.position, axis=0)
        self.min_limit = np.amin(self.position, axis=0)
        self.max_limit = np.amax(self.position, axis=0)
        self.range = self.max_limit-self.min_limit
        self.range = np.sqrt(self.range[0]**2+self.range[1]**2+self.range[2]**2)
        print('center: ', self.center, 'range:', self.range)

    def octree(self):
        def width_first_traversal(position, size, data):
            root = OctNode(position, size, data)

            min = root.position + [-root.size / 2, -root.size / 2, -root.size / 2]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            lbd = root.data[np.all(media, axis=1), :]
            if lbd.shape[0] > 1:
                root.dbl = width_first_traversal(root.position+[-1/4*root.size, -1/4*root.size, -1/4*root.size],
                                                 root.size*1/2, data=lbd)

            min = root.position + [0, -root.size / 2, -root.size / 2]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            rbd = root.data[np.all(media, axis=1), :]
            if rbd.shape[0] > 1:
                root.dbr = width_first_traversal(root.position+[1/4*root.size, -1/4*root.size, -1/4*root.size],
                                                 root.size*1/2, data=rbd)

            min = root.position + [-root.size / 2, 0, -root.size / 2]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            lfd = root.data[np.all(media, axis=1), :]
            if lfd.shape[0] > 1:
                root.dfl = width_first_traversal(root.position+[-1/4*root.size, 1/4*root.size, -1/4*root.size],
                                                 root.size*1/2, data=lfd)

            min = root.position + [0, 0, -root.size / 2]  # minx miny minz
            max = min + root.size / 2  # maxx maxy maxz
            media = np.logical_and(min < root.data, root.data < max)
            rfd = root.data[np.all(media, axis=1), :]
            if rfd.shape[0] > 1:
                root.dfr = width_first_traversal(root.position+[1/4*root.size, 1/4*root.size, -1/4*root.size],
                                                 root.size*1/2, data=rfd)

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

        self.root = width_first_traversal(self.center, size=max(pc.max_limit-pc.min_limit), data=self.position)

        def maxdepth(node):
            if not any(node.children):
                return 0
            else:
                return max([maxdepth(babe)+1 for babe in node.children if babe is not None])
        self.depth = maxdepth(self.root)

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
    h5file = read_data(h5_path)
    trainset = h5file['train_set'][...]
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


def augment_data(base_path='/media/sjtu/software/ASY/pointcloud/train_set3noise'):
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

    pc = np.loadtxt('Dragon_Fusion_bronze_statue.txt')   # n x 3
    np.random.shuffle(pc)  # only shuffle the first axis
    pc = pc[0:10000, :]
    pc = PointCloud(pc)
    pc.add_noise(factor=0.05)
    pc.add_outlier(factor=0.05)
    for i in range(5000):
        if i % 10 == 0:
            print('saving number', i+1, 'th Dragon_Fusion_bronze_statue point clouds')
        pc.half_by_plane()
        np.savetxt(base_path+'/Dragon_Fusion_bronze_statue/Dragon_Fusion_bronze_statue_project'+str(i)+'.txt', pc.visible, delimiter=' ')


    pc = np.loadtxt('blade.txt')   # n x 3
    np.random.shuffle(pc)  # only shuffle the first axis
    pc = pc[0:10000, :]
    pc = PointCloud(pc)
    pc.add_noise(factor=0.05)
    pc.add_outlier(factor=0.05)
    for i in range(5000):
        if i % 10 == 0:
            print('saving number', i+1, 'th blade point clouds')
        pc.half_by_plane()
        np.savetxt(base_path+'/blade/blade_project'+str(i)+'.txt', pc.visible, delimiter=' ')

    pc = np.loadtxt('fullbodyanya1.txt')   # n x 3
    np.random.shuffle(pc)  # only shuffle the first axis
    pc = pc[0:10000, :]
    pc = PointCloud(pc)
    pc.add_noise(factor=0.05)
    pc.add_outlier(factor=0.05)
    for i in range(1, 5000):
        if i % 10 == 0:
            print('saving number', i+1, 'th fullbodyanya1 point clouds')
        try:
            pc.half_by_plane()
            np.savetxt(base_path + '/fullbodyanya1/fullbodyanya1_project' + str(i) + '.txt', pc.visible, delimiter=' ')
        except ValueError:
            try:
                pc.half_by_plane()
                np.savetxt(base_path + '/fullbodyanya1/fullbodyanya1_project' + str(i) + '.txt', pc.visible, delimiter=' ')
            except ValueError:
                pc.half_by_plane()
                np.savetxt(base_path + '/fullbodyanya1/fullbodyanya1_project' + str(i) + '.txt', pc.visible,
                           delimiter=' ')


if __name__ =="__main__":
    # save_data(save_path='/media/sjtu/software/ASY/pointcloud/train_set4noiseout/project_data.h5', base_path='/media/sjtu/software/ASY/pointcloud/train_set4noiseout')
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

    # augment_data(base_path='/media/sjtu/software/ASY/pointcloud/train_set4noiseout')
    # test_data(h5_path='/media/sjtu/software/ASY/pointcloud/train_set2/project_data.h5', rand_trans=True, showinone=True)
    pc = np.loadtxt('/media/sjtu/software/ASY/pointcloud/lab_workpice.txt')
    pc = PointCloud(pc)
    pc.show()

