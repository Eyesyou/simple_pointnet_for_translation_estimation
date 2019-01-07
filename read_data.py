import numpy as np
import h5py
from show_pc import *
import random
from mayavi import mlab
from matplotlib import pyplot as plt
from show_pc import PointCloud
from plyfile import PlyData, PlyElement

def save_data(directory_lists=None, save_path='', n=5000, base_path=''):

    pc_tile = np.empty(shape=(4*n, 1024, 3))

    for i in range(n):
        if i % 10 == 0:
            print('reading number', i + 1, 'th lab1 point clouds')
        try:
            pc_tile[i, :, :] = np.expand_dims(
                np.loadtxt(base_path+'/lab1/lab1_project'+str(i)+'.txt'), axis=0)
        except:
            print(i, 'th point cloud shape is:',
                  np.loadtxt(base_path+'/lab1/lab1_project'+str(i)+'.txt').shape)

    for i, j in enumerate(range(n, 2*n)):
        if i % 10 == 0:
            print('reading number', i + 1, 'th lab2 point clouds')
        try:
            pc_tile[j, :, :] = np.expand_dims(np.loadtxt(base_path+'/lab2/lab_project'+str(i)+'.txt'),
                                              axis=0)
        except:
            print(i, 'th point cloud shape is:',
                  np.loadtxt(base_path+'/lab2/lab_project'+str(i)+'.txt').shape)

    for i, j in enumerate(range(2*n, 3*n)):
        if i % 10 == 0:
            print('reading number', i + 1, 'th lab3 point clouds')
        try:
            pc_tile[j, :, :] = np.expand_dims(
                np.loadtxt(base_path+'/lab3/lab_project'+str(i)+'.txt'), axis=0)
        except:
            print(i, 'th point cloud shape is:',
                  np.loadtxt(base_path+'/lab3/lab_project'+str(i)+'.txt').shape)

    for i, j in enumerate(range(3*n, 4*n)):
        if i % 10 == 0:
            print('reading number', i + 1, 'th lab4 point clouds')
        try:
            pc_tile[j, :, :] = np.expand_dims(np.loadtxt(
                base_path+'/lab4/lab4_project'+str(i)+'.txt'), axis=0)
        except:
            print(i, 'th point cloud shape is:', np.loadtxt(
                base_path+'/lab4/lab4_project'+str(i)+'.txt').shape)

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
    :param showinone:
    :return:
    """
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


def augment_data(base_path='', pc_path=''):
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

    # pc = np.loadtxt('Dragon_Fusion_bronze_statue.txt')   # n x 3

    plydata = PlyData.read(pc_path)
    vertex = np.asarray([list(subtuple) for subtuple in plydata['vertex'][:]])
    pc = vertex[:, 0:3]
    np.random.shuffle(pc)  # will only shuffle the first axis

    if pc.shape[0] > 10000:
        pc = pc[0:10000, :]
    pc = PointCloud(pc)
    pc.add_noise(factor=0.05)
    pc.add_outlier(factor=0.05)
    for i in range(1500,5000):
        if i % 10 == 0:
            print('saving number', i+1, 'th lab4_project point clouds')
        pc.half_by_plane()
        np.savetxt(base_path+'/lab4_project'+str(i)+'.txt', pc.visible, delimiter=' ')

    # pc = np.loadtxt('blade.txt')   # n x 3
    # np.random.shuffle(pc)  # only shuffle the first axis
    # pc = pc[0:10000, :]
    # pc = PointCloud(pc)
    # pc.add_noise(factor=0.05)
    # pc.add_outlier(factor=0.05)
    # for i in range(5000):
    #     if i % 10 == 0:
    #         print('saving number', i+1, 'th blade point clouds')
    #     pc.half_by_plane()
    #     np.savetxt(base_path+'/blade/blade_project'+str(i)+'.txt', pc.visible, delimiter=' ')
    #
    # pc = np.loadtxt('fullbodyanya1.txt')   # n x 3
    # np.random.shuffle(pc)  # only shuffle the first axis
    # pc = pc[0:10000, :]
    # pc = PointCloud(pc)
    # pc.add_noise(factor=0.05)
    # pc.add_outlier(factor=0.05)
    # for i in range(1, 5000):
    #     if i % 10 == 0:
    #         print('saving number', i+1, 'th fullbodyanya1 point clouds')
    #     try:
    #         pc.half_by_plane()
    #         np.savetxt(base_path + '/fullbodyanya1/fullbodyanya1_project' + str(i) + '.txt', pc.visible, delimiter=' ')
    #     except ValueError:
    #         try:
    #             pc.half_by_plane()
    #             np.savetxt(base_path + '/fullbodyanya1/fullbodyanya1_project' + str(i) + '.txt', pc.visible, delimiter=' ')
    #         except ValueError:
    #             pc.half_by_plane()
    #             np.savetxt(base_path + '/fullbodyanya1/fullbodyanya1_project' + str(i) + '.txt', pc.visible,
    #                        delimiter=' ')


if __name__ =="__main__":
    save_data(save_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/project_data.h5',
              base_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece')

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

    # augment_data(base_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/lab4',
    #              pc_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/lab4/final.ply')
    # test_data(h5_path='/media/sjtu/software/ASY/pointcloud/train_set2/project_data.h5', rand_trans=True, showinone=True)
    # pc = np.loadtxt('/media/sjtu/software/ASY/pointcloud/lab_workpice.txt')
    # pc = PointCloud(pc)
    # pc.show()

