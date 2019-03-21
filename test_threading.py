import h5py
import threading
import numpy as np
from show_pc import PointCloud

def save_data(save_path='', base_path='', n=50, use_key_feature=True):
    """
    transform the txt point clouds into h5py dataset for simplicity.
    :param save_path:
    :param n:
    :param base_path:
    :param use_key_feature:
    :return:
    """

    # tf.enable_eager_execution()
    pc_tile = np.empty(shape=(4*n, 1024, 3))
    if use_key_feature:
        pc_key_feature = np.empty(shape=(4*n, int(1024*0.1), 9))  # key feature space, 102=1024*0.1,
        # 9 for multi-scale eigen-value
        #pc_pl = tf.placeholder(tf.float32, shape=(1, 1024, 3))

    def load_data(k):
        for i, j in enumerate(range(k*n, (k+1)*n)):

                if i % 10 == 0:
                    print('reading number', i + 1, 'th lab'+str(k+1)+' point clouds')

                if use_key_feature:
                    pc = np.loadtxt(base_path+'/lab'+str(k+1)+'/lab_project'+str(i)+'.txt')  # pc = tf.convert_to_tensor(pc, dtype=tf.float32)
                    pc = PointCloud(pc)
                    pc.normalize()

                    expand = np.expand_dims(pc.position, axis=0)
                    pc_tile[j, :, :] = expand
                    # print('*****************************************')
                    # print('reading point cloud cost time:{}'.format(t1 - t0))
                    pc_key_eig = get_local_eig_np(expand)   # 1 x nb_keypoints x 9

                    # print('*****************************************')
                    # print('get local cost time:{}'.format(t2 - t1))
                    #pc_key_feature[i, :, :] = np.squeeze(sess.run(pc_key_eig, feed_dict={pc_pl: pc}))
                    pc_key_feature[j, :, :] = np.squeeze(pc_key_eig)
                else:
                    pc_tile[j, :, :] = np.expand_dims(
                        np.loadtxt(base_path+'/lab'+str(k+1)+'/lab_project'+str(i)+'.txt'), axis=0)

                # print('-----------------------------------------')
                # print('one pc cost total:{}second'.format(te-ts))
                # print('----------------------------------------')
        print('current thread ending： ', threading.current_thread().name)

    thread_pool = []
    for k in range(4):  # four objects model
        thread_pool.append(threading.Thread(target=load_data, args=[k, ], daemon=True))
        thread_pool[k].start()

    for j in range(4):
        thread_pool[j].join()  # main thread will waiting for other thread before ending.

    pc_label = np.concatenate(
        [np.zeros(shape=(n,)), np.ones(shape=(n,)), 2 * np.ones(shape=(n,)), 3 * np.ones(shape=(n,))], axis=0)

    train_set_shape = (4*n, 1024, 3)
    train_set_local_shape = (4*n, 102, 9)
    train_label_shape = (4*n, )

    hdf5_file = h5py.File(save_path, mode='a')
    hdf5_file.create_dataset('train_set', train_set_shape, np.float32)  # be careful about the dtype
    hdf5_file.create_dataset('train_labels', train_label_shape, np.uint8)
    hdf5_file.create_dataset('train_set_local', train_set_local_shape, np.float32)
    hdf5_file["train_set"][...] = pc_tile
    hdf5_file["train_labels"][...] = pc_label
    hdf5_file["train_set_local"][...] = pc_key_feature
    hdf5_file.close()


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

        max_nb_nei_pts = np.max(np.asarray([max_nb_nei_pts, current]), axis=0)

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

    np_arr2.astype(int)

    pts_r_cov = get_pts_cov(point_cloud, np_arr2)  # np_arr2 is b x n  b x n x 3 x 3

    eigen_val, _ = np.linalg.eigh(pts_r_cov)  # b x n x 3 orderd

    idx = np.argpartition(eigen_val[:, :, 0], nb_key_pts, axis=1)

    # print(eigen_val[idx])
    key_idx = idx[:, 0:nb_key_pts]

    # print('key points coordinates:', point_cloud[idx, :], 'shape:', point_cloud[idx, :].shape)
    # b_dix = np.indices((batchsize, nb_key_pts))[1]  # b x nb_key
    # print('b_dix: ', b_dix, 'shape:', b_dix.shape)
    # batch_idx = np.concatenate([np.expand_dims(b_dix, axis=-1), np.expand_dims(idx, axis=-1)], axis=-1)  # b x nb_key x 2

    key_eig_val = np.empty((batchsize, nb_key_pts, 3))  # b x nb_keypoints x 3
    for i in range(batchsize):
        key_eig_val[i, :, :] = eigen_val[i, key_idx[i, :], :]

    np_key_arr1 = np.empty((batchsize, nb_key_pts, np_arr1.shape[2]))                                     # np_arr1: b x n x nei1  to  b x  nb_key x  nei1
    np_key_arr3 = np.empty((batchsize, nb_key_pts, np_arr3.shape[2]))
    np_key_arr1[:] = np.nan
    np_key_arr3[:] = np.nan
    for i in range(batchsize):
        np_key_arr1[i, :, :] = np_arr1[i, key_idx[i, :], :]
        np_key_arr3[i, :, :] = np_arr3[i, key_idx[i, :], :]

    key_pts_cov1 = get_pts_cov(point_cloud, np_key_arr1)  #    np_arr1: b x nb_key x nei1     b x nb_key x 3 x 3
    key_pts_cov3 = get_pts_cov(point_cloud, np_key_arr3)  #    np_arr3: b x nb_key x nei3     b x nb_key x 3 x 3

    key_eig_val2 = key_eig_val  # ordered
    key_eig_val1, _ = np.linalg.eigh(key_pts_cov1)  # b x nb_key_pts x 3 ordered
    key_eig_val3, _ = np.linalg.eigh(key_pts_cov3)  # b x nb_key_pts x 3 ordered

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

            this_pt_nei_idx = pts_r_neirhbor_idx[i, j, :][pts_r_neirhbor_idx[i, j, :] >= 0].astype(np.int32)
            #print('this_pt_nei_idx:', this_pt_nei_idx)
            neighbor_pts = pc[i, this_pt_nei_idx, :]  # 1 x nb x 3
            #print('neighbor_pts shape:', neighbor_pts.shape)
            if neighbor_pts.size == 0:
                cov[i, j, :, :] = np.eye(3)
            else:
                cov[i, j, :, :] = np.cov(neighbor_pts, rowvar=False)

    return np.float32(cov)


if __name__ == "__main__":
    save_data(save_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/noise_out lier/111normallized_project_data.h5',
              base_path='/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/noise_out lier', n=5000)