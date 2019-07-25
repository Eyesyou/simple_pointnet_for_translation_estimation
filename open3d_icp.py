import open3d as o3d
import numpy as np
import copy
from show_pc import PointCloud
from sklearn import preprocessing
from mayavi import mlab


def draw_registration_result(source, target):
    if isinstance(source, PointCloud):
        tmp = source
        source = o3d.PointCloud()
        source.points = o3d.Vector3dVector(tmp.position)
    elif isinstance(source, np.ndarray):
        tmp = source
        source = o3d.PointCloud()
        source.points = o3d.Vector3dVector(tmp)
    if isinstance(target, PointCloud):
        tmp = target
        target = o3d.PointCloud()
        target.points = o3d.Vector3dVector(tmp.position)
    elif isinstance(target, np.ndarray):
        tmp = target
        target = o3d.PointCloud()
        target.points = o3d.Vector3dVector(tmp)

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    o3d.draw_geometries([source_temp, target_temp], window_name='Open3D', width=1920, height=1080, left=0, top=0)


def icp_two_pc(source, target, draw_result=False, point2plane=False):
    """
    return target * result = source
    :param source:
    :param target:
    :param draw_result:
    :param point2plane:
    :return:
    """
    if isinstance(source, PointCloud):
        tmp = source
        source = o3d.PointCloud()
        source.points = o3d.Vector3dVector(tmp.position)
    elif isinstance(source, np.ndarray):
        tmp = source
        source = o3d.PointCloud()
        source.points = o3d.Vector3dVector(tmp)
    if isinstance(target, PointCloud):
        tmp = target
        target = o3d.PointCloud()
        target.points = o3d.Vector3dVector(tmp.position)
    elif isinstance(target, np.ndarray):
        tmp = target
        target = o3d.PointCloud()
        target.points = o3d.Vector3dVector(tmp)

    if point2plane:
        result = o3d.registration_icp(source, target, 0.002,
                                      estimation_method=o3d.TransformationEstimationPointToPlane())
    else:
        print('using point to point icp method')
        result = o3d.registration_icp(source, target, 0.002,
                                      estimation_method=o3d.TransformationEstimationPointToPoint())
    transformation = result.transformation
    print('result transformation:', transformation)

    if draw_result:
        source.transform(transformation)  # source point cloud apply the transformation to get target point cloud
        draw_registration_result(source, target)
    return transformation


def draw_normal(point_cloud):
    """

    :param point_cloud:
    :return:
    """
    if isinstance(point_cloud, PointCloud):
        tmp = point_cloud
        source = o3d.PointCloud()
        source.points = o3d.Vector3dVector(tmp.position)
    result = o3d.estimate_normals(source, o3d.KDTreeSearchParamHybrid(
            radius=50, max_nn=9))
    print('result is', result, 'result type is', type(result))


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


def nor4vec(vector):
    """
    :param vector: B x 4
    :return: B x 4
    """
    return vector/np.linalg.norm(vector, axis=1)[:, np.newaxis]


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
        quat_pos = np.concatenate([nor4vec(quat), 20 * np.random.random((batch, 4)) - 10], axis=1)     #B x 7
        homo = np_quat_pos_2_homo(quat_pos)

    batchout = np.matmul(homo, batchout) #Bx4x4 * B x 4 x n
    batchout = np.divide(batchout, batchout[:, np.newaxis, 3, :])
    batchout = batchout[:, :3, :]
    batchout = np.transpose(batchout, (0, 2, 1))

    return batchout


if __name__ == "__main__":
    source = o3d.read_point_cloud("/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/segmentationed/cloud_cluster_0.ply")
    target = o3d.read_point_cloud("/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/8object/hammer_ps.ply")
    result = icp_two_pc(source, target, draw_result=True)
    print('transformation is :', result)
    # threshold = 0.02
    # trans_init = np.asarray(
    #             [[0.862, 0.011, -0.507,  0.5],
    #             [-0.139, 0.967, -0.215,  0.7],
    #             [0.487, 0.255,  0.835, -1.4],
    #             [0.0, 0.0, 0.0, 1.0]])
    # draw_registration_result(source, target, trans_init)
    # print("Initial alignment")
    # evaluation = o3d.evaluate_registration(source, target,
    #         threshold, trans_init)
    # print(evaluation)
    #
    # print("Apply point-to-point ICP")
    # reg_p2p = o3d.registration_icp(source, target, threshold, trans_init,
    #         o3d.TransformationEstimationPointToPoint())
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # print("")
    # draw_registration_result(source, target, reg_p2p.transformation)
    #
    # print("Apply point-to-plane ICP")
    # reg_p2l = o3d.registration_icp(source, target, threshold, trans_init,
    #         o3d.TransformationEstimationPointToPlane())
    # print(reg_p2l)
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    # print("")
    # draw_registration_result(source, target, reg_p2l.transformation)


    # base_path = '/home/sjtu/Documents/ASY/point_cloud_deep_learning/simple_pointnet for translation estimation/pointcloud/fullbodyanya1.ply'
    # pc = PointCloud(base_path)
    # draw_normal(pc)
    #
    # quaternion_range = [0, 0.01]
    # translation_range = [0, 0.01]
    # ran_pos = np.concatenate(
    #     [np.random.uniform(size=(1, 1), low=0.9, high=1.),
    #      np.random.uniform(size=(1, 1), low=quaternion_range[0], high=quaternion_range[1]),
    #      np.random.uniform(size=(1, 1), low=quaternion_range[0], high=quaternion_range[1]),
    #      np.random.uniform(size=(1, 1), low=quaternion_range[0], high=quaternion_range[1]),
    #      np.random.uniform(size=(1, 3), low=translation_range[0], high=translation_range[1])],
    #     axis=1)  # random_ROTATION_and POSITION, batch x 7
    # print('ran_pos:***', ran_pos)
    # ran_pos = np.concatenate([preprocessing.normalize(ran_pos[:, :4], norm='l2'), ran_pos[:, 4:7]], axis=1) # normalize random pose
    # print('ran_pos:**********', ran_pos)
    # ran_pos = np_quat_pos_2_homo(ran_pos)
    #
    # print('random pos is:', ran_pos)
    #
    # pc = PointCloud('anya_2048_1.ply')
    # pc2 = PointCloud('anya_2048_2.ply')
    # pc.normalize()
    # pc2.normalize()
    #
    # pc2 = PointCloud(apply_np_homo(pc2.position[np.newaxis, :], ran_pos))
    # print('before reg:')
    # draw_registration_result(pc, pc2)
    # print('after registration：')
    # result = icp_two_pc(pc.position, pc2.position, draw_result=True)
    # print('difference is :', np.abs(ran_pos-result))