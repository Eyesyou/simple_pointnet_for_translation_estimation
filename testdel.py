import time
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import h5py
from show_pc import PointCloud
import cv2 as cv
import open3d as o3d
# dir = '/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/noise_out lier/lab4'
# dir_list = os.listdir(dir)
# os.chdir(dir)
# # print(dir_list)
# for i in range(len(dir_list)):
#     if re.search('txt', dir_list[i]):
#         id = re.search(r'project(\d*)', dir_list[i]).group(1)
#         new_name = 'lab_project'+str(id)+'.txt'
#         os.rename(dir_list[i], new_name)

# readh5 = h5py.File('/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/data/testply/real_single_1024n.h5')  # file path
# readh55 = h5py.File('/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/8object0.04noise/mykeyptssimuN_data.h5')
#
# pc_tile_r = readh5['train_set'][:]
# pc_local_eigs_r = readh5['train_set_local'][:]
# pc_label_r = readh5['train_labels'][:]
#
# pc_tile = readh55['train_set'][0:183]  # 20000 * 1024 * 3
# pc_local_eigs = readh55['train_set_local'][0:183]  # 20000 * 102 * 9
# pc_label = readh55['train_labels'][0:183]
#
#
# save_path = '/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/data/testply/real_single_1024nc.h5'
# hdf5_file = h5py.File(save_path, mode='a')
# hdf5_file.create_dataset('train_set', (200, 1024, 3), np.float32)  # be careful about the dtype
# hdf5_file.create_dataset('train_set_local', (200, 102, 9), np.float32)
# hdf5_file.create_dataset('train_labels', (200,), np.uint8)
# hdf5_file["train_set"][...] = np.concatenate([pc_tile_r, pc_tile], axis=0)
# hdf5_file["train_set_local"][...] = np.concatenate([pc_local_eigs_r, pc_local_eigs], axis=0)
# hdf5_file["train_labels"][...] = np.concatenate([pc_label_r, pc_label], axis=0)
# hdf5_file.close()
#
# print('pc_tile:', pc_tile.shape)
# print('pc_local_eigs:', pc_local_eigs.shape)
# print('pc_label:', pc_label.shape)

j = 5.69
t1 = time.time()
for i in range(1000000):
    j = j * i
t2 = time.time()
time.sleep(10)
t3 = time.time()
print('t2-t1',t2-t1)
print('t3-t2',t3-t2)