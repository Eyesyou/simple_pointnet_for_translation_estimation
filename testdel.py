import os
import re
dir = '/media/sjtu/software/ASY/pointcloud/lab scanned workpiece/noise_out lier/lab4'
dir_list = os.listdir(dir)
os.chdir(dir)
# print(dir_list)
for i in range(len(dir_list)):
    if re.search('txt', dir_list[i]):
        id = re.search(r'project(\d*)', dir_list[i]).group(1)
        new_name = 'lab_project'+str(id)+'.txt'
        os.rename(dir_list[i], new_name)