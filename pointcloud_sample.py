import numpy as np
import os
import sys
import math
import time

path = os.getcwd()

#打开点云文件，将数据存进数组data nx3
file = open(path + "//lingjian2.txt", 'r')
s = file.readline()
data = []
while s:
    s = s.split()
    s = [float(x) for x in s]
    data.append(s)
    s = file.readline()
file.close()
data = np.array(data)
print(data.shape)

#随机生成重要度
Importance_Ranking = list(range(0, len(data), 1))
np.random.shuffle(Importance_Ranking)

def resolution_kpts(Pointcloud, Importance_Ranking, Voxel_Size, Sampled_Number):
    """
    :param Pointcloud:  点云nx3
    :param Importanace_Ranking:重要度 每个点的重要度排序索引（浮点数） nx1
    :param Voxel_Size:体素大小
    :param Sampled_Number:采样点的个数
    :return:
    """
    ranking_set = {}   # 字典里面每个键代表一个有点的体素
    sampled_pointcloud = np.zeros((Sampled_Number, 3))  #初始化输出点云数组

    #计算点云在空间中的立方体
    distance_max = np.amax(Pointcloud, axis=0)
    distance_min = np.amin(Pointcloud, axis=0)
    #计算立方体x,y方向的体素个数
    number_x = math.ceil((distance_max[0] - distance_min[0]) / Voxel_Size)
    number_y = math.ceil((distance_max[1] - distance_min[1]) / Voxel_Size)
    # 用点云减去最小坐标再除以体素尺寸，得到的nx3为xyz方向上以体素尺寸为单位长度的坐标(浮点数)
    sequence_number = (Pointcloud-distance_min)/Voxel_Size
    for i in range(len(sequence_number)):  #对每个点
        sequence = (math.ceil(sequence_number[i][2])-1) * number_x * number_y + (math.ceil(sequence_number[i][1])-1) * \
                   number_x + math.ceil(sequence_number[i][0])  #计算这个点在体素空间中的位置
        if str(sequence) in ranking_set:
            if Importance_Ranking[i] > ranking_set[str(sequence)][0]:
                ranking_set[str(sequence)] = [Importance_Ranking[i], i]
        else:
            ranking_set[str(sequence)] = [Importance_Ranking[i], i]   #如果字典里面没有这个体素，则需要新建一个该体素的键，然后将【重要度，索引】存进去
    if len(ranking_set) < Sampled_Number:
        print("The value of Voxel_Size is too large and needs to be reduced!!!")
        raise ValueError
    sample_sequence = np.zeros(shape=[len(ranking_set), 2])
    for i, j in enumerate(ranking_set):
        sample_sequence[i, :] = ranking_set[j]    #字典里面每个键都是一个列表，保存的是一个体素内所有点的重要度，取最大的生成一个列表
    sample_sequence = sample_sequence[sample_sequence[:, 0].argsort()]  #排序，得到重要度从大到小的排序
    ind = np.empty((Sampled_Number,))
    for k in range(Sampled_Number):
        sampled_pointcloud[k, :] = Pointcloud[int(sample_sequence[k, 1]), :]
        ind[k] = int(sample_sequence[k, 1])
    return sampled_pointcloud, ind

start_time = time.clock()
for i in range(5):
    sampled_data,_ = resolution_kpts(data, Importance_Ranking, 1, 102)
end_time = time.clock()
average_time = (end_time-start_time)/5
print(average_time)
f = open("result2.txt", 'w')
for num in range(len(sampled_data)):
    f.write(str(sampled_data[num, 0]) + " " + str(sampled_data[num, 1]) + " " + str(sampled_data[num, 2]) + "\n")
f.close()









