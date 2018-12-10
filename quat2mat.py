import numpy as np
def quat2mat(quat):
    ''' Symbolic conversion from quaternion to rotation matrix
    For a unit quaternion

    From: http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    '''
    w, x, y, z = quat
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])

def nor4vec(vector):
    return vector/np.linalg.norm(vector)

qua=np.array([1,0,0,0])
print(quat2mat(qua))