import tensorflow as tf
import numpy as np
import cv2
from show_pc import *
from read_data import *


if __name__ == '__main__':
    a = np.empty([5, 5])
    a[0, 0] = 1
    a[1, 1] = None
    print(a)