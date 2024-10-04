#!/bin/bash/python3
'''
Programming for check abstract concept ability in CNN
'''

# import Kits for analysis
import os
import torch
import re
import cv2
import numpy as np


# define resort function sort
def resort_file(FileList):
    num_rank = []
    for i in range(len(FileList)):  # generate a number_rank
        filename = FileList[i]
        filename_num = re.findall("\d+", filename)  # Find number in Str
        filename_num = int(filename_num[0])  # Trans Str to Int
        num_rank.append(filename_num)  # add file_num to list

    dictionary = dict(zip(FileList, num_rank))
    new = sorted(FileList, key=lambda x: dictionary[x])
    return new


# Main Programming
if __name__ == "__main__":
    # Read images for CNN training
    Dir_path = "D:/project/PAL/images/For_CNN/"

    # Different sub-dir
    Cor = "Cri/"
    InCor = "X/"

    # confirm set size
    FileList1 = os.listdir(f'{Dir_path}{Cor}')
    FileList1 = resort_file(FileList1)
    set_size1 = len(FileList1)
    FileList2 = os.listdir(f'{Dir_path}{InCor}')
    set_size2 = len(FileList2)
    FileList2 = resort_file(FileList2)

    # Create a Train-set (Test-set)
    img = cv2.imread(f"{Dir_path}/cri.bmp")
    data_set = np.zeros((set_size1 + set_size2) * np.size(img))
    data_set = np.reshape(data_set, (set_size1 + set_size2, np.shape(img)[2],
                                       np.shape(img)[0], np.shape(img)[1]))

    # Add list1
    for i in range(set_size1):
        path = f'{Dir_path}{Cor}{FileList1[i]}'
        img = cv2.imread(path)    # read img
        img = cv2.resize(img, (np.shape(img)[0], np.shape(img)[1]))   # re-size img
        # add img to train set
        data_set[i, 0, :, :] = img[:, :, 0]
        data_set[i, 1, :, :] = img[:, :, 1]
        data_set[i, 2, :, :] = img[:, :, 2]
    # Add list2
    for i in range(set_size2):
        path = f'{Dir_path}{InCor}{FileList2[i]}'
        img = cv2.imread(path)  # read img
        img = cv2.resize(img, (np.shape(img)[0], np.shape(img)[1]))  # re-size img
        # add img to train set
        data_set[i + set_size1, 0, :, :] = img[:, :, 0]
        data_set[i + set_size1, 1, :, :] = img[:, :, 1]
        data_set[i + set_size1, 2, :, :] = img[:, :, 2]

    train_index = np.union1d(range(0, 700), range(1000, 1700))
    train_set = data_set[train_index, :, :, :]
    print(np.shape(train_set))
    np.save(f"{Dir_path}train_set.npy", train_set)

    test_index = np.union1d(range(700, 1000), range(1700, 2000))
    test_set = data_set[test_index, :, :, :]
    print(np.shape(test_set))
    np.save(f"{Dir_path}test_set.npy", test_set)

    # Generate a init set for a new net
    img1 = cv2.imread(f"{Dir_path}/cri.bmp")
    img2 = cv2.imread(f"{Dir_path}/squ.bmp")
    init_set = np.zeros(2 * np.size(img))
    init_set = np.reshape(init_set, (2, np.shape(img)[2],
                                     np.shape(img)[0], np.shape(img)[1]))

    init_set[0, 0, :, :] = img1[:, :, 0]
    init_set[0, 1, :, :] = img1[:, :, 1]
    init_set[0, 2, :, :] = img1[:, :, 2]
    init_set[1, 0, :, :] = img2[:, :, 0]
    init_set[1, 1, :, :] = img2[:, :, 1]
    init_set[1, 2, :, :] = img2[:, :, 2]

    print(init_set)
    print(np.shape(init_set))

    np.save(f"{Dir_path}init_set.npy", init_set)













