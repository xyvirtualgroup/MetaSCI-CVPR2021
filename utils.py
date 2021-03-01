"""
@author : Ziheng Cheng, Bo Chen
@Email : zhcheng@stu.xidian.edu.cn      bchen@mail.xidian.edu.cn

Description:


Citation:
    The code prepares for ECCV 2020

Contact:
    Ziheng Cheng
    zhcheng@stu.xidian.edu.cn
    Xidian University, Xi'an, China

    Bo Chen
    bchen@mail.xidian.edu.cn
    Xidian University, Xi'an, China

LICENSE
=======================================================================

The code is for research purpose only. All rights reserved.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

Copyright (c), 2020, Ziheng Cheng
zhcheng@stu.xidian.edu.cn

"""


import scipy.io as scio
import numpy as np


def generate_masks(mask_path):
    mask = scio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask_s = np.sum(mask, axis=2)
    index = np.where(mask_s == 0)
    mask_s[index] = 1

    return mask.astype(np.float32), mask_s.astype(np.float32)

def generate_masks_metaTest(mask_path):
    mask = scio.loadmat(mask_path + '/Mask.mat')
    mask = mask['Mask']
    mask_s = np.sum(mask, axis=2)
    index = np.where(mask_s == 0)
    mask_s[index] = 1

    mask = np.transpose(mask, [3, 0, 1, 2])
    mask_s = np.transpose(mask_s, [2, 0, 1])

    mask = mask[3]
    mask_s = mask_s[3]

    return mask.astype(np.float32), mask_s.astype(np.float32)

def generate_masks_metaTest_v2(mask_path):
    mask = scio.loadmat(mask_path + '/Mask.mat')
    mask = mask['Mask']
    mask_s = np.sum(mask, axis=2)
    index = np.where(mask_s == 0)
    mask_s[index] = 1

    mask = np.transpose(mask, [3, 0, 1, 2])
    mask_s = np.transpose(mask_s, [2, 0, 1])

    return mask.astype(np.float32), mask_s.astype(np.float32)

def generate_masks_metaTest_v3(mask_path):
    mask = scio.loadmat(mask_path)
    mask = mask['mask']
    mask_s = np.sum(mask, axis=2)
    index = np.where(mask_s == 0)
    mask_s[index] = 1

    return mask.astype(np.float32), mask_s.astype(np.float32)

def generate_masks_MAML(mask_path, num_task):
    mask = scio.loadmat(mask_path + '/Mask.mat')
    mask = mask['Mask']
    mask_s = np.sum(mask, axis=2)
    index = np.where(mask_s == 0)
    mask_s[index] = 1

    mask = np.transpose(mask, [3, 0, 1, 2])
    mask_s = np.transpose(mask_s, [2, 0, 1])

    mask = mask[:num_task]
    mask_s = mask_s[:num_task]

    return mask.astype(np.float32), mask_s.astype(np.float32)




