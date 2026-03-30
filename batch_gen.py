# =============================================================================
# batch_gen.py — Batch loading utilities for SMQ
# Adapted from MS-TCN: https://github.com/yabufarha/ms-tcn
# =============================================================================

import torch
import numpy as np
import random
import os
import math

random.seed(42)

class BatchGenerator(object):
    
    """Loads skeleton feature files and produces padded batches with masks.

    Expects .npy files shaped (C, T, V, M). Applies temporal subsampling,
    pads sequences to max length in batch, and returns (batch, mask).
    """

    def __init__(self, features_path, sample_rate, num_features, 
                 num_joints, num_person):
        
        self.list_of_examples = []
        self.index = 0
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.num_features = num_features
        self.num_joints = num_joints
        self.num_person = num_person

    def reset(self):
        """Resets index and reshuffles examples."""
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        """Returns whether more batches are available.

        Returns:
            bool: True if more data remains for this epoch.
        """
        return self.index < len(self.list_of_examples)

    def read_data(self):
        """Reads and shuffles file names.
        """
        self.list_of_examples = os.listdir(self.features_path)
        random.shuffle(self.list_of_examples)

    def num_batches(self, batch_size):
        """Returns the number of batches per epoch."""
        return math.ceil(len(self.list_of_examples) / batch_size)# 向上取整

    def next_batch(self, batch_size):
        """Loads the next batch and pads variable-length sequences.

        Args:
            batch_size (int): Number of samples to load.

        Returns:
            torch.Tensor: Batch tensor (N, C, T_max, V, M).
            torch.Tensor: Mask tensor, same shape, 1=valid, 0=padded.
        """
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        for vid in batch:
            try:
                features = np.load(os.path.join(self.features_path, vid))# 加载磁盘上的npy文件
                # features是四维张量，(C, T, V, M)
                batch_input.append(features[:, ::self.sample_rate, :, :])# 默认sample_rate=1，所以不进行下采样，保持原样
            except IOError:
                print(f'Error loading {vid}')

        # lambda tensor: tensor.shape[1] 是一个匿名函数，输入一个张量，返回这个张量的第二维的大小，也就是时间维度T的长度。
        # 相当于map(get_shape, batch_input)，得到一个列表，里面是每个样本的时间维度长度。
        # map(function, iterable) 会将function应用于iterable中的每个元素，并返回一个迭代器。
        # list()函数将这个迭代器转换成一个列表。
        # 所以，这行代码是创建一个列表，里面存储batch_input的每一个元素的第二维（时间维度）的长度
        length_of_sequences = list(map(lambda tensor: tensor.shape[1], batch_input))

        #填充为0，得到一个形状为 (N, C, T_max, V, M) 的张量，其中 T_max 是当前批次中最长的时间序列长度。
        batch_input_tensor = torch.zeros(len(batch_input), self.num_features, max(length_of_sequences), self.num_joints, self.num_person, dtype=torch.float)

        #填充为0，得到一个形状为 (N, C, T_max, V, M) 的张量
        mask = torch.zeros(len(batch_input), self.num_features, max(length_of_sequences), self.num_joints, self.num_person, dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1], :, :] = torch.from_numpy(batch_input[i])# 第i个张量，即一个序列的feature(.npy)
            # torch.ones...是指创建一个全1的张量，形状和batch_input[i]一样，表示这个序列的有效部分（没有被padding掉的部分）。然后把这个全1的张量赋值给mask的对应位置，表示这个位置是有效的。
            mask[i, :, :np.shape(batch_input[i])[1], :, :] = torch.ones(np.shape(batch_input[i]))
            # 掩码是一个和数据形状相同的张量，里面只有1和0
            # 1：表示这里是真数据，请看这里
            # 0：表示这里是补丁，无效，请忽视它
            
        return batch_input_tensor, mask
