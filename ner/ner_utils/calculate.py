"""
calculate.py: NER 算法中所有计算模块
by: qliu
update date: 2021-12-16
"""
import torch
import os
import numpy as np


def argmax(vec):
    """
    找出同行下，最大值所处的列

    Args:
        - vec: torch.tensor, 二维张量

    """
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    """
    计算对数指数和
    Args:
        vec: torch.tensor, 二维张量 shape (1,m)
    """
    max_score = vec[0, argmax(vec)]  # 每条路径此节点的最大值
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.shape[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_mat(log_M, axis=-1):
    """
    shape (n, m)
    """
    return torch.max(log_M, axis)[0] + torch.log(torch.exp(log_M - torch.max(log_M, axis)[0][:, None]).sum(axis))


def log_sum_exp_batch(log_Tensor, axis=-1):
    """
    shape (batch_size, n, m)
    """
    return torch.max(log_Tensor, axis)[0]+torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis))


def count_f1_score(y_true, y_pred):
    """
    "X", "START", "END", "O" 不计入计算
    对应的 label_id 依次为: 0, 1, 2, 3, 详见 data_loader.py : DataLoader
    """
    start_idx = 4
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    pred_positive_num = len(y_pred[y_pred >= start_idx])
    pred_correct_num = (np.logical_and(
        y_true == y_pred, y_true >= start_idx)).sum()
    true_positive_num = len(y_true[y_true >= start_idx])

    if pred_positive_num == 0:
        precision = 0.000001
    else:
        precision = pred_correct_num / pred_positive_num

    if true_positive_num == 0:
        recall = 0.000001
    else:
        recall = pred_correct_num / true_positive_num

    if precision + recall == 0:
        if precision * recall == 0:
            f1 = 0.000001
        else:
            f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def warmup_linear(x, warmup=0.002):
    """
    梯度更新学习率
    """
    if x < warmup:
        return x/warmup
    return 1.0 - x
