import random
import sklearn.metrics as metrics
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from torch_geometric.data import DataLoader

from Dataset import MolNet


def load_fold_data(i, batch, cpus_per_gpu, k, dataset):

    data = MolNet(root='./dataset', dataset=dataset)

    fold_size = len(data) // k
    val_start = i * fold_size

    if i != k - 1:
                val_end = (i + 1) * fold_size
                valid_set = data[val_start:val_end]
                train_set = data[0:val_start] + data[val_end:]
    else:
                valid_set = data[val_start:]
                train_set = data[0:val_start]

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, pin_memory=True, num_workers=cpus_per_gpu, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=batch, shuffle=False, pin_memory=True, num_workers=cpus_per_gpu, drop_last=False)

    return train_loader, valid_loader


def compute_auc(output, labels):
    # 计算样本权重
    sample_weights = compute_sample_weights(labels)

    # 使用权重计算AUC
    auc = metrics.roc_auc_score(labels, output, sample_weight=sample_weights)

    return auc


def compute_accuracy(output, labels):
    # 计算样本权重
    sample_weights = compute_sample_weights(labels)

    # 计算准确率
    predictions = (output > 0.5).astype(int)
    accuracy = metrics.accuracy_score(labels, predictions, sample_weight=sample_weights)

    return accuracy


def compute_precision_recall(output, labels):
    # 计算样本权重
    sample_weights = compute_sample_weights(labels)

    # 计算精确度和召回率
    predictions = (output > 0.5).astype(int)
    precision = metrics.precision_score(labels, predictions, sample_weight=sample_weights)
    recall = metrics.recall_score(labels, predictions, sample_weight=sample_weights)

    return precision, recall


def compute_sample_weights(labels):
    # 计算样本权重，可以根据样本的标签分布等进行调整
    # 这里简单地示范了一种均衡权重的方法，可根据实际情况进行调整
    num_positive = sum(labels)
    num_negative = len(labels) - num_positive
    weight_positive = num_negative / len(labels)
    weight_negative = num_positive / len(labels)

    sample_weights = [weight_positive if label == 0 else weight_negative for label in labels]

    return sample_weights
if __name__ == "__main__":
    dataset_name = "your_dataset_name"
    batch_size = 32
    valid_size = 0.1
    test_size = 0.1
    cpus_per_gpu = 4
    task = "classification"  # 或者 "regression"，根据你的任务

    # 加载数据
    dataloaders = load_fold_data(dataset_name, batch_size, valid_size, test_size, cpus_per_gpu, task)
