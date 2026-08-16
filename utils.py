import numpy as np
import pandas as pd
from collections import deque
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# 按字段划分训练集、测试集
def split_train_test(df, group_col, test_size=0.2, random_state=42):
    # 获取所有唯一的分组值
    groups = df[group_col].unique()
    
    train_dfs = []
    test_dfs = []
    
    for group in groups:
        # 获取当前分组的数据
        group_df = df[df[group_col] == group]
        
        # 计算测试集大小
        n_test = max(1, int(len(group_df) * test_size))  # 确保每个分组至少有1条测试数据
        
        # 随机打乱并划分
        shuffled = group_df.sample(frac=1, random_state=42)
        test = shuffled.iloc[:n_test]
        train = shuffled.iloc[n_test:]
        
        train_dfs.append(train)
        test_dfs.append(test)
    
    # 合并所有分组的训练集和测试集
    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)
    
    return train_df, test_df


# Calculate accuracy of prediction result and its corresponding label
# output: tensor, labels: tensor
def accuracy(pred_answers, real_answers):
    X = torch.zeros(real_answers.shape, device=real_answers.device)
    X[pred_answers>0.5] = 1.0
    acc = torch.where(X == real_answers, 
                      torch.ones_like(X, device=X.device), 
                      torch.zeros_like(X, device=X.device)).float()
    acc = torch.mean(acc, dim=0)

    return acc
