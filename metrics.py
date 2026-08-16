import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils import accuracy


# Calculate RMSE/AUC/ACC
class CDLoss(nn.Module):

    def __init__(self, k=0.0):
        super(CDLoss, self).__init__()
        self.k = k


    def forward(self, pred_answers, real_answers, Q, temp_Q):
        r"""
        Parameters:
            pred_answers: the correct probability of questions answered at the next timestamp
            real_answers: the real results(0 or 1) of questions answered at the next timestamp
            state: the state of the student
        Shape:
            pred_answers: [batch_size]
            real_answers: [batch_size]
            Q, temp_Q: [batch_size, concept_num]
        Return:
        """
        batch_size = real_answers.shape[0]
        concept_num = temp_Q.shape[1]

        # calculate auc and accuracy metrics
        try:
            y_true = real_answers.cpu().detach().numpy()
            y_pred = pred_answers.cpu().detach().numpy()
            auc = roc_auc_score(y_true, y_pred)  # may raise ValueError
            
            acc = accuracy(pred_answers, real_answers)
            acc = float(acc.cpu().detach().numpy())

            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        except ValueError as e:
            auc, acc, rmse = -1, -1, -1
        

        loss_func = nn.BCELoss()
        loss_func_extra = nn.MSELoss()

        main_loss = loss_func(pred_answers, real_answers) + \
            self.k * loss_func_extra(Q, temp_Q) / concept_num

        loss = main_loss

        return loss, auc, acc, rmse


def doa_eval(y_true, y_pred):
    doa = []
    doa_support = 0
    z_support = 0
    for knowledge_label, knowledge_pred in zip(y_true, y_pred):
        _doa = 0
        _z = 0
        for label, pred in zip(knowledge_label, knowledge_pred):
            if sum(label) == len(label) or sum(label) == 0:
                continue
            pos_idx = []
            neg_idx = []
            for i, _label in enumerate(label): # 找出所有(1, 0) pair
                if _label == 1:
                    pos_idx.append(i)
                else:
                    neg_idx.append(i)
            pos_pred = pred[pos_idx]
            neg_pred = pred[neg_idx]
            invalid = 0
            for _pos_pred in pos_pred:
                _doa += len(neg_pred[neg_pred < _pos_pred])
                invalid += len(neg_pred[neg_pred == _pos_pred])
            _z += (len(pos_pred) * len(neg_pred)) - invalid
        if _z > 0:
            doa.append(_doa / _z)
            z_support += _z # 有效pair个数
            doa_support += 1 # 有效doa
    
    if len(doa) == 0:
        return 0.0  # 或者返回 np.nan，根据你的聚合逻辑决定
    
    return float(np.mean(doa))


def doa_report(stu_id, exer_id, label, user_emb, Q_mat):
    r"""
    Shape:
        stu_id, exer_id, label: [batch_size,]
        user_emb: [batch_size, concept_num]
        Q_mat: [question_num, concept_num]
    Return:
    """
    if user_emb is None:
        return 0.0

    knowledges = []
    knowledge_item = []
    knowledge_user = []
    knowledge_truth = []
    knowledge_theta = []
    for s, (user, item, score) in enumerate(zip(stu_id, exer_id, label)):
        theta = user_emb[s].cpu().detach().numpy()
        knowledge = Q_mat[item].cpu().detach().numpy()
        if isinstance(theta, list) or isinstance(theta, np.ndarray):
            for i, (theta_i, knowledge_i) in enumerate(zip(theta, knowledge)):
                if knowledge_i == 1: 
                    knowledges.append(i) # 知识点ID
                    knowledge_item.append(item.cpu()) # Item ID
                    knowledge_user.append(user.cpu()) # User ID
                    knowledge_truth.append(score.cpu()) # score
                    knowledge_theta.append(theta_i) # matser
        else:  # pragma: no cover
            for i, knowledge_i in enumerate(knowledge):
                if knowledge_i == 1:
                    knowledges.append(i)
                    knowledge_item.append(item.cpu())
                    knowledge_user.append(user.cpu())
                    knowledge_truth.append(score.cpu())
                    knowledge_theta.append(theta)

    knowledge_df = pd.DataFrame({
        "knowledge": knowledges,
        "user_id": knowledge_user,
        "item_id": knowledge_item,
        "score": knowledge_truth,
        "theta": knowledge_theta
    })
    knowledge_ground_truth = []
    knowledge_prediction = []
    for _, group_df in knowledge_df.groupby("knowledge"):
        _knowledge_ground_truth = []
        _knowledge_prediction = []
        for _, item_group_df in group_df.groupby("item_id"):
            _knowledge_ground_truth.append(item_group_df["score"].values)
            _knowledge_prediction.append(item_group_df["theta"].values)
        knowledge_ground_truth.append(_knowledge_ground_truth)
        knowledge_prediction.append(_knowledge_prediction)

    return doa_eval(knowledge_ground_truth, knowledge_prediction)