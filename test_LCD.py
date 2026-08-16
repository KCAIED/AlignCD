import argparse
import datetime
import gc
import os
import pickle
import random
import time
import shutil
import json

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

from metrics import CDLoss
from models import LCD
from process import my_load_dataset

torch.cuda.empty_cache()
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = "cpu"
torch.autograd.set_detect_anomaly(True)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    # load dataset
    _, _, source_concept_num, \
        _, test_loader, \
            Q, student_cog_emb, questiom_cog_emb, \
                student_sim, _ = my_load_dataset(args.source_data_dir, args.batch_size, 
                                                 shuffle=args.shuffle, element_type=args.element_type)
    
    # print(student_cog_emb.shape)
    # print(questiom_cog_emb.shape)

    _, _, concept_num, \
        _, _, \
            target_Q, target_student_cog_emb, target_questiom_cog_emb, \
                _, _ = my_load_dataset(args.target_data_dir, args.batch_size, 
                                       shuffle=args.shuffle, element_type=args.element_type)

    # print(target_student_cog_emb.shape)
    # print(target_questiom_cog_emb.shape)
    # build models
    source_model = LCD(Q, student_cog_emb, questiom_cog_emb, args.hidden_dim, source_concept_num, 
                       base=args.base, use_llm=args.use_llm)
    target_model = LCD(target_Q, target_student_cog_emb, target_questiom_cog_emb, 
                       args.hidden_dim, concept_num, base=args.base, use_llm=args.use_llm)
    
    final_model = LCD(Q, student_cog_emb, questiom_cog_emb, args.hidden_dim, concept_num, 
                       base=args.base, use_llm=args.use_llm)
    
    for name, param in source_model.named_parameters():
        print(name, param.shape)
    print("*****"*10)
    for name, param in target_model.named_parameters():
        print(name, param.shape)
    print("*****"*10)
    for name, param in final_model.named_parameters():
        print(name, param.shape)

    source_model_info = 'exp{}_{}_{}_{}_{}'.format(args.source_data_dir[5:], args.base, int(args.use_llm), 
                                                   args.element_type, args.k)
    target_model_info = 'exp{}_{}_{}_{}_{}'.format(args.target_data_dir[5:], args.base, int(args.use_llm), 
                                                   args.element_type, args.k)
    
    # 获取模型文件夹
    folder_path = "logs"

    
    source_save_dir = os.path.join(folder_path, source_model_info)
    target_save_dir = os.path.join(folder_path, target_model_info)
    print(source_save_dir, target_save_dir)
    source_model_file = os.path.join(source_save_dir, 'LCD.pt')
    target_model_file = os.path.join(target_save_dir, 'LCD.pt')
    source_model.load_state_dict(torch.load(source_model_file))
    target_model.load_state_dict(torch.load(target_model_file))
    
    replace_source = ['student.weight', 'question.weight']
    replace_target = ['concept.weight']
    params_source = dict(source_model.named_parameters())
    params_target = dict(target_model.named_parameters())
    # print(list(params_target.keys()))
    
    # time.sleep(100)
    for name, param_final in final_model.named_parameters():
        if name in replace_source:
            # 同名参数值覆盖 
            param_final.data.copy_(params_source[name].data)
        elif name in replace_target:
            param_final.data.copy_(params_target[name].data)
        else:
            param_final.data.copy_(params_target[name].data)
    """
    for name, param in source_model.named_parameters():
        print(name, param.shape)
    print("*****"*10)
    for name, param in target_model.named_parameters():
        print(name, param.shape)
    return
    """
    cd_loss = CDLoss(k=args.k)


    if args.cuda:
        source_model = source_model.to(device)


    # record the result of experiment
    def test():
    
        loss_test = []
        auc_test = []
        acc_test = []
        rmse_test = []
        doa_test = []

        final_model.eval()
        with torch.no_grad():
            for batch_idx, (students, questions, answers) in enumerate(test_loader):
                
                if args.cuda:
                    students = students.to(device)
                    questions = questions.to(device)
                    answers = answers.to(device)
                    # Q = Q.to(device)

                pred_res, hs_state, _, hs_mas, _, temp_Q = final_model(students, questions)
                
                loss, auc, acc, rmse = cd_loss(pred_res, answers, temp_Q, temp_Q)
                doa = 0.0
                
                loss = float(loss.cpu().detach().numpy())
                if auc != -1 and acc != -1 and rmse != -1:
                    auc_test.append(auc)
                    acc_test.append(acc)
                    rmse_test.append(rmse)
                doa_test.append(doa)

                loss_test.append(loss)
                del loss
   

        record = {
            'loss_test': np.mean(loss_test),
            'auc_test': np.mean(auc_test),
            'acc_test': np.mean(acc_test),
            'rmse_test': np.mean(rmse_test),
            'doa_test': np.mean(doa_test)
        }

        del loss_test
        del auc_test
        del acc_test
        del rmse_test
        del doa_test
        gc.collect()
        if args.cuda:
            torch.cuda.empty_cache()
        return record

    # test model
    record = test()


    class NumpyEncoder(json.JSONEncoder):
        """自定义JSON编码器，处理numpy数据类型"""
        def default(self, obj):
            if isinstance(obj, np.float32):
                return float(obj)  # 转换为Python float
            elif isinstance(obj, np.float64):
                return float(obj)

    record_file = 'test/{}_{}_{}_{}_{}_{}.json'.format(args.source_data_dir[5:], args.target_data_dir[5:],
                                                     args.base, int(args.use_llm), 
                                                     args.element_type, args.k)
    with open(record_file, 'w', encoding='utf-8') as f:
        json.dump(record, f, cls=NumpyEncoder, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cpu", help='Disables CUDA training.')
    parser.add_argument('--no-cuda', action='store_false', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--source-data-dir', type=str, default='data/ASSIST0910', help='Source Dataset.')
    parser.add_argument('--target-data-dir', type=str, default='data/ASSIST17', help='Target Dataset.')
    parser.add_argument('--save-dir', type=str, default='logs',
                        help='Where to save the trained model, leave empty to not save anything.')

    parser.add_argument('--model', type=str, default='LCD', help='Model type to use.')
    parser.add_argument('--base', type=str, default='ncd', help='Base CD Model.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden knowledge states.')
    parser.add_argument('--bias', type=bool, default=True, help='Whether to add bias for neural network layers.')
    parser.add_argument('--use_llm', type=int, default=1, help='whether to use llm')
    parser.add_argument('--element_type', type=str, default="no", help='which embedding to use')
    parser.add_argument('--k', type=float, default=0.1, help='the para of contrastive loss.')

    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--batch-size', type=int, default=2048, help='Number of samples per batch.')
    parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the dataset or not.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--test', type=bool, default=False, help='Whether to test for existed model.')
    parser.add_argument('--test-model-dir', type=str, default='logs/expLCD', help='Existed model file dir.')

    args = parser.parse_known_args()[0]    #  这里与放在py文件中不同
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)