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

def clear_folder(folder_path):
    """
    使用shutil模块删除文件夹及其所有内容
    """
    try:
        # 检查文件夹是否存在
        if os.path.exists(folder_path):
            # 删除文件夹及其所有内容
            shutil.rmtree(folder_path)
            print(f"文件夹 '{folder_path}' 已成功删除")
        else:
            print(f"文件夹 '{folder_path}' 不存在")
    except Exception as e:
        print(f"删除文件夹时出错: {e}")


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

    # 模型命名（含后续的模型/结果）
    model_info = '{}_{}_{}_{}_{}'.format(args.data_dir[5:], args.base, int(args.use_llm), 
                                            args.element_type, args.k)
    
    # Save model and meta-data. Always saves in a new sub-folder.
    log = None
    save_dir = args.save_dir
    if args.save_dir:
        exp_counter = 0
        # now = datetime.datetime.now()
        timestamp = int(time.time()*1000000)
        model_file_name = args.model
        save_dir = '{}/exp{}/'.format(args.save_dir, model_info)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        meta_file = os.path.join(save_dir, 'metadata.pkl')
        model_file = os.path.join(save_dir, model_file_name + '.pt')
        optimizer_file = os.path.join(save_dir, model_file_name + '-Optimizer.pt')
        log_file = os.path.join(save_dir, 'log.txt')
        log = open(log_file, 'w')
        pickle.dump({'args': args}, open(meta_file, "wb"))
    else:
        print("WARNING: No save_dir provided!" + "Testing (within this script) will throw an error.")

    # load dataset
    student_num, question_num, concept_num, \
        train_loader, test_loader, \
            Q, student_cog_emb, questiom_cog_emb, \
                student_sim, question_sim = my_load_dataset(args.data_dir, args.batch_size, shuffle=args.shuffle, 
                                                            element_type=args.element_type)

    # build models
    model = LCD(Q, student_cog_emb, questiom_cog_emb, args.hidden_dim, concept_num, base=args.base, use_llm=args.use_llm)
    cd_loss = CDLoss(k=args.k)

    # build optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    if args.cuda:
        model = model.to(device)


    # record the result of experiment
    record_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    record = {"Setting":
                {"dataset": args.data_dir,
                 "base": args.base,
                 "hidden_dim": args.hidden_dim,
                 "use_llm": args.use_llm,
                 "element_type": args.element_type,
                 "k": args.k,},
                
                "Training": []
            }

    def train(epoch, best_test_auc, record):
    
        t = time.time()
        loss_train = []
        auc_train = []
        acc_train = []
        rmse_train = []
        doa_train = []
        model.train()

        for batch_idx, (students, questions, answers) in enumerate(train_loader):
            
            time_start = time.time()
            
            if args.cuda:
                students = students.to(device)
                questions = questions.to(device)
                answers = answers.to(device)
                # Q = Q.to(device)

            pred_res, hs_state, _, hs_mas, Q, temp_Q = model(students, questions)
            
            loss, auc, acc, rmse = cd_loss(pred_res, answers, Q, temp_Q)
            doa = 0.0
            
            if auc != -1 and acc != -1 and rmse != -1:
                auc_train.append(auc)
                acc_train.append(acc)
                rmse_train.append(rmse)
            doa_train.append(doa)

            print('batch idx: ', batch_idx, 'loss: ', loss.item(), 
                'auc: ', auc, 'acc: ', acc, 'rmse: ', rmse, 'doa: ', doa, end=' ')
            loss_train.append(float(loss.cpu().detach().numpy()))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            del loss
            print('cost time: ', str(time.time() - time_start))


        loss_test = []
        auc_test = []
        acc_test = []
        rmse_test = []
        doa_test = []

        model.eval()
        with torch.no_grad():
            for batch_idx, (students, questions, answers) in enumerate(test_loader):
                
                if args.cuda:
                    students = students.to(device)
                    questions = questions.to(device)
                    answers = answers.to(device)
                    # Q = Q.to(device)

                pred_res, hs_state, _, hs_mas, Q, temp_Q = model(students, questions)
                
                loss, auc, acc, rmse = cd_loss(pred_res, answers, Q, temp_Q)
                doa = 0.0
                
                loss = float(loss.cpu().detach().numpy())
                if auc != -1 and acc != -1 and rmse != -1:
                    auc_test.append(auc)
                    acc_test.append(acc)
                    rmse_test.append(rmse)
                doa_test.append(doa)

                loss_test.append(loss)
                del loss

        print('Epoch: {:04d}'.format(epoch),
            'loss_train: {:.10f}'.format(np.mean(loss_train)),
            'auc_train: {:.10f}'.format(np.mean(auc_train)),
            'acc_train: {:.10f}'.format(np.mean(acc_train)),
            'rmse_train: {:.10f} '.format(np.mean(rmse_train)),
            'doa_train: {:.10f} '.format(np.mean(doa_train)),
            'loss_test: {:.10f}'.format(np.mean(loss_test)),
            'auc_test: {:.10f}'.format(np.mean(auc_test)),
            'acc_test: {:.10f}'.format(np.mean(acc_test)),
            'rmse_test: {:.10f} '.format(np.mean(rmse_test)),
            'doa_test: {:.10f} '.format(np.mean(doa_test)),
            'time: {:.4f}s'.format(time.time() - t))
        
        if args.save_dir and np.mean(auc_test) >= best_test_auc:
            print('Best model so far, saving...')
            torch.save(model.state_dict(), model_file)
            torch.save(optimizer.state_dict(), optimizer_file)

            record["Training"].append({'epoch': epoch,
                                    'loss_train': np.mean(loss_train),
                                    'auc_train': np.mean(auc_train),
                                    'acc_train': np.mean(acc_train),
                                    'rmse_train': np.mean(rmse_train),
                                    'doa_train': np.mean(doa_train),
                                    'loss_test': np.mean(loss_test),
                                    'auc_test': np.mean(auc_test),
                                    'acc_test': np.mean(acc_test),
                                    'rmse_test': np.mean(rmse_test),
                                    'doa_test': np.mean(doa_test),
                                    'best': True,
                                    'time': time.time() - t})
        else:
            record["Training"].append({'epoch': epoch,
                                    'loss_train': np.mean(loss_train),
                                    'auc_train': np.mean(auc_train),
                                    'acc_train': np.mean(acc_train),
                                    'rmse_train': np.mean(rmse_train),
                                    'doa_train': np.mean(doa_train),
                                    'loss_test': np.mean(loss_test),
                                    'auc_test': np.mean(auc_test),
                                    'acc_test': np.mean(acc_test),
                                    'rmse_test': np.mean(rmse_test),
                                    'doa_test': np.mean(doa_test),
                                    'best': False,
                                    'time': time.time() - t})
        log.flush()
        res = np.mean(auc_test)
        del loss_train
        del auc_train
        del acc_train
        del rmse_train
        del doa_train
        del loss_test
        del auc_test
        del acc_test
        del rmse_test
        del doa_test
        gc.collect()
        if args.cuda:
            torch.cuda.empty_cache()
        return res

    # Train model
    if args.test is False:
        
        print('start training!')
        t_total = time.time()
        best_test_auc = -np.inf
        best_epoch = 0
        for epoch in range(args.epochs):
            val_auc = train(epoch, best_test_auc, record)
            if val_auc > best_test_auc:
                best_test_auc = val_auc
                best_epoch = epoch
        print("Optimization Finished!")
        print("Best Epoch: {:04d}".format(best_epoch))
        if args.save_dir:
            print("Best Epoch: {:04d}".format(best_epoch), file=log)
            log.flush()

    # 训练完成后，删除训练文件夹，只保留结果
    # clear_folder(save_dir)
    # now = datetime.datetime.now()
    timestamp = int(time.time()*1000000)
    record_file = os.path.join('result/', '{}.json'.format(model_info))
    class NumpyEncoder(json.JSONEncoder):
        """自定义JSON编码器，处理numpy数据类型"""
        def default(self, obj):
            if isinstance(obj, np.float32):
                return float(obj)  # 转换为Python float
            elif isinstance(obj, np.float64):
                return float(obj)
    with open(record_file, 'w', encoding='utf-8') as f:
        json.dump(record, f, cls=NumpyEncoder, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0", help='Disables CUDA training.')
    parser.add_argument('--no-cuda', action='store_false', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--data-dir', type=str, default='data/ASSIST0910', help='Data dir for loading input data.')
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