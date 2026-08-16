import os
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class CDDataset(Dataset):
    def __init__(self, students, questions, answers):
        super(CDDataset, self).__init__()
        self.students = students          
        self.questions = questions                
        self.answers = answers             

    def __getitem__(self, index):
        return self.students[index], self.questions[index], self.answers[index]

    def __len__(self):
        return len(self.answers)



def pad_collate(batch):
    (students, questions, answers) = zip(*batch)

    students = torch.LongTensor(students)
    questions = torch.LongTensor(questions)
    answers = torch.FloatTensor(answers)

    return students, questions, answers



def my_load_dataset(file_path, batch_size, shuffle=True, element_type="no"):
    r"""
    Parameters:
        file_path: input file path of knowledge tracing data
        batch_size: the size of a student batch
        shuffle: whether to shuffle the dataset or not
    Return:
        student_num: the number of all students
        question_num: the number of all questions
        concept_num: the number of all concepts
        Q: Q Matrix
        cog_emb: LLM Embedding
        train_data_loader: data loader of the training dataset
        test_data_loader: data loader of the test dataset
    """

    train_dataset_path = os.path.join(file_path, "train.csv")
    test_dataset_path = os.path.join(file_path, "test.csv")
    train_df, test_df = pd.read_csv(train_dataset_path), pd.read_csv(test_dataset_path)
    print('train_size: ', train_df.shape[0], 'test_size: ', test_df.shape[0])
    
    Q_dataset_path = os.path.join(file_path, "Q.npy")
    Q = torch.from_numpy(np.load(Q_dataset_path))
    # 执行一定的掩码操作
    question_num, concept_num = Q.shape

    train_dataset_path = os.path.join(file_path, "train.csv")
    test_dataset_path = os.path.join(file_path, "test.csv")

    if element_type == "A":
        student_cog_emb_dataset_path = os.path.join(file_path, "student_final_A_emb.npy")
        question_cog_emb_dataset_path = os.path.join(file_path, "question_final_A_emb.npy")
    elif element_type == "B":
        student_cog_emb_dataset_path = os.path.join(file_path, "student_final_B_emb.npy")
        question_cog_emb_dataset_path = os.path.join(file_path, "question_final_B_emb.npy")
    elif element_type == "C":
        student_cog_emb_dataset_path = os.path.join(file_path, "student_final_C_emb.npy")
        question_cog_emb_dataset_path = os.path.join(file_path, "question_final_C_emb.npy")
    else:
        student_cog_emb_dataset_path = os.path.join(file_path, "student_raw_emb.npy")
        question_cog_emb_dataset_path = os.path.join(file_path, "question_raw_emb.npy")
    
    
    student_cog_emb = torch.from_numpy(np.load(student_cog_emb_dataset_path))
    questiom_cog_emb = torch.from_numpy(np.load(question_cog_emb_dataset_path))
    student_num = student_cog_emb.shape[0]
    assert questiom_cog_emb.shape[0] == question_num

    student_sim_dataset_path = os.path.join(file_path, "student_sim.npy")
    question_sim_dataset_path = os.path.join(file_path, "question_sim.npy")
    student_sim = torch.from_numpy(np.load(student_sim_dataset_path))
    question_sim = torch.from_numpy(np.load(question_sim_dataset_path))


    train_student_list = []
    train_question_list = []
    train_answer_list = []
    test_student_list = []
    test_question_list = []
    test_answer_list = []

    student_all = [0] * student_num
    student_true = [0] * student_num
    question_all = [0] * question_num
    question_true = [0] * question_num
    
    """遍历-统计"""
    
    for index, row in train_df.iterrows():
        student_id = row["student_id"]
        question_id = row["question_id"]
        score = row["correct"]
        
        train_student_list.append(student_id)
        train_question_list.append(question_id)
        train_answer_list.append(score)

        student_all[int(student_id)] += 1.0
        question_all[int(question_id)] += 1.0

        if score >= 1.0:
            student_true[int(student_id)] += 1.0
            question_true[int(question_id)] += 1.0


    for index, row in test_df.iterrows():
        student_id = row["student_id"]
        question_id = row["question_id"]
        score = row["correct"]

        student_all[int(student_id)] += 1.0
        question_all[int(question_id)] += 1.0

        if score >= 1.0:
            student_true[int(student_id)] += 1.0
            question_true[int(question_id)] += 1.0
        
        test_student_list.append(student_id)
        test_question_list.append(question_id)
        test_answer_list.append(score)


    train_dataset = CDDataset(train_student_list, train_question_list, train_answer_list)
    test_dataset = CDDataset(test_student_list, test_question_list, test_answer_list)

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)

    # 统计一下正确率
    student_all = np.array(student_all)
    student_true = np.array(student_true)
    question_all = np.array(question_all)
    question_true = np.array(question_true)
    student_count = student_true / student_all
    question_count = question_true / question_all

    return student_num, question_num, concept_num, \
        train_loader, test_loader, \
            Q, student_cog_emb, questiom_cog_emb, student_sim, question_sim, student_count, question_count
