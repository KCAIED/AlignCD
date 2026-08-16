import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from layers import PosLinear


class LCD(nn.Module):

    def __init__(self, Q, stu_emb, ques_emb, hidden_dim, concept_num,
                 bias=True, has_cuda=False, device='cpu', base="ncd", use_llm=1):
        super(LCD, self).__init__()
        
        self.student_num = stu_emb.shape[0]
        self.question_num = ques_emb.shape[0]
        self.concept_num = concept_num    # 注意是目标数据知识体系
        
        self.llm_dim = stu_emb.shape[1]
        
        self.hidden_dim = hidden_dim
        self.hidden_dim_1 = hidden_dim
        self.hidden_dim_2 = hidden_dim // 2

        self.bias = bias 

        self.has_cuda = has_cuda
        self.device = device
        self.base = base
        self.use_llm = use_llm

        self.register_buffer("Q", Q.float())    # 注意是目标数据知识体系
        self.register_buffer("stu_emb", stu_emb.float())
        self.register_buffer("ques_emb", ques_emb.float())

        # 特殊情况下才进行使用
        self.student = nn.Embedding(self.student_num, self.hidden_dim)
        self.question = nn.Embedding(self.question_num, self.hidden_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
         
        # 以下是独属于特定数据集的
        self.concept = nn.Embedding(self.concept_num, self.hidden_dim)
        # 这两项作为消融实验留伏笔
        self.student_map = nn.Linear(self.llm_dim, self.hidden_dim, bias=self.bias)
        self.question_map = nn.Linear(self.llm_dim, self.hidden_dim, bias=self.bias)

        self.q2c = nn.Linear(self.hidden_dim*2, 1, bias=self.bias)    # 获取临时Q矩阵
        
        if self.base == "mirt":
            self.disc = nn.Linear(self.hidden_dim, 1, bias=self.bias)
        
        elif self.base == "ncd":
            self.mas = nn.Linear(self.hidden_dim, self.concept_num, bias=self.bias)
            self.diff = nn.Linear(self.hidden_dim, self.concept_num, bias=self.bias)
            self.disc = nn.Linear(self.hidden_dim, 1, bias=self.bias)
            self.decoder = nn.Sequential(
                PosLinear(self.concept_num, self.hidden_dim_1, bias=self.bias),
                nn.Sigmoid(),
                nn.Dropout(p=0.5),
                PosLinear(self.hidden_dim_1, self.hidden_dim_2, bias=self.bias),
                nn.Sigmoid(),
                nn.Dropout(p=0.5),
                PosLinear(self.hidden_dim_2, 1, bias=self.bias),
                nn.Sigmoid()
            )
        
        elif self.base == "kscd":
            self.sc = nn.Linear(self.hidden_dim*2, 1, bias=self.bias)
            self.ec = nn.Linear(self.hidden_dim*2, 1, bias=self.bias)
            self.se = nn.Linear(self.concept_num, self.concept_num, bias=self.bias)
        
        elif self.base == "kancd":
            self.sc = nn.Linear(self.hidden_dim, 1, bias=self.bias)
            self.ec = nn.Linear(self.hidden_dim, 1, bias=self.bias)
            self.disc = nn.Linear(self.hidden_dim, 1, bias=self.bias)
            self.decoder = nn.Sequential(
                PosLinear(self.concept_num, self.hidden_dim_1, bias=self.bias),
                nn.Sigmoid(),
                nn.Dropout(p=0.5),
                PosLinear(self.hidden_dim_1, self.hidden_dim_2, bias=self.bias),
                nn.Sigmoid(),
                nn.Dropout(p=0.5),
                PosLinear(self.hidden_dim_2, 1, bias=self.bias),
                nn.Sigmoid()
            )

        
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) < 2:
                    nn.init.xavier_normal_(param.unsqueeze(0))
                else:
                    nn.init.xavier_normal_(param)


    def forward(self, students, questions):
        r"""
        Parameters:
            students: student index matrix
            questions: question index matrix
            answers: answer matrix
        Shape:
            students: [batch_size]
            questions: [batch_size]
            pred_res: [batch_size]
            hs_mas, hq_mas: [student_num/question_num, concept_num/hidden_dim]
        Return:
            pred_res: the correct probability of questions answered
        """
        batch_size = questions.shape[0]
        self.Q.to(device=questions.device)
        self.stu_emb.to(device=questions.device)
        self.ques_emb.to(device=questions.device)

        hc = self.concept.weight    # [concept_num, hidden_dim]
        
        # [batch_size, hidden_dim]
        if self.use_llm==1:
            hs = self.student_map(self.stu_emb[students])    
            hq = self.question_map(self.ques_emb[questions])

            hs_state = self.student_map(self.stu_emb)   # [student_num, hidden_dim]
            hq_state = self.question_map(self.ques_emb)    # [question_num, hidden_dim]
        
        else:
            hs = self.student(students)
            hq = self.question(questions)

            hs_state = self.student.weight   # [student_num, hidden_dim]
            hq_state = self.question.weight   # [question_num, hidden_dim]

        
        # [batch_size, concept_num, hidden_dim*2]
        temp_Q = torch.cat([hq[:, None, :].expand(-1, self.concept_num, -1), 
                            hc[None, :, :].expand(batch_size, -1, -1)], dim=2)
        temp_Q = self.sigmoid(self.q2c(temp_Q))[:,:,0]    # [batch_size, concept_num]


        if self.base == "mirt":
            beta = self.disc(hq)
            pred_res = (hs * hq).sum(dim=1, keepdim=False) + beta[:,0]
            pred_res = self.sigmoid(pred_res)
            hs_mas = None
            # print(pred_res.shape)

        elif self.base == "ncd":
            hs_state = self.sigmoid(self.mas(hs_state))    # [student_num, concept_num]
            hq_state = self.sigmoid(self.diff(hq_state))   # [question_num, concept_num]
            
            hs_mas = self.sigmoid(self.mas(hs))                  # [batch_size, concept_num]
            hq_diff = self.sigmoid(self.diff(hq))    # [batch_size, concept_num]
            hq_disc = self.sigmoid(self.disc(hq))    # [batch_size, 1]
            hq_disc = hq_disc.repeat(1, self.concept_num)    # [batch_size, concept_num]
            # x = self.Q[questions] * (hs_mas - hq_diff) * hq_disc * 10
            x = temp_Q * (hs_mas - hq_diff) * hq_disc * 10
            x = self.decoder(x)
            pred_res = x[:,0]
        
        elif self.base == "kscd":
            # test_s = hs[:, None, :].expand(-1, self.concept_num, -1)
            # test_c = hc[None, :, :].expand(batch_size, -1, -1)
            # print(test_s.shape)
            # print(test_c.shape)
            hs_state = torch.cat([hs_state[:, None, :].expand(-1, self.concept_num, -1), 
                                  hc[None, :, :].expand(self.student_num, -1, -1)], dim=2)
            hs_state = self.sc(hs_state)[:,:,0]    # [student_num, concept_num]
            
            hq_state = torch.cat([hq_state[:, None, :].expand(-1, self.concept_num, -1), 
                                hc[None, :, :].expand(self.question_num, -1, -1)], dim=2)
            hq_state = self.ec(hq_state)[:,:,0]    # [question_num, concept_num]

            # 正式
            hs_mas = torch.cat([hs[:, None, :].expand(-1, self.concept_num, -1), 
                                hc[None, :, :].expand(batch_size, -1, -1)], dim=2)
            hs_mas = self.sc(hs_mas)[:,:,0]    # [batch_size, concept_num]

            hq_mas = torch.cat([hq[:, None, :].expand(-1, self.concept_num, -1), 
                                hc[None, :, :].expand(batch_size, -1, -1)], dim=2)
            hq_mas = self.ec(hq_mas)[:,:,0]    # [batch_size, concept_num]

            # x = self.se(hs_mas-hq_mas) * self.Q[questions]
            # pred_res = self.sigmoid(torch.sum(x, dim=1) / torch.sum(self.Q[questions], dim=1))
            x = self.se(hs_mas-hq_mas) * temp_Q
            pred_res = self.sigmoid(torch.sum(x, dim=1) / torch.sum(temp_Q, dim=1))
        

        elif self.base == "kancd":
            hs_state = hs_state[:, None, :].expand(-1, self.concept_num, -1)
            hc_student_extra = hc[None, :, :].expand(self.student_num, -1, -1)
            hq_state = hq_state[:, None, :].expand(-1, self.concept_num, -1)
            hc_question_extra = hc[None, :, :].expand(self.question_num, -1, -1)
            
            hs_state = self.sigmoid(self.sc(hs_state*hc_student_extra))[:,:,0]
            hq_state = self.sigmoid(self.ec(hq_state*hc_question_extra))[:,:,0]
            
            h_disc = self.sigmoid(self.disc(hq))    # [batch_size, 1]
            h_disc = h_disc.repeat(1, self.concept_num)    # [batch_size, concept_num]

            # [batch_size, concept_num, hidden_dim]
            hs_extra = hs[:, None, :].expand(-1, self.concept_num, -1)
            hq_extra = hq[:, None, :].expand(-1, self.concept_num, -1)
            hc_extra = hc[None, :, :].expand(batch_size, -1, -1)

            hs_mas = self.sigmoid(self.sc(hs_extra*hc_extra))[:,:,0]    # [batch_size, concept_num]
            h_diff = self.sigmoid(self.ec(hq_extra*hc_extra))[:,:,0]    # [batch_size, concept_num]
            # x = self.Q[questions] * (hs_mas - h_diff) * h_disc * 10
            x = temp_Q * (hs_mas - h_diff) * h_disc * 10
            x = self.decoder(x)
            pred_res = x[:,0]
        
        
        return pred_res, hs_state, hq_state, hs_mas, self.Q[questions], temp_Q
