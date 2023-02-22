from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
from os import path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from torch.utils import data
from torch.utils.data import DataLoader
import time



def get_param(shape):
	param = nn.Parameter(torch.Tensor(*shape)); 	
	nn.init.xavier_normal_(param.data)
	return param




class BaseLayer(nn.Module):
    def __init__(self, args, in_feat, out_feat):
        super(BaseLayer, self).__init__()

        self.lamda = args.lamda
        self.w = nn.Linear(in_feat, out_feat)
        self.w_type = nn.Linear(in_feat, args.t_embed)

        self.gamma = nn.Linear(args.t_embed, in_feat) 
        self.beta = nn.Linear(args.t_embed, in_feat)

       
        self.no_film = args.no_film


    def forward(self, embed, idx, paths, masks, neighs, t_tilde=None, ntype=None):
        
      
        t_x = self.w_type(embed) # [batch, t_embed size]
        
        # paths embedding 
        t_p = torch.sum(t_x[paths] * masks.unsqueeze(dim=-1), dim=-2)
        t_p /= torch.sum(masks, dim=2, keepdim=True)
   

        feat = embed[idx] 
        h_l = embed[neighs[:,:,0]]

        # path embed of neighbor node:
        if self.no_film:
            p_x = h_l
        else:
            gamma = F.leaky_relu(self.gamma(t_p))
            beta = F.leaky_relu(self.beta(t_p))
            p_x = (gamma + 1) * h_l + beta
        

        # message
        l_p = torch.unsqueeze(neighs[:,:,1],2) 
        alpha = torch.exp(-self.lamda*l_p)

        a_x = torch.sum(alpha * p_x, dim=1) 

        # final embed 
        update = (feat + a_x) / (neighs.shape[1]+1) 
        output = F.leaky_relu(self.w(update))

        L_film = torch.sum(torch.norm(gamma, dim=1)) + torch.sum(torch.norm(beta, dim=1))
        L_film /= gamma.shape[0]

        return output, 0, 0, [t_x[idx], L_film, 0]



class BaseModel(nn.Module):
    def __init__(self, data, args):
        super(BaseModel, self).__init__()

        self.use_emb = args.use_emb
        self.no_type = args.no_type
        self.op_rnn = args.op_rnn
        self.op_mean = args.op_mean

        self.reduce = True
        if self.use_emb:
            self.embed = get_param((data.num_ent, args.embed))           
            in_feat = args.embed
        else:
            if data.feat.shape[1] > 1000:
                self.fc = nn.Linear(data.feat.shape[1], args.embed)
                in_feat = args.embed
            else:
                self.reduce = False
                in_feat = data.feat.shape[1]
            

        self.layer1 = BaseLayer(args, in_feat, args.hidden)
        self.layer2 = BaseLayer(args, args.hidden, args.hidden)


        self.num = data.num_ent
        self.drop = args.drop

       
        if self.op_rnn:
            self.rnn = nn.RNN(args.t_embed, args.hidden, batch_first=True)
        elif self.op_mean:
            self.mlp = nn.Linear(args.t_embed, args.hidden)

        self.gamma = torch.cuda.FloatTensor([args.gamma]) 


    def forward(self, b_data, x=None):
       
        if self.use_emb:
            x = self.embed
        else:
            if self.reduce:
                x = self.fc(x)

        x1, l1_t1, l1_t2, sup1 = \
             self.layer1(x, b_data['l1'], b_data['paths_l1'],\
                        b_data['mask_l1'] ,b_data['end_l1'])
        t1, film1, e1 = sup1

        x1 = F.normalize(x1)
        x1 = F.dropout(x1, self.drop, training=self.training)

        x2, l2_t1, l2_t2, sup2 = \
            self.layer2(x1, b_data['l2'], b_data['paths_l2'], \
                        b_data['mask_l2'], b_data['end_l2']) 
        x2 = F.normalize(x2)
        t2, film2, e2 = sup2
        
        return  x2, t2, film1+film2  
    

    
    def new_loss(self, output, type, idx):

        a = output[idx[:,0]]            
        b = output[idx[:,1]]
        c = output[idx[:,2]]
        
        if self.no_type:
            pos = torch.norm(a - b, p=2, dim=1)
            neg = torch.norm(a - c, p=2, dim=1)
        else:

            if self.op_mean:
                t_a = type[idx[:,0]]
                t_b = type[idx[:,1]]
                t_c = type[idx[:,2]]

                t_a = F.leaky_relu(self.mlp(t_a))
                t_b = F.leaky_relu(self.mlp(t_b))
                t_c = F.leaky_relu(self.mlp(t_c))

                t_ab = (t_a + t_b)/2 
                t_ac = (t_a + t_c)/2 

            elif self.op_rnn: 
                t_a = torch.unsqueeze(type[idx[:,0]], dim=1)
                t_b = torch.unsqueeze(type[idx[:,1]], dim=1)
                t_c = torch.unsqueeze(type[idx[:,2]], dim=1)

                t_ab = torch.cat((t_a, t_b), dim=1)
                t_ac = torch.cat((t_a, t_c), dim=1)
                
                t_ab, _ = self.rnn(t_ab)
                t_ac, _ = self.rnn(t_ac)

                t_ab = t_ab[:,-1,:]
                t_ac = t_ac[:,-1,:]

            pos = torch.norm(a + t_ab - b, p=2, dim=1)
            neg = torch.norm(a + t_ac - c, p=2, dim=1)

        loss = torch.max((pos - neg), -self.gamma).mean() + self.gamma
        return loss 

    