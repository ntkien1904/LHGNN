from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json, logging
import os, sys

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import optimizer
from dataloader import create_batches, prepare_batch, prepare_batch_gcn

import time
from model import BaseModel
from data_process import get_dataset

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import average_precision_score, ndcg_score

from utils import *

torch.set_printoptions(profile='full')
np.set_printoptions(precision=4, threshold=sys.maxsize)




#--------------------------------------------------------------------------------------------------------
def eval(model, data, args, mode='test', task=1):
    
    time_val = time.time()
    model.eval()  
    num_sample = data.triples[mode].shape[0]
    test_batches = create_batches(num_sample, batch_size=num_sample)

    res_loss = False
    
    if task == 1:
        neg_num=9
        labels = np.zeros(neg_num+1)
        labels[0] = 1

        pos_score = np.empty([0])
        neg_score = np.empty([0,9])

        with torch.no_grad():
            for batch in test_batches:
                #print('here')
                idx, data_batch, l12 = prepare_batch(data, batch, mode)
            
                for k in data_batch.keys():
                    if k == 't0':
                        data_batch[k] = torch.FloatTensor(data_batch[k])
                    else:
                        data_batch[k] = torch.LongTensor(data_batch[k])
                    
                if args.cuda:
                    for k in data_batch.keys():
                        data_batch[k] = data_batch[k].cuda()
                
                if args.use_emb:
                    output, type, _ = model(data_batch)    
                else:
                    output, type, _ = model(data_batch, x=data.feat)

                a = output[idx[:,0]]            
                b = output[idx[:,1]]
                c = output[idx[:,2:11]]  

                if res_loss:
                    pos = torch.sigmoid(torch.sum(torch.mul(a,b), dim=1))
                    neg = torch.sigmoid(torch.sum(torch.mul(a.view(a.shape[0],1,a.shape[1]),c), dim=2))
                
                elif args.no_type:
                    pos = - torch.norm(a - b, p=2,dim=1)
                    
                    neg = torch.empty([c.shape[0],0], device='cuda')
                    for i in range(c.shape[1]):
                        n_score = - torch.norm(a - c[:,i,:], p=2, dim=1)
                        neg = torch.cat((neg, torch.unsqueeze(n_score,1)), dim=1)

                else:
                    if model.op_rnn: 
                        t_a = torch.unsqueeze(type[idx[:,0]], dim=1)
                        t_b = torch.unsqueeze(type[idx[:,1]], dim=1)
                        t_c = torch.unsqueeze(type[idx[:,2:11]], dim=1)

                        t_ab = torch.cat((t_a, t_b), dim=1)
                        t_ab, _ = model.rnn(t_ab)
                        t_ab = t_ab[:,-1,:]

                    elif model.op_mean:
                        t_a = type[idx[:,0]]
                        t_b = type[idx[:,1]]
                        t_c = type[idx[:,2:11]]

                        t_a = F.leaky_relu(model.mlp(t_a))
                        t_b = F.leaky_relu(model.mlp(t_b))
                        t_c = F.leaky_relu(model.mlp(t_c))

                        t_ab = (t_a + t_b)/2 

                    pos = - torch.norm(a + t_ab - b, p=2,dim=1)


                    neg = torch.empty([c.shape[0],0], device='cuda')
                    for i in range(c.shape[1]):
                        c_i = t_c[:,:,i]
                        t_ac = torch.cat((t_a, c_i), dim=1)
                        t_ac, _ = model.rnn(t_ac)
                        t_ac = t_ac[:,-1,:]
                        
                        n_score = - torch.norm(a + t_ac - c[:,i,:], p=2, dim=1)
                        neg = torch.cat((neg, torch.unsqueeze(n_score,1)), dim=1)

                pos_score = np.concatenate((pos_score, pos.cpu().detach().numpy()))
                neg_score = np.concatenate((neg_score, neg.cpu().detach().numpy()))

        pred_list = np.concatenate((np.expand_dims(pos_score, axis=1), neg_score), axis=1)

        sum_ndcg = 0
        sum_mrr = 0
        sum_hit1 = 0

        for i in range(num_sample):    
            true = pred_list[i, 0]
            sort_list = np.sort(pred_list[i])[::-1]

            rank = int(np.where(sort_list == true)[0][0]) + 1
            sum_mrr += (1/rank)

            if mode == 'test':
                if pred_list[i, 0] == np.max(pred_list[i]):
                    sum_hit1 += 1

                NDCG = ndcg_score([labels], [pred_list[i]])
                sum_ndcg += NDCG
              
        H1 = sum_hit1 / num_sample
        MRR = sum_mrr / num_sample
        NDCG = sum_ndcg / num_sample


        if mode == 'test':
            log =   "{:s} MAP/MRR={:.3f}, NDCG={:.3f}, H1={:.3f}, Time: {:.2f}" \
                .format(mode, MRR, NDCG , H1, time.time()-time_val) 
        else:
            log =   "{:s} MAP/MRR={:.3f}, Time: {:.2f}" \
                .format(mode, MRR, time.time()-time_val) 

        print(log)        
        return (MRR, NDCG, H1)
    
    
    else:  
        pos_score = np.empty([0])
        neg_score = np.empty([0])

        with torch.no_grad():
            for batch in test_batches:
                idx, data_batch, l12 = prepare_batch(data, batch, mode)

                for k in data_batch.keys():
                    if k == 't0':
                        data_batch[k] = torch.FloatTensor(data_batch[k])
                    else:
                        data_batch[k] = torch.LongTensor(data_batch[k])
                    
                if args.cuda:
                    for k in data_batch.keys():
                        data_batch[k] = data_batch[k].cuda()
    
                if args.use_emb:
                    output, type, _ = model(data_batch)    
                else:
                    output, type, _ = model(data_batch, x=data.feat)    
                
                a = output[idx[:,0]]            
                b = output[idx[:,1]]
                c = output[idx[:,2]]
            
                if res_loss:
                    pos = torch.sum(torch.mul(a,b), dim=1)
                    neg = torch.sum(torch.mul(a,c), dim=1)   

                elif args.no_type:
                    pos = args.gamma - torch.norm(a - b, p=2,dim=1)
                    neg = args.gamma - torch.norm(a - c, p=2,dim=1)
                    
                else:
                    if model.op_rnn: 
                        t_a = torch.unsqueeze(type[idx[:,0]], dim=1)
                        t_b = torch.unsqueeze(type[idx[:,1]], dim=1)
                        t_c = torch.unsqueeze(type[idx[:,2]], dim=1)

                        t_ab = torch.cat((t_a, t_b), dim=1)
                        t_ab, _ = model.rnn(t_ab)
                        t_ab = t_ab[:,-1,:]

                        t_ac = torch.cat((t_a, t_c), dim=1)
                        t_ac, _ = model.rnn(t_ac)
                        t_ac = t_ac[:,-1,:]
                    
                    elif model.op_mean:
                        t_a = type[idx[:,0]]
                        t_b = type[idx[:,1]]
                        t_c = type[idx[:,2]]

                        t_ab = (t_a + t_b)/2 
                        t_ac = (t_a + t_c)/2 

                    pos = - torch.norm(a + t_ab - b, p=2,dim=1)
                    neg = - torch.norm(a + t_ac - c, p=2,dim=1)

                pos_score = np.concatenate((pos_score, pos.cpu().detach().numpy()))
                neg_score = np.concatenate((neg_score, neg.cpu().detach().numpy()))

        preds = np.where(pos_score >= neg_score, 1.0, 0.0)
        labels = np.ones(num_sample)            
        Acc = accuracy_score(labels, preds)   
        print('Accuracy : {:.4f}'.format(Acc))
        return Acc
       


#--------------------------------------------------------------------------------------------------------
def run(args, seed, data, path):

    print('Seed trial: ', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(seed)


    model_path = os.path.join(path, str(seed))


    model = BaseModel(data, args)
    optimizer = torch.optim.Adam(model.parameters(),\
                                 lr=args.lr, weight_decay=args.decay)

    if args.cuda:
        if not args.use_emb:
            data.feat = data.feat.cuda()
        #data.struc_feat = data.struc_feat.cuda()
        model = model.cuda()


    print('Training ...')
    best_val = 0
    best_epoch = 0
    best_batch = 0
    patience = 0
    batch_break = False

    for i in range(args.epochs):
        print('Epoch: {:d}'.format(i))
       
        minibatches = create_batches(data.triples['train'].shape[0], batch_size=args.batch)
        
        for j in range(len(minibatches)):
            t = time.time()

            # sample data batch for layer 1 and 2
            idx, data_batch, _ = prepare_batch(data, minibatches[j], 'train')
        
            for k in data_batch.keys():
                if k == 't0':
                    data_batch[k] = torch.FloatTensor(data_batch[k])
                else:
                    data_batch[k] = torch.LongTensor(data_batch[k])
            
            if args.cuda:
                for k in data_batch.keys():
                    data_batch[k] = data_batch[k].cuda()


            model.train()
            optimizer.zero_grad()

            
            if args.use_emb:
                output, typ, cons = model(data_batch)    
            else:
                output, typ, cons = model(data_batch, x=data.feat)    
                
            loss = model.new_loss(output, typ, idx)
            loss += args.w_film * cons 

            loss.backward()
            optimizer.step()
            
            
            if (j) % 20 == 0:
                print('Batch: {:d}, Loss: {:.6f}, Film: {:.4f}, Time: {:.5f}'\
                        .format(j+1, loss.item(), cons, time.time()-t))            
            

            if args.dataset == 'ogb':
                step = 50 if i < 1 else 10
            else:
                step = 10 

            # Validate
            if j % step == 0:
                res_val = eval(model, data, args, mode='valid')
               
                if res_val[0] >= best_val:
                    torch.save(model, model_path)
                    best_val = res_val[0]
                    best_epoch = i
                    best_batch = j
                    patience = 0
                else:
                    patience += 1 

                if patience > args.patience:
                    batch_break = True
                    print('Early stopping at epoch {:d}, batch {:d}, MRR={:.3f}'.format(best_epoch, best_batch, best_val))
                    break
            else:
                patience += 1 
            
        if batch_break:
            break


    model = torch.load(model_path)
    #Test 
    task1 = eval(model, data, args, mode='test', task=1)
    task2 = eval(model, data, args, mode='test', task=2)   
    return (task1, task2)


    

if __name__ == '__main__':
    
    args = parse_args()
    print(args)
    args.max_l += 1 # include the source node itself


    acc_list = []    
    ndcg_list = []
    mrr_list = []
    h1_list = []

    np.random.seed(args.seed)
    seed = np.random.choice(100, args.n_runs, replace=False)
    print('Seed: ', seed)

    print('Processing data ...')
    data = get_dataset(args, args.dataset)

    save_path = prepare_saved_path(args)

    for i in range(args.n_runs): 
        res1, res2 = run(args, seed[i], data, save_path)
        
        mrr_list.append(res1[0])
        ndcg_list.append(res1[1])
        h1_list.append(res1[2])
        acc_list.append(res2*100)
    
    m_map = mean_trials(mrr_list, name='MAP/MRR')
    m_ndcg = mean_trials(ndcg_list, name='NDCG')
    m_h1 = mean_trials(h1_list, name='H1')
    m_acc = mean_trials(acc_list, name='Accuracy')
    res = [m_map, m_ndcg, m_h1, m_acc]

    with open(os.path.join(save_path, 'config.txt'), 'a') as f:
        f.write(' '.join([str(i) for i in mrr_list]) + '\n')
        f.write(' '.join([str(i) for i in ndcg_list]) + '\n')
        f.write(' '.join([str(i) for i in h1_list]) + '\n')
        f.write(' '.join([str(i) for i in acc_list]) + '\n')

        for r in res:
            f.write(r + '\n')





