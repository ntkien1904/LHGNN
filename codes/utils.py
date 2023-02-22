import argparse
import math, scipy.stats as st
from matplotlib import axis 
import numpy as np

import random
import numpy as np
import torch


import networkx as nx

from collections import Counter
from ordered_set import OrderedSet
from collections import defaultdict as ddict

import os, logging, tqdm
from sklearn.cluster import KMeans
import datetime


neg_num_samp = 20


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='fb15k-237', help='dataset')
    parser.add_argument('--model', type=str, default='', help='model')

    parser.add_argument('--ds_path', type=str, default='../../data')
    parser.add_argument('--save_path', type=str, default='saved_model')
    parser.add_argument('--custom', type=str, default='')
    parser.add_argument('--ntype',  type=int, default=3, help='num pseudo type')

    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--drop', default=0.5, type=float)
    parser.add_argument('--decay', default=1e-04, type=float, help='weight decay')
    parser.add_argument('--gamma', default=0.2, type=float)
    parser.add_argument('--w_film', default=1e-4, type=float)  
    parser.add_argument('--lamda', default=0.1, type=float) 
   

    parser.add_argument('--max_l', type=int, default=1, help='max path length')
    parser.add_argument('--tpn', type=int, default=50, help='times per node')

    parser.add_argument('--use_emb', action='store_true', help='learn embed or use features')
    parser.add_argument('--embed', type=int, default=200, help='init input size')
    parser.add_argument('--t_embed', type=int, default=10, help='init input size')
    parser.add_argument('--hidden', type=int, default=32, help='hidden size')

    parser.add_argument('--n_runs', type=int, default=5, help='batch size')
    parser.add_argument('--batch', type=int, default=1024, help='batch size')
    parser.add_argument('--t_batch', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=300, help='num of iteration')
    parser.add_argument('--patience', type=int, default=500, help='early stopping')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--cuda', type=bool, default=True, help='use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')

    parser.add_argument('--no_type', action='store_true', help='use type embedding or not')
    parser.add_argument('--no_film', action='store_true', help='use op film')
    parser.add_argument('--op_rnn', type=bool, default=True, help='RNN aggre')
    parser.add_argument('--op_mean', action='store_true', help='mean aggre')


    return parser.parse_args() 




def mean_trials(out_list, name='', log_file=None):
    ci = 1.96 * st.sem(out_list) / math.sqrt(len(out_list))
    log = name + ' Mean: {:.3f} Std: {:.3f}' \
            .format(np.mean(out_list), st.sem(out_list)) 
    print(log)
    return log



def pagerank(graph):
    pr = nx.pagerank(nx.DiGraph(graph))
    return pr


def negative_sampling(graph, relation):
    sub, obj = relation
    
    pos = [sub, obj]
    pos.extend(list(graph[sub]))
    pos.extend(list(graph[obj]))

    pool = set(graph.keys())
    pool = pool - set(pos)

    neg = np.random.choice(list(pool), neg_num_samp, replace=False)

    triple = [sub, obj]
    triple.extend(list(neg))
    return triple


def load_paths(path, num_ent, max_l, tpn, use_ratio=False):
    biased = False    
    num_paths = num_ent * tpn

    paths_idx = np.zeros((num_paths, max_l),dtype=np.int)
    mask_idx = np.zeros((num_paths, max_l),dtype=np.int)

    ends = np.zeros((num_paths,2), dtype=int)


    if biased:
        raw_paths = np.genfromtxt(path, delimiter=' ')

        # more biased to shorter paths
        for i in range(num_ent):
            paths = raw_paths[i*tpn:i*tpn+tpn]
            count = 0
            
            flag = False
            for p in paths:
                l_path = int(np.random.choice(np.arange(2,max_l+1), 1))

                for j in range(2,l_path+1):
                    paths_idx[i*tpn+count,:j] = p[:j]
                    mask_idx[i*tpn+count,:j] = 1 
                    
                    ends[i*tpn+count, 0] = p[j-1]
                    ends[i*tpn+count, 1] = j-1
                    count += 1

                    if count >= tpn:
                        flag = True
                        break
                    
                if flag:
                    break
    
    else:
        # random or set ratio:
        if use_ratio: 
            if max_l == 3:
                ratio = [0.8, 0.2]
            elif max_l == 4:
                ratio = [0.6, 0.2, 0.2]
            else:
                ratio = [1]
        else:
            ratio = None
        
        l_path = np.random.choice(np.arange(2,max_l+1), num_paths, p=ratio)

        for i in range(l_path.shape[0]):
            mask_idx[i,:l_path[i]] = 1

        with open(path, 'r') as f:
            lines = f.readlines()
            row_idx = 0
            for line in lines:
                temp = list(line.strip('\n').split(' '))
                col_idx = 0
                for n in temp:
                    if mask_idx[row_idx, col_idx] == 0:
                        continue
                    paths_idx[row_idx, col_idx] = int(n)
                    col_idx += 1

                ends[row_idx, 0] = int(n)
                ends[row_idx, 1] = col_idx
                row_idx += 1

    return paths_idx, mask_idx, ends


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def path_sampling(graph, max_l, tpn, batch_size=100, path=None):
    random.seed(0)

    # sample neighbor set of each node based on random walks
    batch_num = math.ceil(len(graph.keys()) / batch_size)
    all_node = np.asarray(list(graph.keys()))
    #print(all_node.shape)

    file_walk = 'walks_{:d}_{:d}.txt'.format(tpn, max_l)

    if not os.path.exists(os.path.join(path, file_walk)):
        with open(os.path.join(path, file_walk), 'w') as output:
            
            for b in tqdm.tqdm(range(batch_num), 'random walk sample batch num'):
                ids_arr = all_node[b * batch_size: (b + 1) * batch_size]
                walks = np.repeat(ids_arr, tpn)[None, :]

                for i in range(1, max_l):
                    new_row = np.array([random.choice(graph[j]) for j in walks[i - 1]])[None, :]
                    walks = np.concatenate((walks, new_row), axis=0)

                walks = np.transpose(walks)
                for walk in walks:                
                    tmp = " ".join([str(j) for j in walk])
                    output.write(tmp + '\n')
                
    return file_walk




def get_neigh_matrix(graph, max_degree=50):
    
    max_nei = np.max([len(v) for k,v in graph.items()])

    if max_nei < max_degree:
        max_degree = max_nei

    nei_mat = np.zeros((len(graph.keys()), max_degree), dtype=np.int32)
    mask_mat = np.zeros((len(graph.keys()), max_degree), dtype=np.int32)

    for k in graph.keys():

        if len(graph[k]) > max_degree:
            n = np.random.choice(graph[k], max_degree, replace=False)
            nei_mat[k] = n
            mask_mat[k] = 1
        else:
            nei_mat[k,:len(graph[k])] = graph[k]
            mask_mat[k,:len(graph[k])] = 1
            
    return nei_mat, mask_mat



def build_graph(path, num_ent, links, max_l, tpn):
    # graph with all train edges
    ori_graph = ddict(OrderedSet)
    for i in range(num_ent):
        ori_graph[i] = OrderedSet()

    ent_train = set()
    for sub, obj in links:
        ori_graph[sub].add(obj)
        ori_graph[obj].add(sub)
        
        ent_train.add(sub)
        ent_train.add(obj)
        

    isolate_nodes = list(set(range(num_ent)) - ent_train)
    for n in isolate_nodes:
        ori_graph[n].add(n)


    file_walk = path_sampling(ori_graph, max_l, tpn, path=path)
    paths_idx, mask_idx, ends = load_paths(os.path.join(path, file_walk), num_ent, max_l, tpn)

    
    new_graph = ddict(OrderedSet)
    for i in range(num_ent):
        new_graph[i] = OrderedSet()


    for i in range(mask_idx.shape[0]):
        for j in range(1, max_l):
            if mask_idx[i, j] != 0:
                new_graph[paths_idx[i,0]].add(paths_idx[i,j])
                new_graph[paths_idx[i,j]].add(paths_idx[i,0])


    paths_idx = np.reshape(paths_idx,(-1, tpn, max_l))
    mask_idx = np.reshape(mask_idx,(-1, tpn, max_l))
    ends =  np.reshape(ends,(-1, tpn, 2))

    return new_graph, ori_graph, paths_idx, mask_idx, ends


# for link prediction
def create_train_test_split(data, graph, path, num=0, name=''):
    np.random.seed(0)
    count = 0
    for split in ['train', 'test', 'valid']:
        name_file = os.path.join(path, split + '_triples_{:d}.txt'.format(neg_num_samp))

        if name=='ogb':
            # for fast sampling
            edges = data[split]
            samples = np.random.randint(num, size=(edges.shape[0],neg_num_samp))
            triples = np.concatenate((edges, samples), axis=1)

            with open(name_file,'w')  as f:
                for t in triples:
                    tmp = " ".join([str(j) for j in t])
                    f.write(tmp + '\n')
        else:
            with open(name_file,'w')  as f:
                for relation in data[split]:
                    triple = negative_sampling(graph, relation)

                    tmp = " ".join([str(j) for j in triple])
                    f.write(tmp + '\n')
                    count += 1
                    if count % 10000 == 0:
                        print(count)
    return


def Kmeans_pseudo_type(feat, ntype=3):
    
    kmeans = KMeans(n_clusters=ntype, random_state=0).fit(feat)
    return kmeans.labels_



def prepare_saved_path(args):

    # dataset folder
    save_path = os.path.join(args.save_path, args.dataset)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # model folder index
    now = datetime.datetime.now()

    save_folder =  '_'.join([str(now.day), str(now.month), str(now.strftime("%H:%M:%S"))])
    save_path = os.path.join(save_path, save_folder)
    
    os.mkdir(save_path)
    with open(os.path.join(save_path, 'config.txt'), 'w') as f:
        for k,v in vars(args).items():
            f.write(str(k) + ': ' + str(v) + '\n')

    return save_path


def get_struc_info(num_nodes, graph, pagerank):

    deg, nei_deg, pr, nei_pr = [list(range(num_nodes))] * 4

    for i in range(num_nodes):
        deg[i] = len(graph[i])
        pr[i] = pagerank[i]

        l1_nei_deg = 0
        l1_nei_pr = 0

        for j in graph[i]:
            l1_nei_deg += len(graph[j])
            l1_nei_pr += pagerank[j]

        nei_deg[i] = l1_nei_deg / len(graph[i]) + 1
        nei_pr[i] = l1_nei_pr / len(graph[i]) + 1

    deg = np.log(deg)
    nei_deg = np.log(nei_deg)
    struc_feat = np.column_stack((deg, nei_deg, pr, nei_pr))
    return struc_feat

