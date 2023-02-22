from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from posixpath import split

import numpy as np
from numpy import dtype
from numpy.lib.function_base import copy
import torch
from torch.utils import data
from torch.utils.data import Dataset
from ordered_set import OrderedSet
import random
import time


    
def create_batches(num_ent, batch_size=64, shuffle=True):

    idx_list = np.arange(num_ent, dtype='int32')

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    num_batch = idx_list.shape[0] // batch_size

    cur_idx = 0
    for i in range(num_batch): 
        minibatches.append(idx_list[cur_idx:cur_idx + batch_size])
        cur_idx += batch_size

    if (cur_idx != num_ent):
        minibatches.append(idx_list[cur_idx:])

    return minibatches



def prepare_batch(data, idx, split='train'):

    #t = time.time()

    if split == 'post':
        edges = idx

    else:
        full = data.triples[split]
        # edges of batch
        edges = full[list(idx)]

    #layer 1 and 2    
    nodes_l1 = OrderedSet()
    nodes_l2 = OrderedSet()

    for edge in edges:
        for i in edge:
            nodes_l2.add(i)
        
    for node in nodes_l2:
        nodes_l1.add(node)
        for i in data.graph[node]:
            nodes_l1.add(i)

    
    # paths for layer 1 and 2
    paths_l1 = data.path_idx[nodes_l1]
    paths_l2 = data.path_idx[nodes_l2]

    mask_l1 = data.mask_idx[nodes_l1]
    mask_l2 = data.mask_idx[nodes_l2]
    
    end_l1 = data.ends[nodes_l1]
    end_l2 = data.ends[nodes_l2]
    

    
    # reindex for layer 1 regarding to layer 2 new index    
    l1_to_l2 = dict()
    idx = 0
    for i in nodes_l1:
        l1_to_l2[i] = idx
        idx += 1

    new_nodes_l2 = np.copy(nodes_l2)
    for i in range(len(nodes_l2)):
        new_nodes_l2[i] = l1_to_l2[nodes_l2[i]]

   
    for i in range(paths_l2.shape[0]):
        for j in range(paths_l2.shape[1]):
            for k in range(paths_l2.shape[2]):        
                if mask_l2[i,j,k] == 0:
                    continue

                paths_l2[i,j,k] = l1_to_l2[paths_l2[i,j,k]]
                end_l2[i,j,0] = paths_l2[i,j,k]
    


    # re-index batch
    l2_to_l0 = dict()
    idx = 0
    for i in new_nodes_l2:
        l2_to_l0[i] = idx
        idx += 1


    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            edges[i,j] = l2_to_l0[l1_to_l2[edges[i,j]]]

    
    return edges, { 'l2': new_nodes_l2, 
                    'l1': nodes_l1, 
                    'paths_l2': paths_l2,
                    'paths_l1': paths_l1,
                    'end_l2': end_l2,
                    'end_l1': end_l1,
                    'mask_l2': mask_l2,
                    'mask_l1': mask_l1,
                    't0': 0}, l1_to_l2

 


def prepare_batch_gcn(data, split, idx):
    full = data.triples[split]
   
    # edges of batch
    edges = full[list(idx)]

    #layer 1 and 2    
    nodes_l1 = OrderedSet()
    nodes_l2 = OrderedSet()


    for edge in edges:
        for i in edge:
            nodes_l2.add(int(i))
    end_l2 = data.nei_mat[nodes_l2]
    mask_l2 = data.mask_mat[nodes_l2]

    for node in nodes_l2:
        nodes_l1.add(node)
        for i in range(data.nei_mat[node].shape[0]):
            nodes_l1.add(data.nei_mat[node,i])
 
    end_l1 = data.nei_mat[nodes_l1]
    mask_l1 = data.mask_mat[nodes_l1]

  
    # reindex for layer 1 regarding to layer 2 new index    
    l1_to_l2 = dict()
    idx = 0
    for i in nodes_l1:
        l1_to_l2[i] = idx
        idx += 1
    

    new_nodes_l2 = np.copy(nodes_l2)
    for i in range(len(nodes_l2)):
        new_nodes_l2[i] = l1_to_l2[nodes_l2[i]]

    for i in range(end_l2.shape[0]):
        for j in range(end_l2.shape[1]):
            end_l2[i,j] = l1_to_l2[end_l2[i,j]]


    # re-index batch
    l2_to_l0 = dict()
    idx = 0
    for i in new_nodes_l2:
        l2_to_l0[i] = idx
        idx += 1
    


    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            edges[i,j] = l2_to_l0[l1_to_l2[edges[i,j]]]

    return edges,  {'l2': new_nodes_l2, 
                    'l1': nodes_l1,
                    'end_l2': end_l2,
                    'end_l1': end_l1,
                    'mask_l2': mask_l2,
                    'mask_l1': mask_l1,
                    'paths_l2': 0,
                    'paths_l1': 0,
                    't0': 0}, l1_to_l2
