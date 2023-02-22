from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import pickle
import scipy.sparse as sp
from pprint import pprint

import math
import numpy as np
import torch_geometric
#import dgl

from torch_geometric.datasets import OGB_MAG
from utils import *
import networkx as nx


class KG(object):
    def __init__(self, args, path, ds):     

        args.use_emb = True
        self.max_l = args.max_l
        self.tpn = args.tpn
        self.path = os.path.join(path, ds)

        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train', 'test', 'valid']:
            for line in open(os.path.join(self.path, '{}.txt'.format(split))):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.num_ent		= len(self.ent2id)
        self.num_rel		= len(self.rel2id) // 2
        #self.embed_dim	    = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            for line in open(os.path.join(self.path, '{}.txt'.format(split))):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, obj))

                if split == 'train': 
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel+self.num_rel)].add(sub)
            
            self.data[split] = np.array(self.data[split])


        '''
        ori_graph: graph with edges from train edges
        graph: graph with edges from sampling process
        path_idx: index of all nodes of paths 
        ends: index of last node of paths
        '''

        self.graph, self.ori_graph, self.path_idx, self.mask_idx, self.ends \
            = build_graph(self.path, self.num_ent, self.data['train'], self.max_l, self.tpn)

        self.triples = ddict(list)
        if not os.path.exists(os.path.join(self.path, 'train_triples_{:d}.txt'.format(neg_num_samp))):
            create_train_test_split(self.data, self.graph, self.path)
            

        for split in ['train', 'test', 'valid']:
            self.triples[split] = np.genfromtxt(os.path.join(self.path, split+'_triples_{:d}.txt'.format(neg_num_samp)), \
                                                delimiter=' ', dtype=int)
            if split == 'train':
                self.triples[split] = self.triples[split][:,:3]
            else:
                self.triples[split] = self.triples[split][:,:11]
        


class GTNData(object):
    def __init__(self, args, path, ds):

        self.max_l = args.max_l
        self.tpn = args.tpn
        self.path = os.path.join(path, 'GTN_data', ds)

        with open(self.path+'/node_features.pkl','rb') as f:
            node_features = pickle.load(f)
        
        with open(self.path+'/edges.pkl','rb') as f:
            edges = pickle.load(f)
        
        with open(self.path+'/labels.pkl','rb') as f:
            labels = pickle.load(f)


        if not os.path.exists(self.path+'/node_type.txt'): 
            node_type = {}
            ntype = 0

            if ds == 'acm':
                type0 = np.empty([0], dtype=int)
                for i in range(len(labels)):
                    type0 = np.concatenate((type0, np.array(labels[i])[:,0]))
            else:  
                type0 = set(edges[0].nonzero()[0])
            node_type[ntype] = type0

            for i in range(0, len(edges), 2):
                ntype += 1
                type_i = set(edges[i].nonzero()[1])
                node_type[ntype] = type_i

            with open(self.path+'/node_type.txt', 'w') as f:
                for k,v in node_type.items():
                    for node in v:
                        f.write(str(node) + ' ' + str(k) + '\n')
        else:
            self.type = np.genfromtxt(self.path+'/node_type.txt', delimiter=' ', dtype=int)
            self.type = self.type[self.type[:,0].argsort()][:,1]
            self.ntype = max(self.type) + 1

        self.num_ent = edges[0].shape[0]

        self.feat = torch.from_numpy(node_features).type(torch.FloatTensor)
        self.train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.LongTensor)
        self.train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.LongTensor)
        self.valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.LongTensor)
        self.valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.LongTensor)
        self.test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.LongTensor)
        self.test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.LongTensor)

        self.edges = np.empty((2,0), dtype=int)
        for edge in edges:
            self.edges = np.concatenate((self.edges, np.asarray(edge.nonzero())), axis=1)
        self.edges = self.edges.T
        
        self.triples = ddict(list)
        if not os.path.exists(os.path.join(self.path, 'train_triples_{:d}.txt'.format(neg_num_samp))):
            
            
            np.random.seed(0)
            idx = np.arange(self.edges.shape[0])
            np.random.shuffle(idx)

            num_train = idx[:int(0.8*idx.shape[0])]
            num_val = idx[int(0.8*idx.shape[0]):int(0.9*idx.shape[0])]
            num_test = idx[int(0.9*idx.shape[0]):]

            self.links = dict()
            self.links['train'] = self.edges[num_train]
            self.links['valid'] = self.edges[num_val]
            self.links['test'] = self.edges[num_test]

            self.graph, self.ori_graph, self.path_idx, self.mask_idx, self.ends \
                = build_graph(self.path, self.num_ent, self.links, self.max_l, self.tpn)

            create_train_test_split(self.links, self.ori_graph, self.path, num=self.num_ent, name=args.dataset)
            
        for split in ['train', 'test', 'valid']:
            self.triples[split] = np.genfromtxt(os.path.join(self.path, split+'_triples_{:d}.txt'.format(neg_num_samp)), \
                                                delimiter=' ', dtype=int)
            if split == 'train':
                self.triples[split] = self.triples[split][:,:3]
            else:
                self.triples[split] = self.triples[split][:,:11]


        self.link_train = self.triples['train'][:,:2]
        self.graph, self.ori_graph, self.path_idx, self.mask_idx, self.ends \
            = build_graph(self.path, self.num_ent, self.link_train, self.max_l, self.tpn)


        max_degree = 0
        mean = 0
        for node in self.graph.keys():
            mean += len(self.graph[node])
            if max_degree < len(self.graph[node]):
                max_degree = len(self.graph[node])
        
        self.max_degree = max_degree
        mean /= len(self.graph.keys()) 
        self.pagerank = pagerank(self.graph)
        print('done')

        


class OGB(object):
    def __init__(self, args, path, ds):
        
        self.max_l = args.max_l
        self.tpn = args.tpn
        self.path = os.path.join(path, ds)

        #self.create_sub_graph()


        self.feat = np.load(self.path + '/feat.npy')
        self.feat = torch.FloatTensor(self.feat)
        self.edges = np.load(self.path + '/sub_edges.npy')
        
        self.num_ent = self.feat.shape[0] 
        self.nodes = np.arange(self.num_ent)
        
        self.triples = ddict(list)
        if not os.path.exists(os.path.join(self.path, 'train_triples_{:d}.txt'.format(neg_num_samp))):

            self.links = dict()
            np.random.seed(0)
            idx = np.arange(self.edges.shape[0])
            np.random.shuffle(idx)

            num_train = idx[:int(0.8*idx.shape[0])]
            num_val = idx[int(0.8*idx.shape[0]):int(0.9*idx.shape[0])]
            num_test = idx[int(0.9*idx.shape[0]):]

            self.links['train'] = self.edges[num_train]
            self.links['valid'] = self.edges[num_val]
            self.links['test'] = self.edges[num_test]

            self.graph, self.ori_graph, self.path_idx, self.mask_idx, self.ends \
                = build_graph(self.path, self.num_ent, self.links, self.max_l, self.tpn)

            create_train_test_split(self.links, self.ori_graph, self.path, num=self.num_ent, name=args.dataset)
              
        for split in ['train', 'test', 'valid']:
            self.triples[split] = np.genfromtxt(os.path.join(self.path, split+'_triples_{:d}.txt'.format(neg_num_samp)),delimiter=' ', dtype=int)

            if split == 'train':
                self.triples[split] = self.triples[split][:,:3]
            else:
                self.triples[split] = self.triples[split][:,:11]


        self.link_train = self.triples['train'][:,:2]
        self.graph, self.ori_graph, self.path_idx, self.mask_idx, self.ends \
            = build_graph(self.path, self.num_ent, self.link_train, self.max_l, self.tpn)
    
        self.type = np.genfromtxt(self.path + '/node_type.txt', delimiter='\t', dtype=int)[:,1]
        print('done')



    def sample_node(self, graph, root, num = 100000):

        sub_nodes = set()
        queue = []
        queue.append(root)

        while len(sub_nodes) <= num:
            traverse = queue[0]
            neigh = graph[traverse]            
            ntype = self.ntype[neigh]
        
            sample = []

            n0 = np.where(ntype == 0)[0]
            n0 = neigh[n0]
            n1 = np.where(ntype == 1)[0]
            n1 = neigh[n1]
            
            others = set(neigh) - set(n0) - set(n1)
            sample.extend(list(others))

            if len(n0) < 5:
                sample.extend(list(n0))
            else:
                sample.extend(random.sample(n0, 5))

            if len(n1) < 5:
                sample.extend(list(n1))
            else:
                sample.extend(random.sample(n1, 5))

            queue.extend(sample)
            
            sub_nodes.add(traverse)
            sub_nodes.update(sample) 


            queue.pop(0)

        return list(sub_nodes) 


    def create_sub_graph(self):
        # create subgraph

        data = OGB_MAG(root=self.path, preprocess='metapath2vec')[0]

        labels = data['paper'].y.numpy()
        homo_g = data.to_homogeneous()

        homo_g.edge_index = homo_g.edge_index.T.numpy()
        homo_g.node_type = homo_g.node_type.numpy()
       
        self.feat = homo_g.x.numpy()
        self.edges = homo_g.edge_index.T.numpy()

        self.ntype = homo_g.node_type.numpy()
        

        self.num_ent = self.feat.shape[0]         
        g = ddict(OrderedSet)
        for e in self.edges:
            g[e[0]].add(e[1])
            g[e[1]].add(e[0])

        if not os.path.exists(os.path.join(self.path, 'sub_nodes.npy')):
            root = 1000000
            sub_nodes = self.sample_node(g, root)        
            print(np.asarray(sub_nodes).shape)
            np.save(self.path + '/sub_nodes.npy', np.asarray(sub_nodes))
        else:
            sub_nodes = np.load(self.path + '/sub_nodes.npy')

        ntype = self.ntype[sub_nodes]
        print(Counter(ntype))

        self.feat = self.feat[sub_nodes]

        dict_map = {}
        for i in range(len(sub_nodes)):
            dict_map[sub_nodes[i]] = i

        sub_edges = []
        for n in sub_nodes:
            neigh = g[n]
            sub = [i for i in neigh if i in sub_nodes]
            for s in sub:
                sub_edges.append([dict_map[n], dict_map[s]])
                

        print(np.asarray(sub_edges).shape)

        np.save(self.path + '/sub_edges.npy', np.asarray(sub_edges))
        np.save(self.path + '/feat.npy', self.feat)

        with open(self.path + '/node_type.txt', 'w') as f:
            for i in range(len(ntype)):
                f.write(str(i) + '\t' + str(ntype[i]) + '\n')

        self.labels = dict()
        for n in sub_nodes:
            if n >= labels.shape[0]:
                continue 
            self.labels[dict_map[n]] = labels[n]

        with open(self.path + '/labels.txt', 'w') as f:
            for k in self.labels.keys():
                f.write(str(k) + '\t' + str(self.labels[k]) + '\n')



DATASETS = {
    'fb15k-237': KG,
    'wn18rr': KG,
    'dblp': GTNData,
    'ogb': OGB,
    }


def get_dataset(args, name, path='../data'):
    if name not in DATASETS:
        raise ValueError("Dataset is not supported")
    return DATASETS[name](args, path, name)


if __name__ == "__main__":
    
    args = parse_args()
    args.max_l += 1
    print(args)
    
    np.random.seed(args.seed)
    seed = np.random.choice(100, args.n_runs, replace=False)
    print('Seed: ', seed)

    print('Processing data ...')
    data = get_dataset(args, args.dataset)


