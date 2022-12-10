import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset_str):
    names = ['x', 'y', 'test_x', 'test_y', 'all_x', 'all_y', 'graph']
    
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            objects.append(pkl.load(f))

    x, y, test_x, test_y, all_x, all_y, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    print(x.shape, y.shape, test_x.shape, test_y.shape, all_x.shape, all_y.shape)

    features = sp.vstack((all_x, test_x)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((all_y, test_y))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train, y_val, y_test = np.zeros(labels.shape), np.zeros(labels.shape), np.zeros(labels.shape)
    
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_corpus(dataset_str):
    names = ['x', 'y', 'test_x', 'test_y', 'all_x', 'all_y', 'adj']
    
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            objects.append(pkl.load(f))

    x, y, test_x, test_y, all_x, all_y, adj = tuple(objects)
    print(x.shape, y.shape, test_x.shape, test_y.shape, all_x.shape, all_y.shape)

    features = sp.vstack((all_x, test_x)).tolil()
    labels = np.vstack((all_y, test_y))
    print(len(labels))

    train_idx_orig = parse_index_file(
        "data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = test_x.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(all_x.shape[0], all_x.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train, y_val, y_test = np.zeros(labels.shape), np.zeros(labels.shape), np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size

def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def load_chatbot_corpus():
    category_path = "./data/corpus/wellness_labels.txt"
    answer_path = "./data/corpus/wellness_answer.txt"

    c_f = open(category_path,'r', encoding = "utf-8")
    a_f = open(answer_path,'r', encoding = "utf-8")

    category_lines = c_f.readlines()
    answer_lines = a_f.readlines()

    category = {}
    answer = {}
    for line_num, line_data in enumerate(category_lines):
        data = line_data.split("\n")
        category[line_num]=data[0]

    for line_num, line_data in enumerate(answer_lines):
        data = line_data.split('    ')
        keys = answer.keys()
        if(data[0] in keys):
            answer[data[0]] += [data[1][:-1]]
        else:
            answer[data[0]] =[data[1][:-1]]

    return category, answer