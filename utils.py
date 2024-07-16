import torch
import random
from sklearn import metrics
from sklearn.metrics import f1_score
from torchmetrics import AveragePrecision
import numpy as np
import warnings
import dhg
warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 打印超参数
def print_summary(config):
    # Summary of training information
    print('========================================== Training Summary ==========================================')

    print('========================================== Dataset Information ==========================================')
    print(f'dataset_name:{config.dataset_name}')
    print(f'neg_sample:{config.neg_sample}')

    print('========================================== HGNN type ==========================================')
    print(f'hgnn_type:{config.hgnn_type}')

    print('========================================== Params of HNHN ==========================================')
    print(f'edge_alpha:{config.edge_alpha}')
    print(f'node_beta:{config.node_beta}')

    print('========================================== Params of HGNNs ==========================================')
    print(f'initial_dim:{config.initial_dim}')
    print(f'hidden_dim:{config.hidden_dim}')
    print(f'output_dim:{config.output_dim}')

    print('========================================== Params of training==========================================')
    print(f'batch_size:{config.batch_size}')
    print(f'epoch:{config.epoch}')
    print(f'patient_epoch:{config.patient_epoch}')
    print(f'lr:{config.lr}')
    print(f'weight_decay:{config.weight_decay}')
    print(f'drop_out:{config.drop_out}')

    print('========================================== agg method ==========================================')
    print(f'agg_mode: {config.agg_mode}')

    print('========================================== Params of attention agg ==========================================')
    print(f'n_head: {config.n_head}')


# 计算HNHN的权重矩阵和归一化参数
def HNHN_params(Hg, edge_alpha, node_beta):
    if node_beta == 0:
        indices = Hg.D_v.indices()
        values = torch.tensor(torch.ones(Hg.D_v.size(0))).to(device)
        size = Hg.D_v.size()
        Dv_beta = torch.sparse_coo_tensor(indices, values, size)
        De_sum = Hg.D_e_neg_1
    else:
        Dv = Hg.D_v
        Dv.values()[:] = torch.where(Dv.values() == 0, torch.tensor(1).to(device), Dv.values())
        Dv_beta = torch.pow(Dv, node_beta)
        De_sum = torch.sparse.mm(Hg.H_T, Dv_beta)
        De_sum = torch.sparse.sum(De_sum, dim=1)
        indices = torch.cat([De_sum.indices(), De_sum.indices()], dim=0)
        values = De_sum.values()
        size = Hg.D_e.size()
        De_sum = torch.sparse_coo_tensor(indices, values, size)
        De_sum = torch.pow(De_sum, -1)

    if edge_alpha == 0:
        indices = Hg.D_e.indices()
        values = torch.tensor(torch.ones(Hg.D_e.size(0))).to(device)
        size = Hg.D_e.size()
        De_alpha = torch.sparse_coo_tensor(indices, values, size)
        Dv_sum = Hg.D_v_neg_1
    else:
        De = Hg.D_e
        De_alpha = torch.pow(De, edge_alpha)
        Dv_sum = torch.sparse.mm(Hg.H, De_alpha)
        Dv_sum = torch.sparse.sum(Dv_sum, dim=1)
        size = Hg.D_v.size()
        indices = torch.cat([Dv_sum.indices(), Dv_sum.indices()], dim=0)
        values = Dv_sum.values()
        Dv_sum = torch.sparse_coo_tensor(indices, values, size)
        Dv_sum = torch.pow(Dv_sum, -1)

    return De_alpha, De_sum, Dv_beta, Dv_sum

# 给定正样本和负样本，生成标签并打乱标签和超边数据
def generate_labels(pos_edges, neg_edges):
    edges = pos_edges + neg_edges
    labels = [1] * len(pos_edges) + [0] * len(neg_edges)

    random_seed = 42
    random.seed(random_seed)
    random.shuffle(edges)
    random.seed(random_seed)
    random.shuffle(labels)
    return edges, labels


# 拿到数据集, 返回训练集正样本， 训练集、验证集、测试集以及标签
def DataLoader(dataset_name, neg_sample):
    data = torch.load(f'datasets/{dataset_name}/after_split.pt')
    train_pos_edges = data['train_pos_edges']
    valid_pos_edges = data['valid_pos_edges']
    test_pos_edges = data['test_pos_edges']

    if neg_sample == 'pcns':
        train_neg_edges = data['train_pcns_edges']
        valid_neg_edges = data['valid_pcns_edges']
        test_neg_edges = data['test_pcns_edges']
    elif neg_sample == 'sns':
        train_neg_edges = data['train_sns_edges']
        valid_neg_edges = data['valid_sns_edges']
        test_neg_edges = data['test_sns_edges']
    elif neg_sample == 'mns':
        train_neg_edges = data['train_mns_edges']
        valid_neg_edges = data['valid_mns_edges']
        test_neg_edges = data['test_mns_edges']
    elif neg_sample == 'cns':
        train_neg_edges = data['train_cns_edges']
        valid_neg_edges = data['valid_cns_edges']
        test_neg_edges = data['test_cns_edges']

    # # 在测试集中剔除大小为2的超边
    # filter_test_pos_edges = [edge for edge in test_pos_edges if len(edge) != 2]
    # filter_test_neg_edges = [edge for edge in test_neg_edges if len(edge) != 2]

    train_edges, train_labels = generate_labels(train_pos_edges, train_neg_edges)
    valid_edges, valid_labels = generate_labels(valid_pos_edges, valid_neg_edges)
    test_edges, test_labels = generate_labels(test_pos_edges, test_neg_edges)
    # test_edges, test_labels = generate_labels(filter_test_pos_edges, filter_test_neg_edges)

    return train_pos_edges, train_edges, train_labels, valid_edges, valid_labels, test_edges, test_labels

# 计算4个评价指标, 返回4个数字,输入的两个参数是列表类型或者tensor都可以
def measure(labels, preds):
    # 第一个参数是y_true,第二个参数是y_score预测概率（正类的概率）
    # 举例 y_true = [0, 0, 1, 1],y_score = [0.2, 0.4, 0.7, 0.9]
    auc_roc = metrics.roc_auc_score(np.array(labels.cpu()), np.array(preds.cpu()))

    # 第一个参数是y_score预测概率（正类的概率）,第二个参数是标签，都是一维张量
    average_precision = AveragePrecision(task='binary')
    ap = average_precision(preds.clone().detach(), labels.clone().detach()).item()

    # _, predictions = torch.max(preds, dim=1)
    # correct_count = (predictions == labels).sum().item()
    # accuracy = correct_count / len(labels)
    pred_label = (preds.clone().detach() >= 0.5).int()
    correct_count = (pred_label == labels.clone().detach()).sum().item()
    accuracy = correct_count / len(labels)

    f1 = f1_score(labels.cpu(), pred_label.cpu())

    return auc_roc, ap, accuracy, f1

# 计算每个节点的带超边大小的度数, 返回一个长度为节点数目的张量
def get_size_degree(hyperedges, num_v):
    # 计算带超边大小的度值tensor
    Dv_size_list = [0] * num_v
    for edge in hyperedges:
        edge_size = len(edge)
        for node in edge:
            Dv_size_list[node] += edge_size
    Dv_tensor = torch.tensor(Dv_size_list)
    return Dv_tensor

# 根据某个属性值，对节点进行排序，比如度值大小
def sort_edges(props, hyperedges, sort_desc):
    after_sort_hyperedges = []
    for edge in hyperedges:
        edge = list(edge)
        # reverse 参数值改为 True是降序排序，False是升序排序
        sorted_nodes = sorted(edge, key=lambda node: props[node], reverse=sort_desc)
        after_sort_hyperedges.append(tuple(sorted_nodes))
    return after_sort_hyperedges

# 团扩展的邻接矩阵
def getGroundAdj(train_pos_edges, num_v):
    Hg_ground = dhg.Hypergraph(num_v = num_v, e_list=train_pos_edges, merge_op='sum', device=device)
    normal_graph = dhg.Graph.from_hypergraph_clique(Hg_ground, device=device)
    adj = normal_graph.A
    return adj

