
#用A549构建500条的小图，细胞核350条、细胞质150条，直接用之前训练保存下的参数预测acc为85.06%
#模型迁移微调预测
import csv

import torch,os
import numpy as np
from subgraph_data_processing import Subgraphs
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random,sys, pickle
import argparse
import networkx as nx
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from meta0 import Meta
import time
import copy
import psutil
from memory_profiler import memory_usage

os.environ["KMP_DUPLICATE_LIB_OK"]= 'True'
import numpy as np


# 计算常用指标
def compute_indexes(tp, fp, tn, fn):
    SP = tn / (tn+fp)
    precision = tp / (tp+fp)               # 精确率
    recall = tp / (tp+fn)                  # 召回率
    F1 = (2*precision*recall) / (precision+recall)    # F1
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    ACC = (tp + tn) / (tp + fp + fn + tn)
    return precision, recall, F1, SP, mcc,ACC

def collate(samples):
        graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx = map(list, zip(*samples))

        return graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx


def main():
    mem_usage = memory_usage(-1, interval=.5, timeout=1)
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    root = args.data_dir + 'cell_line\\gene\\'
    with open('E:\\duoladuola\\相关论文\\G-PPI-master\\Gm\\results_all.txt', 'a') as f:
        f.write(str(args) + '\n')

    feat = np.load(root + 'features10.npy', allow_pickle=True)
    feat1 = np.load('E:\\duoladuola\\相关论文\\G-PPI-master\\gene\\gene.npy', allow_pickle=True)
    with open(root + 'cell10.pkl', 'rb') as f:
        dgl_graph = pickle.load(f)
        dgl_graph = dgl_graph[0]
    # print(dgl_graph)
    if args.task_setup == 'Disjoint':
        with open(root + 'label_tu10.pkl', 'rb') as f:
            info = pickle.load(f)
    elif args.task_setup == 'Shared':
        # if args.task_mode == 'True':
        # root = root + '/task' + str(args.task_n) + '/'
        with open(root + 'label_tu10.pkl', 'rb') as f:
            info = pickle.load(f)
    total_class = len(np.unique(np.array(list(info.values()))))
    print('There are {} classes '.format(total_class))
    if args.task_setup == 'Disjoint':
        labels_num = args.n_way
    elif args.task_setup == 'Shared':
        labels_num = total_class
    if len(feat.shape) == 2:
        # single graph, to make it compatible to multiple graph retrieval.
        feat = [feat]
    config = [('GraphConv', [feat[0].shape[1], args.hidden_dim])]
    # if args.h > 1:
    # config = config + [('GraphConv', [args.hidden_dim, args.hidden_dim])] * (args.h - 1)
    config = config + [('GraphConv', [args.hidden_dim, args.hidden_dim])]
    config = config + [('Linear', [args.hidden_dim, labels_num])]

    if args.link_pred_mode == 'True':
        config.append(('LinkPred', [True]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    maml = Meta(args, config,feat1[0].shape[1],256,args.hidden_dim1).to(device)
    feat1 = torch.Tensor(feat1[0].astype(float))
    db_test = Subgraphs(root +'55划分', '9test', info, n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=1500, args = args, adjs = dgl_graph, h = args.h)
    print('------ Start Testing ------')
    s_start = time.time()
    max_memory = 0
    # print('maml.state_dict()',maml.state_dict())
    # maml.eval()
    maml.load_state_dict(torch.load(f'base/{args.model}.pt'), strict=False)
    maml.eval()

    db_t = DataLoader(db_test, 1, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn = collate)


    all_batch = []
    pre_label = []
    tar_label = []
    all_log = []



    test_start = time.time()
    for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry in db_t:
        accs, taget_label_list, pre_label_list, log_p_y0,query_idxs = maml.fine(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry,
                                                                          n_spt, n_qry, g_spt, g_qry, feat, feat1)
        list2_values = [tensor.tolist() for tensor in c_qry]
        result = [n_qry[0][i][j] for i, j in enumerate(list2_values[0])]
        # print(result)
        query_idxs = [tensor.tolist() for tensor in query_idxs]
        result = [result[j] for i, j in enumerate(query_idxs)]
        # print(result)
        pre_l = pre_label_list.tolist()
        taget_l = taget_label_list.tolist()
        log_p = log_p_y0.tolist()
        for i in range(len(result)):
            if result[i] not in all_batch:
                all_batch.append(result[i])
                pre_label.append(pre_l[i])
                tar_label.append(taget_l[i])
                all_log.append(log_p[i])
    cnf_matrix = confusion_matrix(tar_label, pre_label)

    auc_score = roc_auc_score(tar_label, all_log)
    # 从混淆矩阵中提取TN和FP
    TN = cnf_matrix[1, 1]  # 真负类别的数量
    FP = cnf_matrix[1, 0]  # 假正类别的数量
    TP = cnf_matrix[0, 0]  # 真负类别的数量
    FN = cnf_matrix[0, 1]  # 假正类别的数量
    precision, recall, f1, sp, mcc, ACC0 = compute_indexes(TP, FP, TN, FN)

    print('Test acc:', str(ACC0))
    test_end = str(time.time() - test_start)
    print('Test Precision:', str(precision))
    print('Test Recall:', str(recall))  # Sn
    print('Test F1:', str(f1))
    print('Test AUC:', str(auc_score))
    print('Test SP:', str(sp))
    print('Test MCC:', str(mcc))
    print('Total Time:', str(time.time() - s_start))
    print('Max Momory:', str(max_memory))

    # torch.save(model_max.state_dict(), './model.pt')
    with open('E:\\duoladuola\\相关论文\\G-PPI-master\\Gm\\results_all.txt', 'a') as f:
        f.write('Test acc:' + str(ACC0) + '\n')
        f.write('Test Precision:' + str(precision) + '\n')
        f.write('Test Recall:' + str(recall) + '\n')
        f.write('Test F1:' + str(f1) + '\n')
        f.write('Test AUC:' + str(auc_score) + '\n')
        f.write('Test SP:' + str(sp) + '\n')
        f.write('Test MCC:' + str(mcc) + '\n')
        f.write('Test Max Momory:' + str(max_memory) + '\n')
        f.write('Test Time:' + test_end + '\n')
        f.write('One Epoch Time:' + str(time.time() - s_start)[:5] + '\n')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='00')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.0005)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--input_dim', type=int, help='input feature dim', default=None)
    argparser.add_argument('--hidden_dim', type=int, help='hidden dim', default=128)
    argparser.add_argument('--hidden_dim1', type=int, help='hidden dim1', default=2)
    argparser.add_argument('--attention_size', type=int, help='dim of attention_size', default=None)
    argparser.add_argument("--data_dir", default='E:\\duoladuola\\相关论文\\G-PPI-master\\', type=str, required=False, help="The input data dir.")
    argparser.add_argument("--no_finetune", default=True, type=str, required=False, help="no finetune mode.")
    argparser.add_argument("--task_setup", default='Shared', type=str, required=False, help="Select from Disjoint or Shared Setup. For Disjoint-Label, single/multiple graphs are both considered.")
    # argparser.add_argument("--method", default='G-Meta', type=str, required=False, help="Use G-Meta")
    argparser.add_argument('--task_n', type=int, help='task number', default=2)
    argparser.add_argument("--task_mode", default='True', type=str, required=False, help="For Evaluating on Tasks")
    argparser.add_argument("--val_result_report_steps", default=50, type=int, required=False, help="validation report")
    argparser.add_argument("--train_result_report_steps", default=50, type=int, required=False, help="training report")
    argparser.add_argument("--num_workers", default=0, type=int, required=False, help="num of workers")
    argparser.add_argument("--batchsz", default=500, type=int, required=False, help="batch size")
    argparser.add_argument("--link_pred_mode", default='False', type=str, required=False, help="For Link Prediction")
    argparser.add_argument("--h", default=1, type=int, required=False, help="neighborhood size")
    argparser.add_argument('--sample_nodes', type=int, help='sample nodes if above this number of nodes', default=1000)
    #参数关系 batchsz/task_num
    args = argparser.parse_args()
    print(args)

    main()
