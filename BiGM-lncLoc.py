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
from sklearn.metrics import confusion_matrix
from meta import Meta
import time
import copy
import psutil
from memory_profiler import memory_usage

os.environ["KMP_DUPLICATE_LIB_OK"]= 'True'
import numpy as np


# 计算常用指标
def compute_indexes(tp, fp, tn, fn):
    precision = tp / (tp+fp)               # 精确率
    recall = tp / (tp+fn)                  # 召回率
    F1 = (2*precision*recall) / (precision+recall)    # F1
    return precision, recall, F1

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
    with open(root + 'results.txt', 'a') as f:
        f.write(str(args) + '\n')

    feat = np.load(root + 'features10.npy', allow_pickle = True)
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
    maml = Meta(args, config,feat1[0].shape[1],args.hidden_dim1,labels_num).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    max_acc = 0
    model_max = copy.deepcopy(maml)
    feat1 = torch.Tensor(feat1[0].astype(float))
    db_train = Subgraphs(root +'549', '8train', info, n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, batchsz=args.batchsz, args = args, adjs = dgl_graph, h = args.h)
    db_val = Subgraphs(root + '549', '8val', info, n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=100, args = args, adjs = dgl_graph, h = args.h)
    db_test = Subgraphs(root +'549', '8test', info, n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=100, args = args, adjs = dgl_graph, h = args.h)
    print('------ Start Training ------')
    s_start = time.time()
    max_memory = 0
    # print('maml.state_dict()',maml.state_dict())
    # maml.eval()
    for epoch in range(args.epoch):
        # print('maml.state_dict()',maml.state_dict())
        db = DataLoader(db_train, args.task_num, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn = collate)
        s_f = time.time()
        for step, (x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry) in enumerate(db):
            nodes_len = 0
            if step >= 1:
                data_loading_time = time.time() - s_r
            else:
                data_loading_time = time.time() - s_f
            s = time.time()
            # x_spt: a list of #task_num tasks, where each task is a mini-batch of k-shot * n_way subgraphs
            # y_spt: a list of #task_num lists of labels. Each list is of length k-shot * n_way int.
            nodes_len += sum([sum([len(j) for j in i]) for i in n_spt])

            accs = maml(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat, feat1)
            max_memory = max(max_memory, float(psutil.virtual_memory().used/(1024**3)))
            if step % args.train_result_report_steps == 0:
                print('Epoch:', epoch + 1, ' Step:', step, ' training acc:', str(accs[-1])[:5], ' time elapsed:', str(time.time() - s)[:5], ' data loading takes:', str(data_loading_time)[:5], ' Memory usage:', str(float(psutil.virtual_memory().used/(1024**3)))[:5])
            s_r = time.time()

        # validation per epoch
        db_v = DataLoader(db_val, 1, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn = collate)
        accs_all_test = []

        for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry in db_v:

            accs = maml(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat, feat1)
            accs_all_test.append(accs)

        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        print('Epoch:', epoch + 1, ' Val acc:', str(accs[-1])[:5])
        if accs[-1] > max_acc:
            max_acc = accs[-1]
            model_max = copy.deepcopy(maml)

    db_t = DataLoader(db_test, 1, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn = collate)
    accs_all_test = []

    precision_all = []
    recall_all = []
    f1_all = []

    test_start = time.time()
    for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry in db_t:
        accs,taget_label_list,pre_label_list = model_max.fine(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat, feat1)
        accs_all_test.append(accs)
        cnf_matrix= confusion_matrix(pre_label_list, taget_label_list)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        precision, recall, F1 = compute_indexes(TP, FP, TN, FN)
        precision_all.append(precision)
        recall_all.append(recall)
        f1_all.append(F1)


    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    Precision = np.array(precision_all).mean().astype(np.float16)
    Recall = np.array(recall_all).mean().astype(np.float16)
    F1 = np.array(f1_all).mean().astype(np.float16)




    print('Test acc:', str(accs)[:5])
    test_end = str(time.time() - test_start)[:5]
    print('Test Precision:', str(Precision)[:5])
    print('Test Recall:', str(Recall)[:5])
    print('Test F1:', str(F1)[:5])
    print('Total Time:', str(time.time() - s_start)[:5])
    print('Max Momory:', str(max_memory)[:5])

    #torch.save(model_max.state_dict(), './model.pt')
    with open(root + 'results.txt', 'a') as f:
        f.write('Test acc:' + str(accs)[:5] + '\n')
        f.write('Test Precision:' + str(Precision)[:5] + '\n')
        f.write('Test Recall:' + str(Recall)[:5] + '\n')
        f.write('Test F1:' + str(F1)[:5] + '\n')
        f.write('Test Max Momory:' + str(max_memory)[:5] + '\n')
        f.write('Test Time:' + test_end + '\n')
        f.write('One Epoch Time:' + str(time.time() - s_start)[:5] + '\n')




#
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=15)
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
    argparser.add_argument('--hidden_dim1', type=int, help='hidden dim1', default=256)
    argparser.add_argument('--attention_size', type=int, help='dim of attention_size', default=None)
    argparser.add_argument("--data_dir", default='E:\\duoladuola\\相关论文\\G-PPI-master\\', type=str, required=False, help="The input data dir.")
    argparser.add_argument("--no_finetune", default=True, type=str, required=False, help="no finetune mode.")
    argparser.add_argument("--task_setup", default='Shared', type=str, required=False, help="Select from Disjoint or Shared Setup. For Disjoint-Label, single/multiple graphs are both considered.")
    # argparser.add_argument("--method", default='G-Meta', type=str, required=False, help="Use G-Meta")
    argparser.add_argument('--task_n', type=int, help='task number', default=2)
    argparser.add_argument("--task_mode", default='True', type=str, required=False, help="For Evaluating on Tasks")
    argparser.add_argument("--val_result_report_steps", default=100, type=int, required=False, help="validation report")
    argparser.add_argument("--train_result_report_steps", default=100, type=int, required=False, help="training report")
    argparser.add_argument("--num_workers", default=0, type=int, required=False, help="num of workers")
    argparser.add_argument("--batchsz", default=500, type=int, required=False, help="batch size")
    argparser.add_argument("--link_pred_mode", default='False', type=str, required=False, help="For Link Prediction")
    argparser.add_argument("--h", default=1, type=int, required=False, help="neighborhood size")
    argparser.add_argument('--sample_nodes', type=int, help='sample nodes if above this number of nodes', default=1000)
    #参数关系 batchsz/task_num
    args = argparser.parse_args()
    print(args)

    main()
