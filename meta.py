import csv

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
from mnl import mlp
from learner import Classifier
from copy import deepcopy
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def proto_loss_spt(logits, y_t, n_support):
    target_cpu = y_t.to('cpu')  # 目标
    input_cpu = logits.to('cpu')  # 输入h特征表达

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cpu)  # [0,1]
    n_classes = len(classes)  # 2类
    n_query = n_support  # 5

    support_idxs = list(map(supp_idxs, classes))  # {0[56789],1[01234]}

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])  # 原型表达 2*2
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[:n_support], classes))).view(-1)
    query_samples = input_cpu[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)  # 2*1*1[0,1]
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query,
                                     1).long()  # [2,5,1] [[[0],[0],[0],[0],[0]][[1],[1],[1],[1],[1]]]

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)  # (2,5)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val, prototypes


def proto_loss_qry(logits, y_t, prototypes):
    target_cpu = y_t.to('cpu')
    input_cpu = logits.to('cpu')

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    n_query = int(logits.shape[0] / n_classes)

    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero(), classes))).view(-1)
    query_samples = input_cpu[query_idxs]

    dists = euclidean_dist(query_samples, prototypes)

    log_p_y0 = F.log_softmax(-dists, dim=1)
    log_p_y = log_p_y0.view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val, target_inds.tolist(), y_hat.tolist(), log_p_y0,query_idxs


class Meta(nn.Module):
    def __init__(self, args, config, config0, dim0, dim1):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = Classifier(config)
        self.net = self.net.to(device)
        self.net1 = mlp(config0, dim0, dim1)
        self.net1 = self.net1.to(device)
        self.meta_optim = optim.Adam([{'params': self.net1.parameters()}, {'params': self.net.parameters()}],
                                     lr=self.meta_lr)
        # self.method = args.method

    def forward_ProtoMAML(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat, gene_feat):
        """
        b: number of tasks
        setsz: the size for each task

        :param x_spt:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_spt:   [b, setsz]
        :param x_qry:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num = len(x_spt)
        querysz = len(y_qry[0])
        losses_s = [0 for _ in range(self.update_step)]
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        # print(self.net.state_dict())

        for i in range(task_num):
            gene = self.net1(gene_feat)
            gene = torch.Tensor(gene[g_spt[i][0]]).to(device)

            feat_spt = torch.Tensor(np.vstack(([feat[g_spt[i][j]][np.array(x)] for j, x in enumerate(n_spt[i])]))).to(
                device)
            feat_qry = torch.Tensor(np.vstack(([feat[g_qry[i][j]][np.array(x)] for j, x in enumerate(n_qry[i])]))).to(
                device)
            logits, _ = self.net(x_spt[i].to(device), c_spt[i].to(device), feat_spt, vars=None)

            loss, _, prototypes = proto_loss_spt(logits, y_spt[i], self.k_spt)
            losses_s[0] += loss
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway] 输入查询集在原来GCN中得到查询集特征表达 计算与原型表达的欧氏距离 得到损失和准确率
                logits_q, _ = self.net(x_qry[i].to(device), c_qry[i].to(device), feat_qry, self.net.parameters())
                loss_q, acc_q, _, _,_,_ = proto_loss_qry(logits_q, y_qry[i], prototypes)
                losses_q[0] += loss_q
                corrects[0] = corrects[0] + acc_q

            # this is the loss and accuracy after the first update
            with torch.no_grad():  # 输入查询集在更新后GCN中得到查询集特征表达 计算与原型表达的欧氏距离 得到损失和准确率
                logits_q, _ = self.net(x_qry[i].to(device), c_qry[i].to(device), feat_qry, fast_weights)
                loss_q, acc_q, _, _,_,_ = proto_loss_qry(logits_q, y_qry[i], prototypes)
                losses_q[1] += loss_q
                corrects[1] = corrects[1] + acc_q

            feat_spt = torch.add(feat_spt, gene)
            feat_qry = torch.add(feat_qry, gene)
            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1  输入支持集在更新后GCN中得到查询集特征表达 计算原型表达 得到损失和准确率
                logits, _ = self.net(x_spt[i].to(device), c_spt[i].to(device), feat_spt, fast_weights)
                loss, _, prototypes = proto_loss_spt(logits, y_spt[i], self.k_spt)
                losses_s[k] += loss
                # 2. compute grad on theta_pi 梯度返回更新
                grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
                # 3. theta_pi = theta_pi - train_lr * grad 参数更新 输入查询集在更新后GCN中得到查询集特征表达 计算与原型表达的欧氏距离 得到损失和准确率
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q, _ = self.net(x_qry[i].to(device), c_qry[i].to(device), feat_qry, fast_weights)

                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q, acc_q, taget_label, pre_label,_,_ = proto_loss_qry(logits_q, y_qry[i], prototypes)
                losses_q[k + 1] += loss_q
                corrects[k + 1] = corrects[k + 1] + acc_q
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        if torch.isnan(loss_q):
            pass
        else:
            # optimize theta parameters
            self.meta_optim.zero_grad()
            loss_q.backward()
            self.meta_optim.step()

        accs = np.array(corrects) / (task_num)

        return accs

    def fine_MAML(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat,
                              gene_feat):
        querysz = len(y_qry[0])
        corrects = [0 for _ in range(self.update_step_test + 1)]
        # finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        x_spt = x_spt[0]
        y_spt = y_spt[0]
        x_qry = x_qry[0]
        y_qry = y_qry[0]
        c_spt = c_spt[0]
        c_qry = c_qry[0]
        n_spt = n_spt[0]
        n_qry = n_qry[0]
        g_spt = g_spt[0]
        g_qry = g_qry[0]

        feat_spt = torch.Tensor(np.vstack(([feat[g_spt[j]][np.array(x)] for j, x in enumerate(n_spt)]))).to(device)
        feat_qry = torch.Tensor(np.vstack(([feat[g_qry[j]][np.array(x)] for j, x in enumerate(n_qry)]))).to(device)

        # 1. run the i-th task and compute loss for k=0
        gene = self.net1(gene_feat)

        for j, _ in enumerate(n_spt):
            gene0 = torch.Tensor(gene[g_spt[j]]).to(device)
        feat_spt = torch.add(feat_spt, gene0)
        feat_qry = torch.add(feat_qry, gene0)


        logits, _ = net(x_spt.to(device), c_spt.to(device), feat_spt)

        loss, _, prototypes = proto_loss_spt(logits, y_spt, self.k_spt)

        logits_q, _ = net(x_qry.to(device), c_qry.to(device), feat_qry, net.parameters())

        loss_q, acc_q,taget_label, pre_label, log_p_y0,query_idxs = proto_loss_qry(logits_q, y_qry, prototypes)
        accs = np.array(acc_q)
        taget_label_list = torch.tensor(np.array(taget_label).flatten())
        pre_label_list = torch.tensor(np.array(pre_label).flatten())
        log_p_y0 = log_p_y0[:, 1].detach()
        return accs,taget_label_list,pre_label_list,log_p_y0,query_idxs



    def forward(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat, gene_feat):
        # if self.method == 'G-Meta':
        accs = self.forward_ProtoMAML(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat,
                                      gene_feat)
        return accs

    def fine(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat, gene_feat):
        # if self.method == 'G-Meta':
        accs,taget_label_list,pre_label_list,log_p_y0,query_idxs = self.fine_MAML(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat,
                                          gene_feat)
        return accs,taget_label_list,pre_label_list,log_p_y0,query_idxs



